import os
import json
import sys

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


# Make repository-root imports available when running HPCtemp/*.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pythonML.AutoEncoderClassNew import Conv1Dminimal
from pythonML.utils import add_bandwidth_noise, apply_standardizer

# ============================================================
# Configuration
# ============================================================

CSV_PATH = "./data/HaoResults/Hao_anisotropy_Js5_S6tensor_Filter20_range1.csv"

# If your training script printed a different Nk value, update it here.
CKPT_PATH = "./data/ConvChannel_tensor_THED_Js5_Nk12_Ne20_sigma0.0_epoch200.pth"

OUT_DIR = "./plots"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Must match AE_Lanczos_variance.py
N_PARAMETERS = 5
N_COMPONENTS = 6
N_ENERGIES = 20
HIDDEN_DIM = 64

# AE_Lanczos_variance.py currently uses these values.
# Despite the filename, has_variance=False in that script.
HAS_VARIANCE = False
IF_DECODE = False

# Match the training split
TRAIN_RATIO = 0.9
SPLIT_SEED = 1234

# Evaluate on "test", "train", or "all"
EVAL_SPLIT = "test"

# Match the noisy training input
NOISE_SIGMA = 0.0
NOISE_SEED = 1234

BATCH_SIZE = 512

LABEL_LIST = [
    r"$J_1^0$",
    r"$J_1^0\Delta$",
    r"$J_{\pm\pm}$",
    r"$J_{z\pm}$",
    r"$J_2^0$",
]


# ============================================================
# Plot style
# ============================================================

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["mathtext.rm"] = "Times New Roman"
plt.rcParams["mathtext.it"] = "Times New Roman:italic"
plt.rcParams["mathtext.bf"] = "Times New Roman:bold"


# ============================================================
# Data handling
# ============================================================

def load_data(csv_path):
    """
    Load and reshape the Lanczos dataset exactly as in
    AE_Lanczos_variance.py.

    Raw CSV layout:
        dispersion: (N, n_components * n_k * n_energies)
        intensity:  (N, n_components * n_k * n_energies)

    Model layout:
        X: (N, 2 * n_components * n_energies, n_k)
    """
    df = pd.read_csv(csv_path)

    for column in ["data", "disp", "Js"]:
        df[column] = df[column].apply(json.loads)

    dispersion = np.asarray(df["disp"].tolist(), dtype=np.float32)
    intensity = np.asarray(df["data"].tolist(), dtype=np.float32)
    y = np.asarray(df["Js"].tolist(), dtype=np.float32)

    n_samples = len(df)
    n_k = dispersion.shape[1] // N_COMPONENTS // N_ENERGIES

    # (N, components, Nk, energies)
    dispersion_4d = dispersion.reshape(n_samples, N_COMPONENTS, n_k, N_ENERGIES)
    intensity_4d = intensity.reshape(n_samples, N_COMPONENTS, n_k, N_ENERGIES)

    # Move energy before momentum:
    # (N, components, energies, Nk)
    dispersion_channels = np.transpose(
        dispersion_4d, (0, 1, 3, 2)
    ).reshape(n_samples, N_COMPONENTS * N_ENERGIES, n_k)

    intensity_channels = np.transpose(
        intensity_4d, (0, 1, 3, 2)
    ).reshape(n_samples, N_COMPONENTS * N_ENERGIES, n_k)

    # Concatenate all dispersion channels followed by all intensity channels:
    # (N, 240, Nk) for N_COMPONENTS=6 and N_ENERGIES=20.
    X = np.concatenate(
        [dispersion_channels, intensity_channels],
        axis=1,
    )

    if N_PARAMETERS == 4:
        y = np.delete(y, 3, axis=1)  # Remove the 4th column (0-based index)
    # Match training: discard rows containing zero-valued parameters.
    mask = (y != 0).all(axis=1)
    X = X[mask]
    y = y[mask]

    # Match training target convention.
    y[:, 1] *= y[:, 0]

    print(f"Total CSV samples: {n_samples}")
    print(f"Filtered samples: {len(y)}")
    print(f"Input shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Nk={n_k}")

    return X, y


def split_indices(n_total, train_ratio=0.9, split_seed=1234):
    """
    Reproduce torch.utils.data.random_split after set_seed(1234)
    in AE_Lanczos_variance.py.
    """
    n_train = int(train_ratio * n_total)

    generator = torch.Generator().manual_seed(split_seed)
    permutation = torch.randperm(
        n_total,
        generator=generator,
    ).cpu().numpy()

    train_indices = permutation[:n_train]
    test_indices = permutation[n_train:]

    return train_indices, test_indices


# ============================================================
# Model handling
# ============================================================

def to_numpy(value):
    if isinstance(value, np.ndarray):
        return value.astype(np.float32)

    if torch.is_tensor(value):
        return value.detach().cpu().numpy().astype(np.float32)

    return np.asarray(value, dtype=np.float32)


def load_model_and_stats(checkpoint_path, input_shape):
    """
    Rebuild Conv1Dminimal using the dimensions stored in the
    preprocessed input and load the trained checkpoint.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found:\n{checkpoint_path}\n\n"
            "Check the Nk value in CKPT_PATH. The exact checkpoint name "
            "is printed by AE_Lanczos_variance.py after training."
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=False,
    )

    _, input_channels, input_length = input_shape

    model = Conv1Dminimal(
        in_channels=input_channels,
        input_length=input_length,
        hidden_dim=HIDDEN_DIM,
        latent_dim=N_PARAMETERS,
        has_variance=HAS_VARIANCE,
        ifdecode=IF_DECODE,
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()

    X_mean = to_numpy(checkpoint["X_mean"])
    X_std = to_numpy(checkpoint["X_std"])

    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Model input channels: {input_channels}")
    print(f"Model input length: {input_length}")
    print(f"Device: {DEVICE}")

    return model, X_mean, X_std


@torch.no_grad()
def predict(model, X_raw, X_mean, X_std, batch_size=512):
    """
    Standardize input using checkpoint statistics and predict the
    five latent Hamiltonian parameters.
    """
    X_standardized = apply_standardizer(X_raw, X_mean, X_std)

    predictions = []

    for start in range(0, len(X_standardized), batch_size):
        end = min(start + batch_size, len(X_standardized))

        X_batch = torch.from_numpy(
            X_standardized[start:end].astype(np.float32)
        ).to(DEVICE)

        output = model(X_batch)

        # Conv1Dminimal returns a dictionary.
        if isinstance(output, dict):
            prediction = output["mu"]
        else:
            prediction = output

        predictions.append(prediction.detach().cpu().numpy())

    return np.concatenate(predictions, axis=0)


# ============================================================
# Metrics and plotting
# ============================================================

def r2_score_np(y_true, y_pred):
    residual_sum = np.sum((y_true - y_pred) ** 2)
    total_sum = np.sum((y_true - np.mean(y_true)) ** 2)

    if total_sum < 1e-12:
        return np.nan

    return 1.0 - residual_sum / total_sum


def add_error_inset(ax, errors, bins=35):
    inset = inset_axes(ax,width="42%",height="42%",loc="lower right",borderpad=1.2)

    inset.hist(errors,bins=bins,color="#4c72b0",alpha=0.85,edgecolor="white")
    inset.axvline(0.0,color="red",linestyle="--",linewidth=1.0)

    inset.set_xlabel("Error (pred - true)",fontsize=8)
    inset.set_ylabel("Density",fontsize=8)
    inset.grid(alpha=0.15)

    mean_error = np.mean(errors)
    std_error = np.std(errors)
    mae = np.mean(np.abs(errors))
    error_width = 12 * std_error
    shift_center = mean_error*5
    inset.set_xlim(shift_center - error_width, shift_center + error_width)
    inset.set_xticks([])
    inset.set_yticks([])

    inset.text(
        0.05,
        0.95,
        (
            f"mean={mean_error:.3f}\n"
            f"std={std_error:.3f}\n"
            f"mae={mae:.3f}"
        ),
        transform=inset.transAxes,
        va="top",
        fontsize=9,
        bbox=dict(
            boxstyle="round",
            facecolor="white",
            alpha=0.6,
        ),
    )


def plot_parity_with_error_insets(
    y_true,
    y_pred,
    labels,
    output_path,
):
    n_parameters = y_true.shape[1]

    figure, axes = plt.subplots(
        1,
        n_parameters,
        figsize=(4.2 * n_parameters, 4.2),
        constrained_layout=True,
    )

    # Ensure axes is iterable even if configuration changes to one parameter.
    axes = np.atleast_1d(axes)

    for parameter_index in range(n_parameters):
        ax = axes[parameter_index]

        true_values = y_true[:, parameter_index]
        predicted_values = y_pred[:, parameter_index]
        errors = predicted_values - true_values

        lower = min(true_values.min(), predicted_values.min())
        upper = max(true_values.max(), predicted_values.max())

        padding = 0.05 * (upper - lower + 1e-8)
        lower -= padding
        upper += padding

        ax.scatter(
            true_values,
            predicted_values,
            s=12,
            alpha=0.45,
            color="#1f77b4",
            linewidths=0,
        )

        ax.plot(
            [lower, upper],
            [lower, upper],
            "r--",
            linewidth=1.4,
        )

        ax.set_xlim(lower, upper)
        ax.set_ylim(lower, upper)
        ax.set_xlabel("True", fontsize=12)
        ax.set_ylabel("Predicted", fontsize=12)
        ax.set_title(labels[parameter_index], fontsize=14)
        ax.grid(alpha=0.25)

        mae = np.mean(np.abs(errors))
        rmse = np.sqrt(np.mean(errors ** 2))
        r2 = r2_score_np(true_values, predicted_values)

        ax.text(
            0.04,
            0.96,
            (
                f"MAE={mae:.4f}\n"
                f"RMSE={rmse:.4f}\n"
                f"$R^2$={r2:.4f}"
            ),
            transform=ax.transAxes,
            va="top",
            fontsize=12,
            bbox=dict(
                boxstyle="round",
                facecolor="white",
                alpha=0.8,
            ),
        )

        add_error_inset(ax, errors)

    figure.savefig(
        output_path,
        format="pdf",
        bbox_inches="tight",
    )
    plt.close(figure)


# ============================================================
# Main
# ============================================================

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    X_raw, y = load_data(CSV_PATH)

    train_indices, test_indices = split_indices(
        len(X_raw),
        train_ratio=TRAIN_RATIO,
        split_seed=SPLIT_SEED,
    )

    if EVAL_SPLIT == "train":
        evaluation_indices = train_indices
    elif EVAL_SPLIT == "test":
        evaluation_indices = test_indices
    elif EVAL_SPLIT == "all":
        evaluation_indices = np.arange(len(X_raw))
    else:
        raise ValueError(
            f"Unknown EVAL_SPLIT={EVAL_SPLIT!r}. "
            "Choose 'train', 'test', or 'all'."
        )

    X_eval = X_raw[evaluation_indices].copy()
    y_eval = y[evaluation_indices].copy()

    X_eval = add_bandwidth_noise(X_eval,sigma=NOISE_SIGMA,seed=NOISE_SEED)

    model, X_mean, X_std = load_model_and_stats(CKPT_PATH,input_shape=X_raw.shape,)

    y_pred = predict(model,X_eval,X_mean,X_std,batch_size=BATCH_SIZE,)

    output_name = (
        f"Js5_sigma{NOISE_SIGMA}_{EVAL_SPLIT}_error.pdf"
    )
    output_path = os.path.join(OUT_DIR, output_name)

    plot_parity_with_error_insets(
        y_true=y_eval,
        y_pred=y_pred,
        labels=LABEL_LIST,
        output_path=output_path,
    )

    errors = y_pred - y_eval

    print(f"Saved figure: {output_path}")
    print(f"Evaluation split: {EVAL_SPLIT}")
    print(f"Evaluation samples: {len(evaluation_indices)}")
    print(f"Noise sigma: {NOISE_SIGMA}")

    print("\nParameter metrics:")
    for parameter_index, label in enumerate(LABEL_LIST):
        parameter_errors = errors[:, parameter_index]
        mae = np.mean(np.abs(parameter_errors))
        rmse = np.sqrt(np.mean(parameter_errors ** 2))
        r2 = r2_score_np(
            y_eval[:, parameter_index],
            y_pred[:, parameter_index],
        )

        print(
            f"{label}: "
            f"MAE={mae:.6f}, "
            f"RMSE={rmse:.6f}, "
            f"R2={r2:.6f}"
        )


if __name__ == "__main__":
    main()