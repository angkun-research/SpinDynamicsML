import os
import sys
import json

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)
from pythonML.AutoEncoderClassNew import Conv1Dminimal


# =========================
# Plot style
# =========================
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["mathtext.rm"] = "Times New Roman"
plt.rcParams["mathtext.it"] = "Times New Roman:italic"
plt.rcParams["mathtext.bf"] = "Times New Roman:bold"


# =========================
# User configuration
# =========================
CSV_PATH = (
    "./data/"
    "triangular_scalar_uniform_Js4_Bdep_Nk250_NJ20_QZ0_unitless.csv"
)

CKPT_CLEAN = (
    "./data/ConvChannel_scalar_LSWT_Js4_Nk250_sigma0.0_unitless.pth"
)
CKPT_NOISY = (
    "./data/ConvChannel_scalar_LSWT_Js4_Nk250_sigma0.1_unitless.pth"
)

METRICS_GAUSSIAN = (
    "./data/"
    "LSWT_gaussian_noise_best20_validation_metrics_epochs200_unitless.csv"
)
METRICS_POISSON = (
    "./data/"
    "LSWT_poisson_noise_best20_validation_metrics_epochs200_unitless.csv"
)

LSWT_CLEAN_PATH = (
    "./data/LSWT_Js4_Bdep_Nk250_test_0_unitless.csv"
)
LSWT_NOISY_PATH = (
    "./data/LSWT_Js4_Bdep_Nk250_test_1_unitless.csv"
)

OUTPUT_DIR = "./plots"

LATENT_DIM = 4
HIDDEN_DIM = 64
HAS_VARIANCE = True
IF_DECODE = True

SIGMA_NOISE_FOR_PLOT = 0.1
NOISE_SEED = 1234

SPLIT_SEED = 1234
TRAIN_RATIO = 0.9
TEST_INDICES = [5]

# Matches PlotData.py: Poisson horizontal coordinate is 10/T.
POISSON_X_SCALE = 10.0

# Fixed LSWT CSV files above must correspond to the selected test sample.
FIGSIZE = (3.5, 6.6)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# Data and model helpers
# =========================
def add_bandwidth_noise(X, sigma, seed=1234, intensity_channel=1):
    rng = np.random.default_rng(seed)
    eps = rng.normal(0.0, 1.0, size=X.shape).astype(np.float32)

    max_vals = X.max(axis=2, keepdims=True)
    min_vals = X.min(axis=2, keepdims=True)
    bandwidth = max_vals - min_vals

    bandwidth[:, intensity_channel:intensity_channel + 1, :] = (
        max_vals[:, intensity_channel:intensity_channel + 1, :]
    )

    return X + sigma * bandwidth * eps


def to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy().astype(np.float32)
    return np.asarray(x, dtype=np.float32)


def load_data(csv_path):
    df = pd.read_csv(csv_path)

    for col in ["data", "disp", "Js"]:
        df[col] = df[col].apply(json.loads)

    disp = np.array(df["disp"].tolist(), dtype=np.float32)
    inten = np.array(df["data"].tolist(), dtype=np.float32)

    X_raw = np.stack([disp, inten], axis=1)
    y = np.array(df["Js"].tolist(), dtype=np.float32)
    y[:, 1] *= y[:, 0]

    return X_raw, y


def load_lswt_disp(csv_path):
    df = pd.read_csv(csv_path)
    return np.asarray(json.loads(df.iloc[0]["disp"]), dtype=np.float32)


def get_test_global_indices(
    n_total,
    train_ratio=TRAIN_RATIO,
    split_seed=SPLIT_SEED,
):
    n_train = int(train_ratio * n_total)
    generator = torch.Generator().manual_seed(split_seed)
    permutation = torch.randperm(n_total, generator=generator).tolist()
    return permutation[n_train:]


def load_model_and_stats(ckpt_path, input_length):
    ckpt = torch.load(
        ckpt_path,
        map_location=DEVICE,
        weights_only=False,
    )

    model = Conv1Dminimal(
        in_channels=2,
        input_length=input_length,
        hidden_dim=HIDDEN_DIM,
        latent_dim=LATENT_DIM,
        has_variance=HAS_VARIANCE,
        ifdecode=IF_DECODE,
    ).to(DEVICE)

    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    return model, to_numpy(ckpt["X_mean"]), to_numpy(ckpt["X_std"])


@torch.no_grad()
def reconstruct_and_predict(model, X_raw_one, X_mean, X_std):
    Xn = (X_raw_one[None, ...] - X_mean) / X_std
    xt = torch.from_numpy(Xn.astype(np.float32)).to(DEVICE)

    out = model(xt)

    recon_normalized = out["x_recon"].detach().cpu().numpy()
    recon_raw = (recon_normalized * X_std + X_mean)[0]

    y_mu = out["mu"][0].detach().cpu().numpy()
    y_var = (
        out["var"][0].detach().cpu().numpy()
        if out["var"] is not None
        else None
    )

    return recon_raw, y_mu, y_var


# =========================
# Panel (a): validation metrics
# =========================
def load_and_summarize(csv_path):
    results = pd.read_csv(csv_path)

    required_columns = {"sigma", "epoch", "mean_variance", "mare"}
    missing = required_columns - set(results.columns)
    if missing:
        raise ValueError(
            f"{csv_path}: missing columns {sorted(missing)}"
        )

    summary = (
        results.groupby("sigma", as_index=False)
        .agg(
            mean_variance_mean=("mean_variance", "mean"),
            mean_variance_std=("mean_variance", "std"),
            mare_mean=("mare", "mean"),
            mare_std=("mare", "std"),
        )
        .sort_values("sigma")
    )

    print(f"\nValidation summary: {csv_path}")
    print(summary.to_string(index=False))
    return summary


def plot_validation_metrics(ax, gaussian, poisson):
    datasets = [
        (gaussian, 1.0, "-", "o", "s", ""),
        (poisson, POISSON_X_SCALE, "--", "^", "d", " (Poisson)"),
    ]

    for summary, xscale, linestyle, var_marker, mare_marker, suffix in datasets:
        x = summary["sigma"].to_numpy() * xscale

        ax.errorbar(
            x,
            summary["mean_variance_mean"],
            yerr=summary["mean_variance_std"],
            color="#1f77b4",
            marker=var_marker,
            markersize=4,
            linestyle=linestyle,
            linewidth=1.5,
            capsize=3,
            label=r"$\langle\sigma^2\rangle$" + suffix,
        )

        ax.errorbar(
            x,
            summary["mare_mean"],
            yerr=summary["mare_std"],
            color="#d62728",
            marker=mare_marker,
            markersize=3,
            linestyle=linestyle,
            linewidth=1.5,
            capsize=3,
            label="MARE" + suffix,
        )

    ax.set(
        xlabel=r"$\sigma_\epsilon$ (or $10/T$ Poisson)",
        ylabel="Error metric",
        yscale="log",
        ylim=(1e-5, 1e-1),
    )
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="best", fontsize=8)


# =========================
# Panel (b): decoder comparison
# =========================
def plot_decoder_comparison(
    ax,
    x_true,
    x_noisy,
    recon_clean,
    recon_noisy,
    lswt_disp_clean,
    lswt_disp_noisy,
):
    k = np.linspace(0.0, 1.0, x_true.shape[-1])

    ax.plot(
        k,
        x_true[0],
        color="#1f77b4",
        label=r"$\varepsilon(\mathbf{k})$ ($\sigma$=0.0)",
        linewidth=2.5,
        alpha=0.8,
    )
    ax.plot(
        k,
        x_noisy[0],
        color="#ff7f0e",
        label=r"$\varepsilon(\mathbf{k})+\epsilon$ ($\sigma$=0.1)",
        linewidth=1.5,
        alpha=0.8,
    )
    ax.plot(
        k,
        recon_clean[0],
        color="#d62728",
        label=r"$\tilde{\varepsilon}(\mathbf{k})$ ($\sigma$=0.0)",
        linewidth=2,
        linestyle="--",
    )
    ax.plot(
        k,
        recon_noisy[0],
        color="#2ca02c",
        label=r"$\tilde{\varepsilon}(\mathbf{k})$ ($\sigma$=0.1)",
        linewidth=1.5,
        alpha=0.8,
    )

    ax.set(
        xlabel=r"$(h,h,0)$",
        ylabel=r"$\omega/|J_1|$",
        xlim=(0.0, 1.0),
        ylim=(0.0, 10.0),
    )
    ax.grid(alpha=0.3)
    ax.legend(loc="upper center", fontsize=8)

    axins = inset_axes(
        ax,
        width="42%",
        height="35%",
        loc="lower center",
        borderpad=1.0,
    )

    axins.plot(
        k,
        x_true[0],
        color="#1f77b4",
        linewidth=2.5,
        alpha=0.9,
        label=r"$J_{\mathrm{true}}$",
    )
    axins.plot(
        np.linspace(0.0, 1.0, len(lswt_disp_clean)),
        lswt_disp_clean,
        color="#d62728",
        linestyle="--",
        linewidth=2,
        label=r"LSWT($\mu_{\sigma=0.0}$)",
    )
    axins.plot(
        np.linspace(0.0, 1.0, len(lswt_disp_noisy)),
        lswt_disp_noisy,
        color="#2ca02c",
        linewidth=1,
        alpha=0.8,
        label=r"LSWT($\mu_{\sigma=0.1}$)",
    )

    axins.set_xlim(0.0, 1.0)
    axins.tick_params(axis="both", labelsize=7)
    axins.legend(
        fontsize=7,
        loc="lower center",
        framealpha=0.9,
    )


def print_predictions(
    test_idx,
    gidx,
    y_true,
    y_mu_clean,
    y_var_clean,
    y_mu_noisy,
    y_var_noisy,
):
    def fmt(values):
        return np.array2string(
            values,
            precision=6,
            separator=", ",
        )

    print(f"\nTest index: {test_idx}; global index: {gidx}")
    print(f"y_true = {fmt(y_true)}")
    print(f"mu_sigma=0.0 = {fmt(y_mu_clean)}")
    if y_var_clean is not None:
        print(f"sqrt(var)_sigma=0.0 = {fmt(np.sqrt(y_var_clean))}")

    print(f"mu_sigma=0.1 = {fmt(y_mu_noisy)}")
    if y_var_noisy is not None:
        print(f"sqrt(var)_sigma=0.1 = {fmt(np.sqrt(y_var_noisy))}")


# =========================
# Main
# =========================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    gaussian_summary = load_and_summarize(METRICS_GAUSSIAN)
    poisson_summary = load_and_summarize(METRICS_POISSON)

    X_raw, y_raw = load_data(CSV_PATH)
    N, C, L = X_raw.shape
    if C != 2:
        raise ValueError("Expected channels [dispersion, intensity].")

    X_noisy_raw = add_bandwidth_noise(
        X_raw,
        sigma=SIGMA_NOISE_FOR_PLOT,
        seed=NOISE_SEED,
    )

    test_global = get_test_global_indices(N)

    model_clean, mean_clean, std_clean = load_model_and_stats(
        CKPT_CLEAN,
        input_length=L,
    )
    model_noisy, mean_noisy, std_noisy = load_model_and_stats(
        CKPT_NOISY,
        input_length=L,
    )

    lswt_disp_clean = load_lswt_disp(LSWT_CLEAN_PATH)
    lswt_disp_noisy = load_lswt_disp(LSWT_NOISY_PATH)

    for test_idx in TEST_INDICES:
        if not 0 <= test_idx < len(test_global):
            print(
                f"Skip test_idx={test_idx}: "
                f"out of range [0, {len(test_global) - 1}]"
            )
            continue

        gidx = test_global[test_idx]
        x_true = X_raw[gidx]
        x_noisy = X_noisy_raw[gidx]

        recon_clean, y_mu_clean, y_var_clean = reconstruct_and_predict(
            model_clean,
            x_true,
            mean_clean,
            std_clean,
        )
        recon_noisy, y_mu_noisy, y_var_noisy = reconstruct_and_predict(
            model_noisy,
            x_noisy,
            mean_noisy,
            std_noisy,
        )

        fig, (ax_a, ax_b) = plt.subplots(
            2,
            1,
            figsize=FIGSIZE,
            constrained_layout=True,
        )
        fig.get_layout_engine().set(h_pad=0.0, hspace=0.0)

        plot_validation_metrics(
            ax_a,
            gaussian_summary,
            poisson_summary,
        )
        plot_decoder_comparison(
            ax_b,
            x_true,
            x_noisy,
            recon_clean,
            recon_noisy,
            lswt_disp_clean,
            lswt_disp_noisy,
        )

        for panel_ax, label in [(ax_a, "a"), (ax_b, "b")]:
            panel_ax.text(
                -0.16,
                1.02,
                label,
                transform=panel_ax.transAxes,
                fontsize=14,
                fontweight="bold",
                ha="left",
                va="bottom",
                clip_on=False,
            )
            panel_ax.tick_params(axis="both", labelsize=11)
            panel_ax.xaxis.label.set_size(12)
            panel_ax.yaxis.label.set_size(10)

        print_predictions(
            test_idx,
            gidx,
            y_raw[gidx],
            y_mu_clean,
            y_var_clean,
            y_mu_noisy,
            y_var_noisy,
        )

        out_name = "fig_lswt_nn_performance.pdf"
        if len(TEST_INDICES) > 1:
            out_name = f"fig_lswt_nn_performance_test{test_idx:04d}.pdf"

        out_path = os.path.join(OUTPUT_DIR, out_name)
        fig.savefig(out_path, format="pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
