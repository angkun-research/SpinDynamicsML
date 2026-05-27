import os
import json
import copy
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from utils import pick_best_device

from NeuralNetworkClass import RegressionMLP, train, test
from NeuralNetworkClass import ConvType1, ConvType2, ConvType3
from NeuralNetworkClass import TransformerType1, ConvTransformer

# -----------------------------
# Config
# -----------------------------
CSV_PATH = './data/triangular_anisotropy_uniform_Js4_B50_Nk20.csv'
MODEL_OUT = './data'
BATCH_SIZE = 100
LR = 1e-3
DW = 1e-5
EPOCHS = 100
HIDDEN_DIM = 64
DEVICE = pick_best_device()

SEED = 1234
TRAIN_RATIO = 0.9

# Choose one model here
#MODEL_NAME = 'conv_transformer'
MODEL_NAME = 'mlp'
#MODEL_NAME = 'conv1'
#MODEL_NAME = 'conv2'
#MODEL_NAME = 'conv3'
#MODEL_NAME = 'transformer1'
print(f"Using model: {MODEL_NAME}")

# Noise settings
# noise_std is the base amplitude
# channel_scale lets each input component have its own amplitude
# For X shape [N, C, L], channel_scale should have length C
CLEAN_NOISE_STD = 0.0
NOISY_TRAIN_NOISE_STD = 0.05
NOISY_TEST_NOISE_STD = 0.05

# -----------------------------
# Reproducibility
# -----------------------------
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(SEED)

# -----------------------------
# Data utilities
# -----------------------------
def load_raw_data(csv_path):
    df = pd.read_csv(csv_path)
    for col in ['data', 'disp', 'Js']:
        df[col] = df[col].apply(json.loads)

    disp = np.array(df['disp'].tolist(), dtype=np.float32)   # [N, L]
    inten = np.array(df['data'].tolist(), dtype=np.float32)   # [N, L]
    X = np.stack([disp, inten], axis=1)                       # [N, 2, L]

    y = np.array(df['Js'].tolist(), dtype=np.float32)
    y[:, 1] *= y[:, 0]

    return X, y

def fit_standardizer(X_train_raw):
    X_mean = X_train_raw.mean(axis=0, keepdims=True)
    X_std = X_train_raw.std(axis=0, keepdims=True) + 1e-8
    return X_mean, X_std

def apply_standardizer(X, X_mean, X_std):
    return (X - X_mean) / X_std

def split_indices(n_samples, train_ratio=0.9, seed=1234):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_samples)
    n_train = int(train_ratio * n_samples)
    return perm[:n_train], perm[n_train:]

def add_componentwise_noise(X, sigma, component_std, seed=1234):
    """
    x_j -> x_j + N(0, sigma^2) * std(x_j)

    X shape: [N, C, L]
    component_std:
      - shape [1, C, 1] for one std per component/channel
      - shape [1, C, L] for one std per component and position
    """
    if sigma <= 0:
        return X.copy()

    rng = np.random.default_rng(seed)
    eps = rng.normal(0.0, sigma, size=X.shape).astype(np.float32)
    return X + eps * component_std

def add_bandwidth_noise(X, sigma, seed=1234):
    rng = np.random.default_rng(seed)
    eps = rng.normal(0.0, 1.0, size=X.shape).astype(np.float32)
    bandwidth = X.max(axis=2, keepdims=True) - X.min(axis=2, keepdims=True)
    return X + sigma * bandwidth * eps

def make_dataset(X, y):
    X_t = torch.from_numpy(X.astype(np.float32))
    y_t = torch.from_numpy(y.astype(np.float32))
    return TensorDataset(X_t, y_t)

def make_loader(dataset, batch_size, shuffle):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)

# -----------------------------
# Model factory
# -----------------------------
def build_model(model_name, input_dim, input_length, output_dim, hidden_dim):
    if model_name == 'mlp':
        return RegressionMLP(in_channels=input_dim, input_length=input_length, hidden_dim=hidden_dim, output_dim=output_dim).to(DEVICE)
    if model_name == 'conv1':
        return ConvType1(in_channels=input_dim, input_length=input_length, hidden_dim=hidden_dim, output_dim=output_dim).to(DEVICE)
    if model_name == 'conv2':
        return ConvType2(in_channels=input_dim, input_length=input_length, hidden_dim=hidden_dim, output_dim=output_dim).to(DEVICE)
    if model_name == 'conv3':
        return ConvType3(in_channels=input_dim, input_length=input_length, hidden_dim=hidden_dim, output_dim=output_dim).to(DEVICE)
    if model_name == 'transformer1':
        return TransformerType1(in_channels=input_dim, input_length=input_length, hidden_dim=hidden_dim, output_dim=output_dim).to(DEVICE)
    if model_name == 'conv_transformer':
        return ConvTransformer(in_channels=input_dim, input_length=input_length, hidden_dim=hidden_dim, output_dim=output_dim).to(DEVICE)

    raise ValueError(f"Unknown model_name: {model_name}")

# -----------------------------
# Experiment runner
# -----------------------------
def run_experiment(
    X_train_clean,
    y_train,
    X_test_clean,
    y_test,
    train_noise_std,
    test_noise_std_list,
    channel_scale,
    model_name,
    seed,
):
    """
    Train one model on either clean or noisy training data,
    then evaluate on multiple test noise conditions.
    """
    X_train_in = add_noise(X_train_clean, noise_std=train_noise_std, channel_scale=channel_scale, seed=seed + 1)
    train_ds = make_dataset(X_train_in, y_train)
    train_loader = make_loader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    input_dim = X_train_in.shape[1]
    input_length = X_train_in.shape[2]
    output_dim = y_train.shape[1]

    model = build_model(model_name, input_dim, input_length, output_dim, HIDDEN_DIM)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=DW)

    print(f"\n=== Training model={model_name}, train_noise_std={train_noise_std} ===")
    print(f"Train samples: {len(train_ds)}")

    prev_updates = 0
    for epoch in range(1, EPOCHS + 1):
        print(f"Epoch {epoch}/{EPOCHS} — device={DEVICE}")
        prev_updates = train(model, train_loader, optimizer, prev_updates, writer=None, device=DEVICE)

    results = {}
    model.eval()

    for test_noise_std in test_noise_std_list:
        X_test_in = add_noise(X_test_clean, noise_std=test_noise_std, channel_scale=channel_scale, seed=seed + 100 + int(test_noise_std * 1000))
        test_ds = make_dataset(X_test_in, y_test)
        test_loader = make_loader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

        print(f"\nTesting with test_noise_std={test_noise_std}")
        test(model, test_loader, cur_step=0, writer=None, device=DEVICE)

        # Optional manual metric collection
        with torch.no_grad():
            preds = []
            targets = []
            for xb, yb in test_loader:
                xb = xb.to(DEVICE)
                yb = yb.to(DEVICE)
                preds.append(model(xb).cpu())
                targets.append(yb.cpu())
            preds = torch.cat(preds, dim=0)
            targets = torch.cat(targets, dim=0)

            mse = torch.mean((preds - targets) ** 2).item()
            mae = torch.mean(torch.abs(preds - targets)).item()

        results[test_noise_std] = {
            "mse": mse,
            "mae": mae,
        }

    return model, results

def run_experiment(
    X_train_raw,
    y_train,
    X_test_raw,
    y_test,
    X_mean,
    X_std,
    component_std,
    train_sigma,
    test_sigma_list,
    model_name,
    seed,
):
    """
    Train on either clean or noisy data, then test on clean and noisy data.

    Parameters
    ----------
    X_train_raw : np.ndarray
        Raw training inputs, shape [N, C, L].
    y_train : np.ndarray
        Training targets.
    X_test_raw : np.ndarray
        Raw test inputs, shape [N, C, L].
    y_test : np.ndarray
        Test targets.
    X_mean, X_std : np.ndarray
        Standardization statistics fitted once and reused everywhere.
    component_std : np.ndarray
        Per-component noise scale, usually computed from raw training data.
        Shape can be [1, C, 1] or [1, C, L].
    train_sigma : float
        Noise level used for training inputs.
    test_sigma_list : list[float]
        Noise levels used for test inputs.
    model_name : str
        Which model to build.
    seed : int
        Base seed for reproducible noise.
    """
    # Build training inputs
    X_train_noisy_raw = add_componentwise_noise(
        X_train_raw,
        sigma=train_sigma,
        component_std=component_std,
        seed=seed + 1,
    )
    X_train_in = apply_standardizer(X_train_noisy_raw, X_mean, X_std)

    train_ds = make_dataset(X_train_in, y_train)
    train_loader = make_loader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    input_dim = X_train_in.shape[1]
    input_length = X_train_in.shape[2]
    output_dim = y_train.shape[1]

    model = build_model(model_name, input_dim, input_length, output_dim, HIDDEN_DIM)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=DW)

    print(f"\n=== Training model={model_name}, train_sigma={train_sigma} ===")
    print(f"Train samples: {len(train_ds)}")
    print(f"Input shape after noise/standardization: {X_train_in.shape}")

    prev_updates = 0
    for epoch in range(1, EPOCHS + 1):
        print(f"Epoch {epoch}/{EPOCHS} — device={DEVICE}")
        prev_updates = train(model, train_loader, optimizer, prev_updates, writer=None, device=DEVICE)

    results = {}
    model.eval()

    for test_sigma in test_sigma_list:
        X_test_noisy_raw = add_componentwise_noise(
            X_test_raw,
            sigma=test_sigma,
            component_std=component_std,
            seed=seed + 100 + int(test_sigma * 1000),
        )
        X_test_in = apply_standardizer(X_test_noisy_raw, X_mean, X_std)

        test_ds = make_dataset(X_test_in, y_test)
        test_loader = make_loader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

        print(f"\nTesting with test_sigma={test_sigma}")
        test(model, test_loader, cur_step=0, writer=None, device=DEVICE)

        with torch.no_grad():
            preds = []
            targets = []
            for xb, yb in test_loader:
                xb = xb.to(DEVICE)
                yb = yb.to(DEVICE)
                preds.append(model(xb).cpu())
                targets.append(yb.cpu())

            preds = torch.cat(preds, dim=0)
            targets = torch.cat(targets, dim=0)

            mse = torch.mean((preds - targets) ** 2).item()
            mae = torch.mean(torch.abs(preds - targets)).item()

        results[test_sigma] = {
            "mse": mse,
            "mae": mae,
        }

        print(f"test_sigma={test_sigma}: mse={mse:.6e}, mae={mae:.6e}")

    return model, results

# -----------------------------
# Main
# -----------------------------
X, y = load_raw_data(CSV_PATH)

print("Raw X shape:", X.shape)
print("Raw y shape:", y.shape)

train_idx, test_idx = split_indices(len(X), train_ratio=TRAIN_RATIO, seed=SEED)

X_train_raw = X[train_idx]
y_train = y[train_idx]
X_test_raw = X[test_idx]
y_test = y[test_idx]

# one shared standardization transform
X_mean, X_std = fit_standardizer(X_train_raw)

# component-wise raw std for noise amplitude
component_std = X_std
# if you only want one std per channel, use:
# component_std = X_train_raw.std(axis=(0, 2), keepdims=True).reshape(1, X.shape[1], 1) + 1e-8

# clean data
X_train_clean = apply_standardizer(X_train_raw, X_mean, X_std)
X_test_clean = apply_standardizer(X_test_raw, X_mean, X_std)

# noisy train / noisy test in raw space, then standardize with the same transform
X_train_noisy_raw = add_componentwise_noise(
    X_train_raw, sigma=NOISY_TRAIN_NOISE_STD, component_std=component_std, seed=SEED + 1
)
X_test_noisy_raw = add_componentwise_noise(
    X_test_raw, sigma=NOISY_TEST_NOISE_STD, component_std=component_std, seed=SEED + 2
)

X_train_noisy = apply_standardizer(X_train_noisy_raw, X_mean, X_std)
X_test_noisy = apply_standardizer(X_test_noisy_raw, X_mean, X_std)

print("Standardized X_train range:", X_train_clean.min(), X_train_clean.max())
print("Standardized X_test range:", X_test_clean.min(), X_test_clean.max())
print("y range:", y.min(), y.max())

# Experiment 1:
# train on clean, test on clean and noisy
model_clean, results_clean_train = run_experiment(
    X_train_raw=X_train_raw,
    y_train=y_train,
    X_test_raw=X_test_raw,
    y_test=y_test,
    X_mean=X_mean,
    X_std=X_std,
    component_std=component_std,
    train_sigma=0.0,
    test_sigma_list=[0.0, NOISY_TEST_NOISE_STD],
    model_name=MODEL_NAME,
    seed=SEED,
)

# save clean model
torch.save({
    "epoch": EPOCHS,
    "model_state": model_clean.state_dict(),
    "X_mean": X_mean,
    "X_std": X_std,
}, os.path.join(MODEL_OUT, f"{MODEL_NAME}_anisotropy_LSWT_Js4_epoch{EPOCHS}_clean.pth"))
print("Training finished. Models saved to:", MODEL_OUT)

# Experiment 2:
# train on noisy, test on clean and noisy
model_noisy, results_noisy_train = run_experiment(
    X_train_raw=X_train_raw,
    y_train=y_train,
    X_test_raw=X_test_raw,
    y_test=y_test,
    X_mean=X_mean,
    X_std=X_std,
    component_std=component_std,
    train_sigma=NOISY_TRAIN_NOISE_STD,
    test_sigma_list=[0.0, NOISY_TEST_NOISE_STD],
    model_name=MODEL_NAME,
    seed=SEED + 10,
)


# Summary
summary = {
    "clean_train": results_clean_train,
    "noisy_train": results_noisy_train,
}
print("\n=== Summary ===")
print(summary)

# save noisy model
torch.save({
    "epoch": EPOCHS,
    "model_state": model_noisy.state_dict(),
    "X_mean": X_mean,
    "X_std": X_std,
}, os.path.join(MODEL_OUT, f"{MODEL_NAME}_anisotropy_LSWT_Js4_epoch{EPOCHS}_noisy.pth"))

print("Training finished. Models saved to:", MODEL_OUT)
