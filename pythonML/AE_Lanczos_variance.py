import os
import json
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd

from AutoEncoderClassNew import train, test
from AutoEncoderClassNew import FCautoencoder, Conv1Dminimal

# --- Config ---
CSV_PATH = './data/HaoResults/Hao_anisotropy_Js4_uniform_trace_Filter100_Bsats.csv'
BATCH_SIZE = 100
LR = 1e-3
DW = 1e-5
EPOCHS = 200 #100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_OUT = './data'
NumOfJ = 4 #8 #10
mag_recon_loss = 0.0
has_variance = True #False
ifdecode = True #False #True
sigma = 0.0 # noise
#os.makedirs(MODEL_OUT, exist_ok=True)

# --- Load data (same logic as RegressionTest.py) ---
def load_raw_data(csv_path):
    df = pd.read_csv(csv_path)
    # Hao's data format
    for col in ['data', 'disp', 'Js']:
        df[col] = df[col].apply(json.loads)
    disp = np.array(df['disp'].tolist(), dtype=np.float32)
    inten = np.array(df['data'].tolist(), dtype=np.float32)
    y = np.array(df['Js'].tolist(), dtype=np.float32)
    print(f"Total samples: {len(df)}")
    #X = np.stack([disp, inten], axis=1)

    y = np.delete(y, 3, axis=1)   # remove 4th column (0-based index), shape becomes (N, 4)
    N = len(df)
    n_comp = 6 #6
    n_e = 20 #3 #20
    n_k = disp.shape[1] // n_comp // n_e
    # Recover (N, 6, n_k, n_e)
    disp4 = disp.reshape(N, n_comp, n_k, n_e)
    inten4 = inten.reshape(N, n_comp, n_k, n_e)
    # Move energy before momentum, then flatten (component, energy) -> channels
    # (N, 6, n_e, n_k) -> (N, 6 * n_e, n_k)
    # disp_ch = np.transpose(disp4, (0, 1, 3, 2)).reshape(N, n_comp * n_e, n_k)
    # data_ch = np.transpose(inten4, (0, 1, 3, 2)).reshape(N, n_comp * n_e, n_k)
    # Keep only one component (component 0)
    disp4 = disp4[:, 0:1, :, :]   # shape: (N, 1, n_k, n_e)
    inten4 = inten4[:, 0:1, :, :] # shape: (N, 1, n_k, n_e)
    disp_ch = np.transpose(disp4, (0, 1, 3, 2)).reshape(N, 1 * n_e, n_k)
    data_ch = np.transpose(inten4, (0, 1, 3, 2)).reshape(N, 1 * n_e, n_k)
    X = np.concatenate([disp_ch, data_ch], axis=1) # (N, 12 * n_e, n_k)

    mask = (y != 0).all(axis=1) 
    X = X[mask]
    y = y[mask]
    print(f"Filtered samples: {len(y)}")
    print(f"X shape: {X.shape}, y shape: {y.shape}")

    y[:, 1] *= y[:, 0]
    return X, y


def fit_standardizer(X_train_raw):
    X_mean = X_train_raw.mean(axis=0, keepdims=True)
    X_std = X_train_raw.std(axis=0, keepdims=True) + 1e-8
    return X_mean, X_std

def apply_standardizer(X, X_mean, X_std):
    return (X - X_mean) / X_std

def add_bandwidth_noise(X, sigma, seed=1234, intensity_channel=1):
    rng = np.random.default_rng(seed)
    eps = rng.normal(0.0, 1.0, size=X.shape).astype(np.float32)
    max_vals = X.max(axis=2, keepdims=True)
    min_vals = X.min(axis=2, keepdims=True)
    # default: bandwidth = max - min (used for dispersion)
    bandwidth = max_vals - min_vals
    # for intensity channel use max only (min is effectively 0)
    #bandwidth[:, intensity_channel:intensity_channel+1, :] = max_vals[:, intensity_channel:intensity_channel+1, :]
    return X + sigma * bandwidth * eps

def make_dataset(X, y):
    return TensorDataset(
        torch.from_numpy(X.astype(np.float32)),
        torch.from_numpy(y.astype(np.float32)),
    )

def make_loader(dataset, batch_size, shuffle):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


X, y = load_raw_data(CSV_PATH)
X_mean, X_std = fit_standardizer(X)
X_noisy = apply_standardizer(add_bandwidth_noise(X, sigma, seed=1234), X_mean, X_std)
dataset = make_dataset(X_noisy, y)
# Convert to tensors and dataloaders
# X_t = torch.from_numpy(X)
# y_t = torch.from_numpy(y)
# dataset = TensorDataset(X_t, y_t)

n = len(dataset)
print(f"Total samples: {n}")
n_train = int(0.9 * n)
n_val = n - n_train
train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

# --- Model ---
input_dim = X.shape[1]
relative_loss = False
in_channels = X.shape[1]  # number of channels for Conv1DEncoder
#print(X.shape)
# model = FCautoencoder(input_dim=input_dim, hidden_dim=64, latent_dim=NumOfJ, 
#     has_variance=has_variance, ifdecode=ifdecode).to(DEVICE)
model = Conv1Dminimal(in_channels=in_channels, input_length=X.shape[2], hidden_dim=64, latent_dim=NumOfJ,
    has_variance=has_variance, ifdecode=ifdecode).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=DW)
# print number of parameters
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Model has {n_params} trainable parameters.')

prev_updates = 0
for epoch in range(1, EPOCHS + 1):
    print(f"Epoch {epoch}/{EPOCHS} — device={DEVICE}")
    prev_updates = train(model, train_loader, optimizer, prev_updates, has_variance=has_variance,
        latent_dim=NumOfJ, batch_size=BATCH_SIZE, device=DEVICE)
    test(model, val_loader, has_variance=has_variance, cur_step=epoch, latent_dim=NumOfJ, batch_size=BATCH_SIZE, device=DEVICE)

#save checkpoint at only the final epoch
# torch.save({
#     "epoch": EPOCHS,
#     "model_state": model.state_dict(),
#     "optimizer_state": optimizer.state_dict(),
#     "X_mean": X_mean,
#     "X_std": X_std,
# }, os.path.join(MODEL_OUT, f"ConvChannel_anisotropy_Lanczos_Js{NumOfJ}_epoch{EPOCHS}_N{N}.pth"))

# print("Training finished. Models saved to:", MODEL_OUT)


# ---- Print a single test example: true Js vs predicted Js ----
model.eval()
idx = 0  # change this to inspect a different validation example

#x_one, j_true = train_ds[1] #val_ds[:3]  # val_ds is a Subset; returns (X, Js)
#x_one = x_one.unsqueeze(0).to(DEVICE)  # add batch dim: (1, input_dim)
x_batch, j_true = train_ds[:3]   # x_batch: (3, 2, L)
x_batch = x_batch.to(DEVICE)     # no unsqueeze

with torch.no_grad():
    out = model.encode(x_batch)

j_pred = out.squeeze(0).detach().cpu().numpy()
j_true_np = j_true.detach().cpu().numpy()

print("Single validation sample")
print("J_true:", j_true_np)
print("J_pred:", j_pred)
