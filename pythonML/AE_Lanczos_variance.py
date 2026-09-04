import os
import json
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd

from AutoEncoderClassNew import train, test
from AutoEncoderClassNew import FCautoencoder, Conv1Dminimal
from utils import set_seed
from utils import fit_standardizer, apply_standardizer,add_bandwidth_noise, make_dataset

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
seed = 1234
set_seed(seed)

# --- Load data (same logic as RegressionTest.py) ---
def load_raw_data(csv_path):
    df = pd.read_csv(csv_path)
    # Hao's data format
    for col in ['data', 'disp', 'Js']:
        df[col] = df[col].apply(json.loads)
    # wrong naming in previous data
    disp = np.array(df['data'].tolist(), dtype=np.float32)   # (N, L)
    inten = np.array(df['disp'].tolist(), dtype=np.float32)  # (N, L)
    # disp = np.array(df['disp'].tolist(), dtype=np.float32)
    # inten = np.array(df['data'].tolist(), dtype=np.float32)
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

def load_raw_data_gmm(csv_path, n_comp=1, n_e=20, num_J = 4):
    df = pd.read_csv(csv_path)
    for col in ['means', 'variances', 'weights', 'Js']:
        df[col] = df[col].apply(json.loads)
    means = np.array(df['means'].tolist(), dtype=np.float32)       # (N, Nk*n_pts)
    variances = np.array(df['variances'].tolist(), dtype=np.float32)
    weights = np.array(df['weights'].tolist(), dtype=np.float32)
    y = np.array(df['Js'].tolist(), dtype=np.float32)
    print(f"Total samples: {len(df)}")

    if num_J == 4:
        y = np.delete(y, 3, axis=1)   # remove 4th column (0-based index), shape becomes (N, 4)

    N = len(df)
    n_k = means.shape[1] // n_comp // n_e
    # flat layout is k-major, n_pts-minor -> (N, n_k, n_pts)
    means_r = means.reshape(N, n_k, n_comp, n_e)
    variances_r = variances.reshape(N, n_k, n_comp, n_e)
    weights_r = weights.reshape(N, n_k, n_comp, n_e)

    # move n_pts before n_k so sequence dim (q-points) is last, matching old disp_ch/data_ch layout
    means_ch = np.transpose(means_r, (0, 2, 3, 1)).reshape(N, n_comp * n_e, n_k)       # (N, n_comp * n_e, n_k)
    variances_ch = np.transpose(variances_r, (0, 2, 3, 1)).reshape(N, n_comp * n_e, n_k)
    weights_ch = np.transpose(weights_r, (0, 2, 3, 1)).reshape(N, n_comp * n_e, n_k)

    X = np.concatenate([means_ch, variances_ch, weights_ch], axis=1)  # (N, 3*n_pts, n_k)
    mask = (y != 0).all(axis=1)
    X = X[mask]
    y = y[mask]
    print(f"Filtered samples: {len(y)}")
    print(f"X shape: {X.shape}, y shape: {y.shape}")

    y[:, 1] *= y[:, 0]
    return X, y

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

train_mares = []
train_mvars = []
test_mares = []
test_mvars = []
for epoch in range(1, EPOCHS + 1):
    print(f"Epoch {epoch}/{EPOCHS} — device={DEVICE}")
    all_out_train = train(model, train_loader, optimizer,has_variance=has_variance,
        latent_dim=NumOfJ, batch_size=BATCH_SIZE, device=DEVICE)
    all_out_test = test(model, val_loader, has_variance=has_variance, latent_dim=NumOfJ, batch_size=BATCH_SIZE, device=DEVICE)
    train_mares.append(all_out_train["train_mare"])
    train_mvars.append(all_out_train["train_var_mean"])
    test_mares.append(all_out_test["test_mare"])
    test_mvars.append(all_out_test["test_var_mean"])

print("Minimum train MARE:", min(train_mares), "at epoch", np.argmin(train_mares) + 1)
print("Minimum train MVAR:", min(train_mvars), "at epoch", np.argmin(train_mvars) + 1)
print("Minimum test MARE:", min(test_mares), "at epoch", np.argmin(test_mares) + 1)
print("Minimum test MVAR:", min(test_mvars), "at epoch", np.argmin(test_mvars) + 1)

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
