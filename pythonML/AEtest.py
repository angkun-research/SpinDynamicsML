import os
import json
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

# import model + train/test helpers from your module
#from pythonML.AutoEncoderClass import FCautoencoder, train, test
from AutoEncoderClass import FCBsatDecoder, FCautoencoder, train, test
from AutoEncoderClass import ConvEncoder, TransformerEncoder,Conv1DEncoder
from AutoEncoderClass import FCBsatDecoder

# --- Config ---
#CSV_PATH = './data/triangular_heisenberg_Js3_h20.csv' #_qsPath400.csv' #qsBZ400.csv' 
#CSV_PATH = './data/triangular_anisotropy_Js8_h20_qsBZ100.csv'
#CSV_PATH = './data/triangular_anisotropy_temp.csv'
#'./data/honeycomb_heisenberg_sweep.csv' #'./data/triangular_heisenberg_sweep.csv'
#CSV_PATH = './data/HaoResults/Hao_anisotropy_Js5_Bdep.csv'
#CSV_PATH = './data/HaoResults/Hao_anisotropy_LSWT_Tensor_Js5_Bdep.csv'
#CSV_PATH = './data/HaoResults/Hao_anisotropy_Js5_Filter20_Bsat_N1009.csv' ##Hao_anisotropy_Js5_Filter20_N1009.csv' #'./data/HaoResults/Hao_anisotropy_Js5_B50.csv'
#CSV_PATH = './data/HaoResults/Hao_anisotropy_Js5_S6tensor_Filter20_Bsats.csv'
#CSV_PATH = './data/HaoResults/Hao_anisotropy_Js5_S6tensor_Filter3_Bsats.csv'
CSV_PATH = './data/triangular_anisotropy_uniform_Js4_B50_Nk20.csv'
BATCH_SIZE = 100
LR = 1e-3
DW = 1e-5
EPOCHS = 100 #100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_OUT = './data'
NumOfJ = 4 #8 #10
mag_recon_loss = 0.0
#os.makedirs(MODEL_OUT, exist_ok=True)

# --- Load data (same logic as RegressionTest.py) ---
import pandas as pd
df = pd.read_csv(CSV_PATH)

for col in ['data', 'disp', 'Js']:
    df[col] = df[col].apply(json.loads)

# for col in ['disp', 'Js', 'IntSxx', 'IntSyy', 'IntSzz', 'IntSxy', 'IntSxz', 'IntSyz']:
#      df[col] = df[col].apply(json.loads)

# Build feature matrix X and target J = [J1,J2,J3]
#X = np.array([row['disp'] for _, row in df.iterrows()], dtype=np.float32)
X = np.array([row['data'] + row['disp'] for _, row in df.iterrows()], dtype=np.float32)
#X = np.array([row['data'] + row['disp'] + [float(row["B"])] for _, row in df.iterrows()], dtype=np.float32)
#X = np.array([row['data'] + row['disp'] + [float(row["Bsat"])] for _, row in df.iterrows()], dtype=np.float32)
#X = np.array([row['disp'] + row['IntSxx'] + row['IntSyy'] + row['IntSzz'] + row['IntSxy'] + row['IntSxz'] + row['IntSyz'] for _, row in df.iterrows()], dtype=np.float32)

#y = df[['J1', 'J2', 'J3']].values.astype(np.float32)
y = np.array(df['Js'].tolist(), dtype=np.float32)
#y = np.delete(y, 3, axis=1)  # now shape: (N, 4)
y[:, 1] *= y[:, 0]   # make J[1] dimensionful by multiplying by J[0] elementwise

# Optional: simple normalization (per-feature)
X_mean = X.mean(axis=0, keepdims=True)
print("shape of X_mean:", X_mean.shape)
X_std = X.std(axis=0, keepdims=True) + 1e-8
X = (X - X_mean) / X_std
print(X.max(), X.min())
print(y.max(), y.min())

# add random noise to X for data augmentation 
NOISE_STD = 0.02 #0.01
print(f"Adding Gaussian noise with std={NOISE_STD} to input features for data augmentation.")
noise = np.random.normal(loc=0.0, scale=NOISE_STD, size=X.shape).astype(np.float32)
X = X + noise

# Convert to tensors and dataloaders
X_t = torch.from_numpy(X)
y_t = torch.from_numpy(y)
dataset = TensorDataset(X_t, y_t)

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
hidden_dim = 256 #1024
print(f"Hidden dimension: {hidden_dim}")
model = FCautoencoder(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=NumOfJ, relative_loss=relative_loss).to(DEVICE)
#model = ConvEncoder(input_dim=input_dim, hidden_dim=1024, latent_dim=3,kernel_size=3).to(DEVICE)
#model = TransformerEncoder(input_dim=input_dim, hidden_dim=1024, latent_dim=3).to(DEVICE)
#model = Conv1DEncoder(input_dim=input_dim,hidden_dim=64, latent_dim=NumOfJ, kernel_size=5,mag_recon_loss=mag_recon_loss).to(DEVICE)
#model = FCBsatDecoder(input_dim=input_dim, hidden_dim=1024, latent_dim=NumOfJ, relative_loss=relative_loss, mag_recon_loss=mag_recon_loss).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=DW)
# print number of parameters
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Model has {n_params} trainable parameters.')

prev_updates = 0
for epoch in range(1, EPOCHS + 1):
    print(f"Epoch {epoch}/{EPOCHS} — device={DEVICE}")
    prev_updates = train(model, train_loader, optimizer, prev_updates, writer=None, device=DEVICE)
    test(model, val_loader, cur_step=epoch, writer=None, device=DEVICE)

#save checkpoint at only the final epoch
# torch.save({
#     "epoch": EPOCHS,
#     "model_state": model.state_dict(),
#     "optimizer_state": optimizer.state_dict(),
#     "X_mean": X_mean,
#     "X_std": X_std,
# }, os.path.join(MODEL_OUT, f"FCAE_anisotropy_LSWT_Js{NumOfJ}_epoch{EPOCHS}_N250000.pth"))

# print("Training finished. Models saved to:", MODEL_OUT)


# ---- Print a single test example: true Js vs predicted Js ----
model.eval()
idx = 0  # change this to inspect a different validation example

x_one, j_true = train_ds[:3] #val_ds[:3]  # val_ds is a Subset; returns (X, Js)
x_one = x_one.unsqueeze(0).to(DEVICE)  # add batch dim: (1, input_dim)
#x_one = x_one.to(DEVICE)

with torch.no_grad():
    out = model.encode(x_one)

j_pred = out.squeeze(0).detach().cpu().numpy()
j_true_np = j_true.detach().cpu().numpy()

print("Single validation sample")
print("J_true:", j_true_np)
print("J_pred:", j_pred)
