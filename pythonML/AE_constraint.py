import os
import json
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

# import model + train/test helpers from your module
#from pythonML.AutoEncoderClass import FCautoencoder, train, test
from AutoEncoderClass import train, test
from AutoEncoderClass import FCBsats3Decoder

# --- Config ---
CSV_PATH = './data/HaoResults/Hao_anisotropy_Js5_Filter20_Bsats.csv' ##Hao_anisotropy_Js5_Filter20_N1009.csv' #'./data/HaoResults/Hao_anisotropy_Js5_B50.csv'
#CSV_PATH = './data/HaoResults/Hao_anisotropy_Js5_S6tensor_Filter20_Bsats.csv'
BATCH_SIZE = 100
LR = 1e-3
DW = 1e-5
EPOCHS = 100 #100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_OUT = './data'
NumOfJ = 5 
mag_recon_loss = 0.2 #1.0
#os.makedirs(MODEL_OUT, exist_ok=True)

# --- Load data (same logic as RegressionTest.py) ---
import pandas as pd
df = pd.read_csv(CSV_PATH)

for col in ['data', 'disp', 'Js','Bsats']:
    df[col] = df[col].apply(json.loads)

# Build feature matrix X and target J = [J1,J2,J3]
#X = np.array([row['data'] + row['disp'] for _, row in df.iterrows()], dtype=np.float32)
X = np.array([row['data'] + row['disp'] + [float(row["B"])] + row['Bsats'] for _, row in df.iterrows()], dtype=np.float32)
y = np.array(df['Js'].tolist(), dtype=np.float32)
#y = np.delete(y, 3, axis=1)  # now shape: (N, 4)

# Optional: simple normalization (per-feature)
X_mean = X.mean(axis=0, keepdims=True)
X_std = X.std(axis=0, keepdims=True) + 1e-8
X = (X - X_mean) / X_std
print(X.max(), X.min())
print(y.max(), y.min())

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
#print(X.shape)
model = FCBsats3Decoder(input_dim=input_dim, hidden_dim=1024, latent_dim=NumOfJ, relative_loss=relative_loss, mag_recon_loss=mag_recon_loss).to(DEVICE)


optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=DW)
# print number of parameters
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Model has {n_params} trainable parameters.')

prev_updates = 0
for epoch in range(1, EPOCHS + 1):
    print(f"Epoch {epoch}/{EPOCHS} — device={DEVICE}")
    prev_updates = train(model, train_loader, optimizer, prev_updates, writer=None, device=DEVICE)
    test(model, val_loader, cur_step=epoch, writer=None, device=DEVICE)

# #save checkpoint at only the final epoch
torch.save({
    "epoch": EPOCHS,
    "model_state": model.state_dict(),
    "optimizer_state": optimizer.state_dict(),
    "X_mean": X_mean,
    "X_std": X_std,
}, os.path.join(MODEL_OUT, f"FCAE_anisotropy_Js{NumOfJ}_epoch{EPOCHS}_scalar.pth"))

print("Training finished. Models saved to:", MODEL_OUT)



# ---- Print a single test example: true Js vs predicted Js ----
# model.eval()
# idx = 0  # change this to inspect a different validation example

# x_one, j_true = train_ds[:3] #val_ds[:3]  # val_ds is a Subset; returns (X, Js)
# x_one = x_one.unsqueeze(0).to(DEVICE)  # add batch dim: (1, input_dim)

# with torch.no_grad():
#     out = model.encode(x_one)

# j_pred = out.squeeze(0).detach().cpu().numpy()
# j_true_np = j_true.detach().cpu().numpy()

# print("Single validation sample")
# print("J_true:", j_true_np)
# print("J_pred:", j_pred)
