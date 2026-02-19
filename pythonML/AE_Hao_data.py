import os
import json
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd

from AutoEncoderClass import FCautoencoder, train, test

# --- Config ---
CSV_PATH = './data/HaoResults/CombinedData_J5_d1000.csv'
BATCH_SIZE = 64
LR = 1e-3
DW = 1e-5
EPOCHS = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#MODEL_OUT = './data/HaoResults/models'
NumOfJ = 5

#os.makedirs(MODEL_OUT, exist_ok=True)
print(f"Using device: {DEVICE}")

# --- Load data ---
df = pd.read_csv(CSV_PATH)
# columns were serialized as JSON strings — parse back to lists
df['Xs'] = df['Xs'].apply(json.loads)
df['Js'] = df['Js'].apply(json.loads)

X = np.array(df['Xs'].tolist(), dtype=np.float32)
y = np.array(df['Js'].tolist(), dtype=np.float32)

# normalize features
# X_mean = X.mean(axis=0, keepdims=True)
# X_std = X.std(axis=0, keepdims=True) + 1e-8
# X = (X - X_mean) / X_std
print(f"Feature matrix X shape: {X.shape}, target J shape: {y.shape}")

# tensors and loaders
X_t = torch.from_numpy(X)
y_t = torch.from_numpy(y)
dataset = TensorDataset(X_t, y_t)

n = len(dataset)
n_train = int(0.9 * n)
n_val = n - n_train
train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

# --- Model ---
input_dim = X.shape[1]
model = FCautoencoder(input_dim=input_dim, hidden_dim=1024, latent_dim=NumOfJ, relative_loss=False).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=DW)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Model has {n_params} trainable parameters. Device={DEVICE}')

prev_updates = 0
for epoch in range(1, EPOCHS + 1):
    print(f"Epoch {epoch}/{EPOCHS}")
    prev_updates = train(model, train_loader, optimizer, prev_updates, writer=None, device=DEVICE)
    test(model, val_loader, cur_step=epoch, writer=None, device=DEVICE)

# # save final checkpoint
# torch.save({
#     "epoch": EPOCHS,
#     "model_state": model.state_dict(),
#     "optimizer_state": optimizer.state_dict(),
#     "X_mean": X_mean,
#     "X_std": X_std,
# }, os.path.join(MODEL_OUT, f"FCAE_Hao_Js{NumOfJ}_epoch{EPOCHS}.pth"))

# print("Training finished. Model saved:", MODEL_OUT)