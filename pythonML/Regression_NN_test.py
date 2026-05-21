import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from utils import pick_best_device

from NeuralNetworkClass import RegressionMLP, train, test
from NeuralNetworkClass import ConvType1, ConvType2, ConvType3
from NeuralNetworkClass import TransformerType1, ConvTransformer

CSV_PATH = './data/triangular_anisotropy_uniform_Js4_B50_Nk20.csv'
BATCH_SIZE = 100
LR = 1e-3
DW = 1e-5
EPOCHS = 100
DEVICE = pick_best_device()
MODEL_OUT = './data'
INPUT_NOISE_STD = 0.0 #0.02

df = pd.read_csv(CSV_PATH)

for col in ['data', 'disp', 'Js']:
    df[col] = df[col].apply(json.loads)

disp = np.array(df['disp'].tolist(), dtype=np.float32)   # (N, L)
inten = np.array(df['data'].tolist(), dtype=np.float32)  # (N, L)
X = np.stack([disp, inten], axis=1)  # (N, 2, L)
#X = X.reshape(X.shape[0], -1)  # flatten to (N, 2*L)

y = np.array(df['Js'].tolist(), dtype=np.float32)
y[:, 1] *= y[:, 0]

X_mean = X.mean(axis=0, keepdims=True)
X_std = X.std(axis=0, keepdims=True) + 1e-8
X = (X - X_mean) / X_std

print("X shape:", X.shape)
print("y shape:", y.shape)
print("X range:", X.min(), X.max())
print("y range:", y.min(), y.max())

noise = np.random.normal(loc=0.0, scale=INPUT_NOISE_STD, size=X.shape).astype(np.float32)
X = X + noise

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

in_channel = X.shape[1]
input_length = X.shape[2]
output_dim = y.shape[1]
hidden_dim = 64 #1024 #256

model = RegressionMLP(in_channels=in_channel, input_length=input_length, hidden_dim=hidden_dim, output_dim=output_dim).to(DEVICE)
#model = ConvType3(in_channels=in_channel, input_length=input_length, hidden_dim=hidden_dim, output_dim=output_dim).to(DEVICE)
#model = TransformerType1(in_channels=in_channel, input_length=input_length, hidden_dim=hidden_dim, output_dim=output_dim).to(DEVICE)
#model = ConvTransformer(in_channels=in_channel, input_length=input_length, hidden_dim=hidden_dim, output_dim=output_dim).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=DW)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model has {n_params} trainable parameters.")

prev_updates = 0
for epoch in range(1, EPOCHS + 1):
    print(f"Epoch {epoch}/{EPOCHS} — device={DEVICE}")
    prev_updates = train(model, train_loader, optimizer, prev_updates, writer=None, device=DEVICE)
    test(model, val_loader, cur_step=epoch, writer=None, device=DEVICE)


#save checkpoint at only the final epoch
torch.save({
    "epoch": EPOCHS,
    "model_state": model.state_dict(),
    "optimizer_state": optimizer.state_dict(),
    "X_mean": X_mean,
    "X_std": X_std,
}, os.path.join(MODEL_OUT, f"ConvTrans_anisotropy_LSWT_Js{output_dim}_epoch{EPOCHS}_N250000.pth"))

print("Training finished. Models saved to:", MODEL_OUT)


# check 3 random samples from validation set
model.eval()
x_batch, y_true = val_ds[:3]
x_batch = x_batch.to(DEVICE)

with torch.no_grad():
    y_pred = model(x_batch).cpu().numpy()

print("Three validation sample")
print("y_true:", y_true.cpu().numpy())
print("y_pred:", y_pred)