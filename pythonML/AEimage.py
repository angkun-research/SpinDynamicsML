import os
import json
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd

# import model + train/test helpers from your module
#from pythonML.AutoEncoderClass import FCautoencoder, train, test
from AutoEncoderClass import FCautoencoder, train, test
from AutoEncoderClass import ConvEncoder, TransformerEncoder
from AutoEncoderClass import ConvDownSampling, SimpleConvAutoencoder, ViTAutoencoder
from AutoEncoderClass import Conv1DAutoencoder,ResNetEncoder
from utils import create_heatmap_from_flat

# --- Config ---
CSV_PATH = './data/triangular_heisenberg_Js10_h20_qsBZ.csv'
BATCH_SIZE = 100
LR = 1e-3
DW = 1e-5
EPOCHS = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_OUT = './data'
#os.makedirs(MODEL_OUT, exist_ok=True)

# --- Load data (same logic as RegressionTest.py) ---
df = pd.read_csv(CSV_PATH)

for col in ['data', 'disp', 'Js']:
    df[col] = df[col].apply(json.loads)

# find max E and min E for setting Egrid later (optional)
all_disp = np.concatenate(df['disp'].values)
E_max = all_disp.max()
E_min = all_disp.min()
print(f"Dispersion energy range: {E_min} to {E_max} meV")

# Build feature matrix X and target J = [J1,J2,J3]
Nq = len(df.iloc[0]['disp']) # assuming 1 band per q-point
Nbands = 1
Egrid = np.linspace(E_min, E_max, 300) #None
images = []
for idx, row in df.iterrows():
    flat_disp = row['disp']
    flat_data = row['data']
    # ensure correct lengths
    if len(flat_disp) != len(flat_data):
        raise ValueError(f"disp/data length mismatch at row {idx}")
    # create_heatmap_from_flat returns (Egrid, img) where img shape = (Nq, len(Egrid))
    _, img = create_heatmap_from_flat(flat_disp, flat_data, Nbands=Nbands, Nq=Nq, Egrid=Egrid, sigma=0.2, normalize=True)
    # flatten the 2D heatmap into a 1D feature vector
    images.append(img.ravel())

X = np.vstack(images).astype(np.float32)
#y = df[['J1', 'J2', 'J3']].values.astype(np.float32)
y = np.array(df['Js'].tolist(), dtype=np.float32)

# Optional: simple normalization (per-feature)
# X_mean = X.mean(axis=0, keepdims=True)
# X_std = X.std(axis=0, keepdims=True) + 1e-8
# X = (X - X_mean) / X_std
print(X.max(), X.min())
print(y.max(), y.min())

# Convert to tensors and dataloaders
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
mag_recon_loss = 100.0 #100.0
ifdecode = False
#model = FCautoencoder(input_dim=input_dim, hidden_dim=1024, latent_dim=8, mag_recon_loss=mag_recon_loss,ifdecode=ifdecode).to(DEVICE)
#model = ConvEncoder(input_dim=input_dim, hidden_dim=1024, latent_dim=3,kernel_size=3, mag_recon_loss=mag_recon_loss).to(DEVICE)
#model = TransformerEncoder(input_dim=input_dim, hidden_dim=1024, latent_dim=3, mag_recon_loss=mag_recon_loss).to(DEVICE)
#model = ConvDownSampling(Nq,len(Egrid), base_channels=32,latent_dim=3,kernel_size=3, mag_recon_loss=mag_recon_loss).to(DEVICE)
#model = SimpleConvAutoencoder(Nq,len(Egrid), base_channels=16,latent_dim=3, mag_recon_loss=mag_recon_loss).to(DEVICE)
#model = ViTAutoencoder(Nq,len(Egrid),mag_recon_loss=mag_recon_loss).to(DEVICE)
#model = Conv1DAutoencoder(Nq,len(Egrid),mag_recon_loss=mag_recon_loss, ifdecode=ifdecode).to(DEVICE)
model = ResNetEncoder(Nq,len(Egrid),latent_dim=10,pre_fc_out_channels=1).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=DW)
# print number of parameters
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Model has {n_params} trainable parameters.')

prev_updates = 0
for epoch in range(1, EPOCHS + 1):
    print(f"Epoch {epoch}/{EPOCHS} — device={DEVICE}")
    prev_updates = train(model, train_loader, optimizer, prev_updates, writer=None, device=DEVICE)
    test(model, val_loader, cur_step=epoch, writer=None, device=DEVICE)

# # save checkpoint at only the final epoch
# torch.save({
#     "epoch": EPOCHS,
#     "model_state": model.state_dict(),
#     "optimizer_state": optimizer.state_dict(),
#     # "X_mean": X_mean,
#     # "X_std": X_std,
# }, 
# #os.path.join(MODEL_OUT, f"ResNet18_Triangular_Images_epoch{EPOCHS}_Hidden{32}_kernel{3}_magLoss{mag_recon_loss}.pth"))
# os.path.join(MODEL_OUT, f"ResNet18_Triangular_Images_epoch{EPOCHS}.pth"))

# print("Training finished. Models saved to:", MODEL_OUT)