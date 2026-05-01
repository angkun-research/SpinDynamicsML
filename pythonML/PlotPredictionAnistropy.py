import os
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from AutoEncoderClass import FCBsats3Decoder

# --- Config (match training settings) ---
CSV_PATH = './data/HaoResults/Hao_anisotropy_Js5_Filter20_Bsats.csv'
#CSV_PATH = './data/HaoResults/Hao_anisotropy_Js5_S6tensor_Filter20_Bsats.csv'
MODEL_OUT = './data/'
CHECKPOINT = os.path.join(MODEL_OUT, 'FCAE_anisotropy_Js5_epoch100_scalar.pth')  # adjust if different
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 200

# --- Load dataset ---
df = pd.read_csv(CSV_PATH)
for col in ['data', 'disp', 'Js','Bsats']:
    df[col] = df[col].apply(json.loads)

X = np.array([row['data'] + row['disp'] + [float(row["B"])] + row['Bsats'] for _, row in df.iterrows()], dtype=np.float32)
y = np.array(df['Js'].tolist(), dtype=np.float32)

# load normalization if present in checkpoint
ckpt = torch.load(CHECKPOINT, map_location='cpu',weights_only=False)
X_mean = ckpt.get('X_mean', X.mean(axis=0, keepdims=True))
X_std = ckpt.get('X_std', X.std(axis=0, keepdims=True) + 1e-8)

X = (X - X_mean) / X_std

# --- DataLoader ---
X_t = torch.from_numpy(X)
y_t = torch.from_numpy(y)
dataset = torch.utils.data.TensorDataset(X_t, y_t)
loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# --- Rebuild model and load weights ---
NumOfJ = y.shape[1]
input_dim = X.shape[1]
mag_recon_loss = 0.2 # 1.0
model = FCBsats3Decoder(input_dim=input_dim, hidden_dim=1024, latent_dim=NumOfJ, mag_recon_loss=mag_recon_loss).to(DEVICE)
model.load_state_dict(ckpt['model_state'])
model.eval()

# helper to extract latent predictions robustly
def extract_latent(model, xb, yb_dim):
    out = None
    with torch.no_grad():
        try:
            if hasattr(model, "encode"):
                out = model.encode(xb)
            else:
                o = model(xb)
                if isinstance(o, torch.Tensor):
                    out = o
                elif isinstance(o, (list, tuple)):
                    # choose first tensor with matching last dim, else first tensor
                    for item in o:
                        if isinstance(item, torch.Tensor) and item.dim() > 1 and item.shape[1] == yb_dim:
                            out = item
                            break
                    if out is None:
                        for item in o:
                            if isinstance(item, torch.Tensor):
                                out = item
                                break
        except Exception:
            o = model(xb)
            out = o if isinstance(o, torch.Tensor) else (o[0] if isinstance(o, (list,tuple)) else None)
    if out is None:
        return None
    if not isinstance(out, torch.Tensor):
        out = torch.tensor(out, device=xb.device, dtype=xb.dtype)
    if out.dim() == 1:
        out = out.unsqueeze(0)
    return out

# --- Collect predictions and compute squared errors ---
preds_list = []
trues_list = []
with torch.no_grad():
    for xb, yb in loader:
        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)
        z = extract_latent(model, xb, yb.shape[1])
        if z is None:
            raise RuntimeError("Failed to extract latent predictions from model.")
        # ensure correct dims
        if z.shape[1] != yb.shape[1]:
            minc = min(z.shape[1], yb.shape[1])
            z = z[:, :minc]
            yb = yb[:, :minc]
        preds_list.append(z.cpu().numpy())
        trues_list.append(yb.cpu().numpy())

preds = np.vstack(preds_list)
print("Predictions shape:", preds.shape)
trues = np.vstack(trues_list)
squared_errors = (preds - trues) ** 2
NumOfJ_detected = squared_errors.shape[1]
print("Squared errors shape:", squared_errors.shape)

Label_list = [r'$J_1^0$', r'$\Delta$', r'$J_{\pm\pm}$', r'$J_{z\pm}$', r'$J_2^0$']

# --- Load dataset ---
# df_single = pd.read_csv('./data/triangular_heisenberg_Js10_h20_qsBZ_single.csv')
# for col in ['data', 'disp', 'Js']:
#     df_single[col] = df_single[col].apply(json.loads)

# X_single = np.array([row['data'] + row['disp'] for _, row in df_single.iterrows()], dtype=np.float32)
# y_single = np.array(df_single['Js'].tolist(), dtype=np.float32)

# X_single = (X_single - X_mean) / X_std
# X_single_t = torch.from_numpy(X_single)
# with torch.no_grad():
#     X_single_t = X_single_t.to(DEVICE)
#     z_single = extract_latent(model, X_single_t, y_single.shape[1])
#     z_single = z_single.cpu().numpy()

# print("Single sample prediction:", z_single)

# 1b) Error distribution per coupling component J_k
# errors = preds - trues   # signed error: (pred - true)

# ncols = 5
# nrows = int(np.ceil(NumOfJ_detected / ncols))
# fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3*nrows), squeeze=False)

# for k in range(NumOfJ_detected):
#     r = k // ncols
#     c = k % ncols
#     ax = axes[r][c]

#     ek = errors[:, k]
#     ax.hist(ek, bins=40, density=True, alpha=0.75, edgecolor='black')
#     ax.axvline(0.0, color='red', linestyle='--', linewidth=1)  # perfect prediction
#     ax.set_title(Label_list[k])
#     ax.set_xlabel('Prediction error (pred - true)')
#     ax.set_ylabel('Density')
#     ax.grid(True, alpha=0.3)

# # hide unused axes
# for k in range(NumOfJ_detected, nrows * ncols):
#     r = k // ncols
#     c = k % ncols
#     axes[r][c].axis('off')

# plt.tight_layout()
# plt.savefig('/Users/angkunwu/Desktop/error_distribution_per_J.png', dpi=300)
# plt.show()
# plt.close()


# 3) J_true vs J_pred for each coupling component J_k
ncols = 5
nrows = int(np.ceil(NumOfJ_detected / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows), squeeze=False)

for k in range(NumOfJ_detected):
    r = k // ncols
    c = k % ncols
    ax = axes[r][c]

    jt = trues[:, k]
    jp = preds[:, k]

    # scatter points
    ax.scatter(jt, jp, s=8, alpha=0.5)

    # ideal line y = x
    vmin = min(jt.min(), jp.min())
    vmax = max(jt.max(), jp.max())
    ax.plot([vmin, vmax], [vmin, vmax], 'r--', linewidth=1)

    ax.set_title(Label_list[k])
    ax.set_xlabel('J_true')
    ax.set_ylabel('J_pred')
    ax.grid(True, alpha=0.3)

# hide unused axes
for k in range(NumOfJ_detected, nrows * ncols):
    r = k // ncols
    c = k % ncols
    axes[r][c].axis('off')

plt.tight_layout()
plt.savefig('/Users/angkunwu/Desktop/Jtrue_vs_Jpred_per_component.png', dpi=300)
plt.show()
plt.close()