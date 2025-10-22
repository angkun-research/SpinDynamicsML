import os
import argparse
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import matplotlib.pyplot as plt

# import your model
from AutoEncoderClass import FCautoencoder

def load_csv_features(csv_path):
    df = pd.read_csv(csv_path)
    for col in ['data', 'disp']:
        df[col] = df[col].apply(json.loads)
    X = np.array([row['data'] + row['disp'] for _, row in df.iterrows()], dtype=np.float32)
    y = df[['J1', 'J2', 'J3']].values.astype(np.float32)
    return X, y

def main():
    # Configuration (edit these instead of using command-line args)
    CSV_PATH = './data/triangular_heisenberg_sweep.csv'
    CKPT_PATH = './data/autoencoder_epoch100.pth'   # change to your checkpoint
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    X_raw, y = load_csv_features(CSV_PATH)
    X = X_raw.copy()
    # load checkpoint (map to CPU/GPU as requested)
    device = torch.device(DEVICE)
    print(f'Loading model checkpoint from {CKPT_PATH} to device={device}')
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)

    # simple normalization (per-feature)
    X_mean = X.mean(axis=0, keepdims=True)
    X_std = X.std(axis=0, keepdims=True) + 1e-8
    X = (X - X_mean) / X_std

    # build model and load weights
    input_dim = X.shape[1]
    model = FCautoencoder(input_dim=input_dim, hidden_dim=1024, latent_dim=3).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    # encode in batches to avoid OOM
    X_t = torch.from_numpy(X).to(device).float()
    batch = 256
    zs = []
    recons = []
    with torch.no_grad():
        for i in range(0, X_t.size(0), batch):
            xb = X_t[i:i+batch]
            # use encode() to get latent (encode already applies tanh in current AutoEncoderClass)
            z = model.encode(xb)
            xrec = model.decode(z)
            zs.append(z.detach().cpu().numpy())
            recons.append(xrec.detach().cpu().numpy())
    Z = np.vstack(zs)
    X_rec_norm = np.vstack(recons)
    X_rec = X_rec_norm * X_std + X_mean

    # # plot 20 random paired points in the same 3D axes
    # n_show = min(20, y.shape[0])
    # rng = np.random.default_rng(42)
    # idx = rng.choice(y.shape[0], size=n_show, replace=False)
    # y_s = y[idx]
    # z_s = Z[idx]

    # fig = plt.figure(figsize=(7, 7))
    # ax = fig.add_subplot(111, projection='3d')

    # # true J: circles, latent z: triangles
    # ax.scatter(y_s[:, 0], y_s[:, 1], y_s[:, 2],
    #            c='C1', s=80, label='true J (o)', marker='o', depthshade=True)
    # ax.scatter(z_s[:, 0], z_s[:, 1], z_s[:, 2],
    #            c='C0', s=80, label='latent z (^)', marker='^', depthshade=True)

    # # connect each true J to its corresponding latent point
    # for i in range(n_show):
    #     ax.plot([y_s[i, 0], z_s[i, 0]],
    #             [y_s[i, 1], z_s[i, 1]],
    #             [y_s[i, 2], z_s[i, 2]],
    #             c='gray', alpha=0.6, linewidth=0.8)

    # ax.set_title(f'{n_show} random paired points: true J (o) vs latent z (^)')
    # ax.set_xlabel('J1 / z1'); ax.set_ylabel('J2 / z2'); ax.set_zlabel('J3 / z3')
    # try:
    #     ax.set_box_aspect([1, 1, 1])
    # except Exception:
    #     pass
    # ax.legend()
    # plt.tight_layout()

    # # out_path = './data/latent_vs_true_sample20.png'
    # # plt.savefig(out_path, dpi=200)
    # # print(f'Figure saved to {out_path}')
    # plt.show()

    # choose 3 random samples to visualize (deterministic)
    rng = np.random.default_rng(42)
    n_recon = 3
    idx_recon = rng.choice(X_raw.shape[0], size=n_recon, replace=False)
    colors = ['C0', 'C1', 'C2']

    # plot all 6 curves on the same axes (original = solid, recon = dashed)
    plt.figure(figsize=(10, 5))
    x_axis = np.arange(X_raw.shape[1])
    for i, idx in enumerate(idx_recon):
        orig = X_raw[idx]
        recon = X_rec[idx]
        c = colors[i % len(colors)]
        plt.plot(x_axis, orig, label=f'orig idx {idx}', color=c, linestyle='-')
        plt.plot(x_axis, recon, label=f'recon idx {idx}', color=c, linestyle='--')
    plt.xlabel('feature index')
    plt.ylabel('value')
    plt.title('Original (solid) and reconstructed (dashed) curves — 3 samples')
    plt.legend(loc='upper right', fontsize='small', ncol=2)
    plt.tight_layout()
    #plt.savefig(OUT, dpi=200)
    #print(f'Figure saved to {OUT}')
    plt.show()

if __name__ == '__main__':
    main()