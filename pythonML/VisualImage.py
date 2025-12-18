import json
import numpy as np
import pandas as pd
import torch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import matplotlib.pyplot as plt

# import your model
from AutoEncoderClass import FCautoencoder
from utils import create_heatmap_from_flat

def load_csv_features(csv_path):
    df = pd.read_csv(csv_path)
    for col in ['data', 'disp']:
        df[col] = df[col].apply(json.loads)
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
        _, img = create_heatmap_from_flat(flat_disp, flat_data, Nbands=Nbands, Nq=Nq, Egrid=Egrid, sigma=0.2, normalize=False)
        # flatten the 2D heatmap into a 1D feature vector
        images.append(img.ravel())

    X = np.vstack(images).astype(np.float32)
    y = df[['J1', 'J2', 'J3']].values.astype(np.float32)
    return X, y

def main():
    # Configuration (edit these instead of using command-line args)
    CSV_PATH = './data/triangular_heisenberg_sweep.csv'
    CKPT_PATH = './data/FCAE_Triangular_Images_epoch100_magLoss100.0.pth' #'./data/FCAE_Triangular_Images_epoch100.pth'
    OUT_PATH = '../Desktop/fig1.png'
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    X_raw, y = load_csv_features(CSV_PATH)
    X = X_raw.copy()
    # load checkpoint (map to CPU/GPU as requested)
    device = torch.device(DEVICE)
    print(f'Loading model checkpoint from {CKPT_PATH} to device={device}')
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)

    # # simple normalization (per-feature)
    # X_mean = X.mean(axis=0, keepdims=True)
    # X_std = X.std(axis=0, keepdims=True) + 1e-8
    # X = (X - X_mean) / X_std

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
    X_rec = X_rec_norm # * X_std + X_mean

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

    # plt.savefig(OUT_PATH, dpi=1000)
    # print(f'Figure saved to {OUT_PATH}')
    # plt.show()

    #choose 3 random samples to visualize (deterministic)
    rng = np.random.default_rng(42)
    n_recon = 3
    idx_recon = rng.choice(X_raw.shape[0], size=n_recon, replace=False)
    # colors = ['C0', 'C1', 'C2']

    # infer Nq from feature dimension (features = Nq * len(Egrid)=300)
    Egrid_len = 300
    Nq = X_raw.shape[1] // Egrid_len
    X_rec_mat = X_rec.reshape((-1, Nq, Egrid_len))
    X_raw_mat = X_raw.reshape((-1, Nq, Egrid_len))

    # choose common color scale for fair comparison
    vmin = min(X_raw_mat.min(), X_rec_mat.min())
    vmax = max(X_raw_mat.max(), X_rec_mat.max())

    # create grid of subplots: n_recon rows, 2 columns (original | recon)
    fig, axes = plt.subplots(n_recon, 2, figsize=(10, 3 * n_recon))
    axes = np.atleast_2d(axes)
    for row_i, sample_idx in enumerate(idx_recon):
        ax0 = axes[row_i, 0]
        ax1 = axes[row_i, 1]

        im0 = ax0.imshow(X_raw_mat[sample_idx].T, origin='lower', aspect='auto',
                         cmap='viridis', vmin=vmin, vmax=vmax)
        ax0.set_title(f'Sample {sample_idx} — original')
        ax0.set_xlabel('Energy bin')
        ax0.set_ylabel('q index')

        im1 = ax1.imshow(X_rec_mat[sample_idx].T, origin='lower', aspect='auto',
                         cmap='viridis', vmin=vmin, vmax=vmax)
        ax1.set_title(f'Sample {sample_idx} — reconstructed')
        ax1.set_ylabel('Energy bin')
        ax1.set_xlabel('q index')

    # single colorbar for all subplots
    #fig.colorbar(im1, ax=axes.ravel().tolist(), shrink=0.8)
    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=300)
    print(f'Figure saved to {OUT_PATH}')
    plt.show()

if __name__ == '__main__':
    main()