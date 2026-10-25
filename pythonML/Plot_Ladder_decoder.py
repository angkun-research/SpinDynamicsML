import os
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

plt.rcParams['font.family'] = 'serif'       # or 'sans-serif', 'monospace'
plt.rcParams['font.serif'] = ['Times New Roman']
# plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pythonML.AutoEncoderClassNew import Conv1Dminimal
from pythonML.utils import create_heatmap

# =========================
# User config
# =========================
#CSV_PATH = './data/HaoResults/Hao_anisotropy_Js4_uniform_trace_Filter100_Bsats.csv'
CSV_PATH = './data/HaoResults/THL_dataset12_Js4_Scalar_Qz0_Nk60_Filter20.csv'

CKPT_CLEAN = "./data/ConvChannel_scalar_Ladder_Js4_Nk60_Ne20_sigma0.0_epoch198.pth"
CKPT_NOISY = "./data/ConvChannel_scalar_Ladder_Js4_Nk60_Ne20_sigma0.1_epoch196.pth"

LATENT_DIM = 4
HIDDEN_DIM = 64
HAS_VARIANCE = True
IF_DECODE = True

# Noise level used to build "true dispersion with noise"
SIGMA_NOISE_FOR_PLOT = 0.1
NOISE_SEED = 1234

# Rebuild the same 90/10 split style
SPLIT_SEED = 1234
TRAIN_RATIO = 0.9

# Choose indices from test set (indices relative to test subset)
TEST_INDICES = [8] #[123] #[0, 3, 10]
target = [1.0, 0.7, 0.2, 0.051]
E_MIN = 0.0
E_MAX = 20 #15.0
E_BINS = 100
SIGMA_DSSF = 0.5 #0.2 #0.05

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# index 8
# $y_{true}=[1. , 1. , 1. , 0.5]$
# $\mu_{\sigma=0.0}=[1.018004, 1.028298, 1.016899, 0.5007  ]$
# $\sqrt{\mathrm{var}}_{\sigma=0.0}=[0.010493, 0.014247, 0.009588, 0.004778]$
# $\mu_{\sigma=0.1}=[1.004141, 0.973033, 1.003829, 0.509457]$
# $\sqrt{\mathrm{var}}_{\sigma=0.1}=[0.009262, 0.014023, 0.012518, 0.008106]$

def add_bandwidth_noise(X, sigma, seed=1234):
    rng = np.random.default_rng(seed)
    eps = rng.normal(0.0, 1.0, size=X.shape).astype(np.float32)

    max_vals = X.max(axis=2, keepdims=True)
    min_vals = X.min(axis=2, keepdims=True)
    bandwidth = max_vals - min_vals

    # Treat second half (intensity channels) as max-based bandwidth
    c_half = X.shape[1] // 2
    bandwidth[:, c_half:, :] = max_vals[:, c_half:, :]

    return X + sigma * bandwidth * eps


def to_numpy(x):
    if isinstance(x, np.ndarray):
        return x.astype(np.float32)
    if torch.is_tensor(x):
        return x.detach().cpu().numpy().astype(np.float32)
    return np.asarray(x, dtype=np.float32)


def load_raw_data(csv_path, n_comp=1, n_e=20):
    df = pd.read_csv(csv_path)
    for col in ["data", "disp", "Js"]:
        df[col] = df[col].apply(json.loads)

    # Keep the same swapped mapping as AE_Lanczos_variance.py
    # disp = np.array(df["data"].tolist(), dtype=np.float32)   # (N, Lflat)
    # inten = np.array(df["disp"].tolist(), dtype=np.float32)  # (N, Lflat)
    disp = np.array(df["disp"].tolist(), dtype=np.float32)   # (N, Lflat)
    inten = np.array(df["data"].tolist(), dtype=np.float32)  # (N, Lflat)
    y = np.array(df["Js"].tolist(), dtype=np.float32)

    y[:, 1] *= y[:, 0]

    N = len(df)
    n_k = disp.shape[1] // n_comp // n_e

    # (N, n_comp, n_k, n_e)
    disp4 = disp.reshape(N, n_comp, n_k, n_e)
    inten4 = inten.reshape(N, n_comp, n_k, n_e)

    # Keep only component 0 -> (N, 1, n_k, n_e)
    disp4 = disp4[:, 0:1, :, :]
    inten4 = inten4[:, 0:1, :, :]

    # (N, channels=n_e, n_k)
    disp_ch = np.transpose(disp4, (0, 1, 3, 2)).reshape(N, n_e, n_k)
    inten_ch = np.transpose(inten4, (0, 1, 3, 2)).reshape(N, n_e, n_k)

    # Final AE input: first half dispersion channels, second half intensity channels
    X = np.concatenate([disp_ch, inten_ch], axis=1)  # (N, 2*n_e, n_k)

    # Filter zeros in targets
    mask = (y != 0).all(axis=1)
    X = X[mask]
    y = y[mask]

    return X, y

def load_lanczos_imag(csv_path, n_comp=1, n_e=100):
    df = pd.read_csv(csv_path)
    for col in ["data", "disp", "Js"]:
        df[col] = df[col].apply(json.loads)

    disp = np.array(df["disp"].tolist(), dtype=np.float32)   # (N, Lflat)
    inten = np.array(df["data"].tolist(), dtype=np.float32)  # (N, Lflat)

    N = len(df)
    n_k = disp.shape[1] // n_comp // n_e

    # (N, n_comp, n_k, n_e)
    disp4 = disp.reshape(N, n_comp, n_k, n_e)
    inten4 = inten.reshape(N, n_comp, n_k, n_e)

    # Keep only component 0 -> (N, 1, n_k, n_e)
    disp4 = disp4[:, 0:1, :, :]
    inten4 = inten4[:, 0:1, :, :]

    # (N, channels=n_e, n_k)
    disp_ch = np.transpose(disp4, (0, 1, 3, 2)).reshape(N, n_e, n_k)
    inten_ch = np.transpose(inten4, (0, 1, 3, 2)).reshape(N, n_e, n_k)

    x_ch = np.concatenate([disp_ch, inten_ch], axis=1)  # (N, 2*n_e, n_k)
    return x_ch

def get_test_global_indices(n_total, train_ratio=0.9, split_seed=1234):
    n_train = int(train_ratio * n_total)
    g = torch.Generator().manual_seed(split_seed)
    perm = torch.randperm(n_total, generator=g).tolist()
    test_global = perm[n_train:]
    return test_global

def get_train_global_indices(n_total, train_ratio=0.9, split_seed=1234):
    n_train = int(train_ratio * n_total)
    g = torch.Generator().manual_seed(split_seed)
    perm = torch.randperm(n_total, generator=g).tolist()
    return perm[:n_train]

def build_model(input_length, in_channels=2):
    model = Conv1Dminimal(
        in_channels=in_channels,
        input_length=input_length,
        hidden_dim=HIDDEN_DIM,
        latent_dim=LATENT_DIM,
        has_variance=HAS_VARIANCE,
        ifdecode=IF_DECODE,
    )
    return model


def load_model_and_stats(ckpt_path, input_length, in_channels=200):
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)

    model = build_model(input_length=input_length, in_channels=in_channels).to(DEVICE)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    X_mean = to_numpy(ckpt["X_mean"])
    X_std = to_numpy(ckpt["X_std"])
    return model, X_mean, X_std


@torch.no_grad()
def reconstruct_one(model, X_raw_one, X_mean, X_std):
    # X_raw_one: (2, L)
    Xn = (X_raw_one[None, ...] - X_mean) / X_std  # (1,2,L)
    xt = torch.from_numpy(Xn.astype(np.float32)).to(DEVICE)

    out = model(xt)  # uses mu for decoding internally
    x_recon_n = out["x_recon"].detach().cpu().numpy()  # (1,2,L)

    x_recon_raw = x_recon_n * X_std + X_mean
    return x_recon_raw[0]  # (2, L)

@torch.no_grad()
def predict_y_and_var(model, X_raw_one, X_mean, X_std):
    Xn = (X_raw_one[None, ...] - X_mean) / X_std
    xt = torch.from_numpy(Xn.astype(np.float32)).to(DEVICE)

    out = model(xt)
    y_mu = out["mu"][0].detach().cpu().numpy()
    y_var = out["var"][0].detach().cpu().numpy() if out["var"] is not None else None
    return y_mu, y_var

def channels_to_dssf(x_ch, e_min, e_max, e_bins, sigma):
    # x_ch shape: (2*n_e, n_k), layout [disp channels..., intensity channels...]
    C, L = x_ch.shape
    nbands = C // 2
    disp_mat = x_ch[:nbands, :]
    inten_mat = np.clip(x_ch[nbands:, :], 0.0, None)
    #return disp_mat[:,0], inten_mat[:,0]
    egrid = np.linspace(e_min, e_max, e_bins)
    egrid, img = create_heatmap(disp_mat, inten_mat, Egrid=egrid, sigma=sigma, normalize=True)
    return egrid, img[0,:]

def find_closest_index(y_raw, target):
    target = np.asarray(target, dtype=np.float32)
    dists = np.linalg.norm(y_raw - target[None, :], axis=1)
    idx = int(np.argmin(dists))
    return idx, dists[idx]

def main():
    os.makedirs("./plots", exist_ok=True)
    X_raw, y_raw = load_raw_data(CSV_PATH)  # (N, C, L)
    N, C, L = X_raw.shape

    # Build noisy truth input for comparison curve
    X_noisy_raw = add_bandwidth_noise(X_raw, sigma=SIGMA_NOISE_FOR_PLOT, seed=NOISE_SEED)

    # Test subset mapping
    #test_global = get_test_global_indices(N, train_ratio=TRAIN_RATIO, split_seed=SPLIT_SEED)
    #test_global = list(range(N))
    test_global = get_train_global_indices(N, train_ratio=TRAIN_RATIO, split_seed=SPLIT_SEED)
    # test_idx, dist = find_closest_index(y_raw[test_global], target)
    # gidx = test_global[test_idx]
    # print(
    #     f"Closest test index: {test_idx}, global index: {gidx}, "
    #     f"y_raw[gidx]={y_raw[gidx]}, dist={dist}"
    # )
    # exit()
    # Load both models + their own standardizers
    model_clean, mean_clean, std_clean = load_model_and_stats(CKPT_CLEAN, input_length=L, in_channels=C)
    model_noisy, mean_noisy, std_noisy = load_model_and_stats(CKPT_NOISY, input_length=L, in_channels=C)

    lanczos_img_true = load_lanczos_imag("./data/THL_Js4_Nk60_Ne100_idx8.csv")
    lanczos_img_0 = load_lanczos_imag("./data/THL_Js4_Nk60_Ne100_idx8_sigma0.0.csv")
    lanczos_img_1 = load_lanczos_imag("./data/THL_Js4_Nk60_Ne100_idx8_sigma0.1.csv")
    #Es_lanczos_0, Is_lanczos_0 = channels_to_dssf(lanczos_img_0[0], e_min=E_MIN, e_max=E_MAX, e_bins=E_BINS, sigma=SIGMA_DSSF)
    #Es_lanczos_1, Is_lanczos_1 = channels_to_dssf(lanczos_img_1[0], e_min=E_MIN, e_max=E_MAX, e_bins=E_BINS, sigma=SIGMA_DSSF)
    Q_cut_ind = 0
    _, N_E_2, _ = lanczos_img_true.shape
    nbands = N_E_2 // 2
    Es_raw_true, Is_raw_true = lanczos_img_true[0][:nbands, Q_cut_ind], lanczos_img_true[0][nbands:, Q_cut_ind]
    Es_raw_0, Is_raw_0 = lanczos_img_0[0][:nbands, Q_cut_ind], lanczos_img_0[0][nbands:, Q_cut_ind]
    Es_raw_1, Is_raw_1 = lanczos_img_1[0][:nbands, Q_cut_ind], lanczos_img_1[0][nbands:, Q_cut_ind]

    for test_idx in TEST_INDICES:

        gidx = test_global[test_idx]

        x_true = X_raw[gidx]        # (C, L)
        x_true_noisy = X_noisy_raw[gidx]
        y_true = y_raw[gidx]

        # Decoder from clean model: feed clean sample
        recon_clean = reconstruct_one(model_clean, x_true, mean_clean, std_clean)

        # Decoder from noisy model: feed noisy sample
        recon_noisy = reconstruct_one(model_noisy, x_true_noisy, mean_noisy, std_noisy)

        y_mu_clean, y_var_clean = predict_y_and_var(model_clean, x_true, mean_clean, std_clean)
        y_mu_noisy, y_var_noisy = predict_y_and_var(model_noisy, x_true_noisy, mean_noisy, std_noisy)

        # Intensity at k=(0,0) 
        Es_true, Is_true = channels_to_dssf(x_true, e_min=E_MIN, e_max=E_MAX, e_bins=E_BINS, sigma=SIGMA_DSSF)
        Es_true_noisy, Is_true_noisy = channels_to_dssf(x_true_noisy, e_min=E_MIN, e_max=E_MAX, e_bins=E_BINS, sigma=SIGMA_DSSF)
        Es_dec_clean, Is_dec_clean = channels_to_dssf(recon_clean, e_min=E_MIN, e_max=E_MAX, e_bins=E_BINS, sigma=SIGMA_DSSF)
        Es_dec_noisy, Is_dec_noisy = channels_to_dssf(recon_noisy, e_min=E_MIN, e_max=E_MAX, e_bins=E_BINS, sigma=SIGMA_DSSF)

        fig, ax = plt.subplots(figsize=(5.5, 4.5), constrained_layout=True)

        ax.plot(Es_true, Is_true,label=r"$S(\mathbf{k}=\Gamma, E)$ ($\sigma$=0.0)", linewidth=2.5, alpha=0.8, color="#1f77b4")
        ax.plot(Es_true_noisy, Is_true_noisy,label=r"$S(\mathbf{k}=\Gamma, E+\epsilon_E)+\epsilon_I$ ($\sigma$=0.1)", 
            linewidth=1.5, alpha=0.8, color="#ff7f0e")
        ax.plot(Es_dec_clean, Is_dec_clean,color="#d62728", label=r"$\tilde{S}(\mathbf{k}=\Gamma, E)$ ($\sigma$=0.0)", 
            linewidth=2, alpha=1.0,ls="--")
        ax.plot(Es_dec_noisy, Is_dec_noisy,label=r"$\tilde{S}(\mathbf{k}=\Gamma, E)$ ($\sigma$=0.1)", 
            linewidth=1.5,alpha=0.8, color="#2ca02c")
        
        #Inset with only the 3 close curves
        axins = inset_axes(ax, width="50%", height="30%", loc="right", borderpad=1.2)
        # axins.plot(Es_true, Is_true,color="#1f77b4", linewidth=2.5, alpha=0.8, label=r"$J_\mathrm{true}$")
        # axins.plot(Es_lanczos_0, Is_lanczos_0, color="#d62728", ls="--", linewidth=2, label=r"$\mu_{\sigma=0.0}(J)$")
        # axins.plot(Es_lanczos_1, Is_lanczos_1, color="#2ca02c", linewidth=1.5, alpha=0.8,label=r"$\mu_{\sigma=0.1}(J)$")
        axins.plot(Es_raw_true, Is_raw_true,color="#1f77b4", linewidth=2.5, alpha=0.8, label=r"$J_\mathrm{true}$")
        axins.plot(Es_raw_0, Is_raw_0, color="#d62728", ls="--", linewidth=2, label=r"$\mu_{\sigma=0.0}(J)$")
        axins.plot(Es_raw_1, Is_raw_1, color="#2ca02c", linewidth=1.5, alpha=0.8,label=r"$\mu_{\sigma=0.1}(J)$")
        axins.set_xlim(7.0, 21.0)
        axins.legend(fontsize=10, loc="upper right", framealpha=0.9)

        txt = (
            rf"$y_{{true}}={np.array2string(y_true, precision=6, separator=', ')}$" "\n"
            rf"$\mu_{{\sigma=0.0}}={np.array2string(y_mu_clean, precision=6, separator=', ')}$" "\n"
            rf"$\sqrt{{\mathrm{{var}}}}_{{\sigma=0.0}}={np.array2string(np.sqrt(y_var_clean), precision=6, separator=', ')}$" "\n"
            rf"$\mu_{{\sigma=0.1}}={np.array2string(y_mu_noisy, precision=6, separator=', ')}$" "\n"
            rf"$\sqrt{{\mathrm{{var}}}}_{{\sigma=0.1}}={np.array2string(np.sqrt(y_var_noisy), precision=6, separator=', ')}$"
        )
        print(txt)
        ax.set_xlabel(r"$\omega/J_1$", fontsize=12)
        ax.set_ylabel(r"Intensity $S(\mathbf{k}=\Gamma, \omega)$", fontsize=12)
        ax.tick_params(axis='both', labelsize=10)
        ax.set_xlim(6, 18)
        ax.legend()
        #fig.tight_layout()
        out_name = f"IE_Ladder_compare_test{test_idx:04d}.pdf"
        out_path = os.path.join("./plots", out_name)
        plt.savefig(out_path, format="pdf", bbox_inches="tight")
        print(f"Saved: {out_path}")
        plt.close()


def parse_maybe_json(v):
    if pd.isna(v):
        return np.nan
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return np.nan
        try:
            out = json.loads(s)   # handles "[...]" or "1.23"
        except json.JSONDecodeError:
            out = float(s)
    else:
        out = float(v)

    # normalize to list if you expect vector-like output
    if isinstance(out, (int, float, np.floating)):
        out = [float(out)]
    return out
if __name__ == "__main__":
    main()