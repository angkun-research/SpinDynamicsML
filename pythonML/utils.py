import numpy as np
import subprocess
import torch
from torch.utils.data import TensorDataset

def pick_best_device(min_free_gb=10.0):
    if not torch.cuda.is_available():
        return torch.device("cpu")

    query = [
        "nvidia-smi",
        "--query-gpu=index,utilization.gpu,memory.free,memory.total",
        "--format=csv,noheader,nounits",
    ]

    try:
        output = subprocess.check_output(query, encoding="utf-8")
    except Exception as exc:
        print(f"Could not query nvidia-smi: {exc}")
        return torch.device("cpu")

    best_gpu = None
    best_score = None

    for line in output.strip().splitlines():
        gpu_id, gpu_util, free_mem, total_mem = [x.strip() for x in line.split(",")]
        gpu_id = int(gpu_id)
        gpu_util = float(gpu_util)
        free_mem = float(free_mem) / 1024.0  # MB -> GB
        total_mem = float(total_mem) / 1024.0

        print(f"GPU {gpu_id}: util={gpu_util:.0f}% free={free_mem:.2f} GB / total={total_mem:.2f} GB")

        if free_mem < min_free_gb:
            continue

        score = (gpu_util, -free_mem)
        if best_score is None or score < best_score:
            best_score = score
            best_gpu = gpu_id

    if best_gpu is None:
        print("No GPU meets the free-memory threshold. Falling back to CPU.")
        return torch.device("cpu")

    print(f"Selecting GPU {best_gpu}")
    return torch.device(f"cuda:{best_gpu}")

def create_heatmap(disp_mat, data_mat, Egrid=None, sigma=0.2, normalize=False):
    """
    disp_mat, data_mat: numpy arrays with shape (Nbands, Nq)
    returns: Egrid (1D), img (Nq x len(Egrid))
    """
    if Egrid is None:
        Egrid = np.linspace(0.0, 10.0, 300)
    Nbands, Nq = disp_mat.shape

    # rows = q index, cols = band index
    E_mat = disp_mat.T  # shape (Nq, Nbands)
    I_mat = data_mat.T

    img = np.zeros((Nq, len(Egrid)), dtype=np.float64)
    normalization = 1.0 / (sigma * np.sqrt(2.0 * np.pi))
    # vectorized accumulation of Gaussians per q-point
    # for each q (row) sum contributions of all bands
    for j in range(Nq):
        E0 = E_mat[j]        # (Nbands,)
        A  = I_mat[j]        # (Nbands,)
        # broadcast to (Nbands, len(Egrid)) then sum over bands
        contrib = A[:, None] * normalization * np.exp(-0.5 * ((Egrid[None, :] - E0[:, None]) / sigma) ** 2)
        img[j, :] = contrib.sum(axis=0)

    if normalize:
        img = img - img.min()
        s = img.max()
        if s > 0:
            img = img / s

    return Egrid, img

def create_heatmap_from_flat(flat_disp, flat_data, Nbands, Nq, Egrid=None, sigma=0.1, normalize=False):
    disp = np.array(flat_disp, dtype=np.float64).reshape((Nbands, Nq))
    data = np.array(flat_data, dtype=np.float64).reshape((Nbands, Nq))
    return create_heatmap(disp, data, Egrid=Egrid, sigma=sigma, normalize=normalize)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # deterministic behavior
    torch.use_deterministic_algorithms(True)   # raise error on nondeterministic ops
    torch.backends.cudnn.deterministic = True # disable nondeterministic CuDNN algorithms
    torch.backends.cudnn.benchmark = False # disable CuDNN auto-tuning (introduces nondeterminism)


def fit_standardizer(X_train_raw):
    X_mean = X_train_raw.mean(axis=0, keepdims=True)
    X_std = X_train_raw.std(axis=0, keepdims=True) + 1e-8
    return X_mean, X_std

def apply_standardizer(X, X_mean, X_std):
    return (X - X_mean) / X_std

def add_bandwidth_noise(X, sigma, seed=1234, noise_type='default'):
    rng = np.random.default_rng(seed)
    eps = rng.normal(0.0, 1.0, size=X.shape).astype(np.float32)

    max_vals = X.max(axis=2, keepdims=True)
    min_vals = X.min(axis=2, keepdims=True)
    bandwidth = max_vals - min_vals

    if noise_type == 'default':
        # Treat second half (intensity channels) as max-based bandwidth
        c_half = X.shape[1] // 2
        bandwidth[:, c_half:, :] = max_vals[:, c_half:, :]
    elif noise_type == 'gmm':
        # Treat the gaussian weight with no noise
        c_23 = 2 * X.shape[1] // 3
        bandwidth[:, c_23:, :] = 0

    return X + sigma * bandwidth * eps
    

def make_dataset(X, y):
    return TensorDataset(
        torch.from_numpy(X.astype(np.float32)),
        torch.from_numpy(y.astype(np.float32)),
    )
