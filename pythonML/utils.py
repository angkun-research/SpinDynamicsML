import numpy as np
import subprocess
import torch

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

def create_heatmap(disp_mat, data_mat, Egrid=None, sigma=0.1, normalize=False):
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

    # vectorized accumulation of Gaussians per q-point
    # for each q (row) sum contributions of all bands
    for j in range(Nq):
        E0 = E_mat[j]        # (Nbands,)
        A  = I_mat[j]        # (Nbands,)
        # broadcast to (Nbands, len(Egrid)) then sum over bands
        contrib = A[:, None] * np.exp(-0.5 * ((Egrid[None, :] - E0[:, None]) / sigma) ** 2)
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
