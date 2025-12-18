import numpy as np

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
