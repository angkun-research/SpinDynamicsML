from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from Latent_THL_Decoder_dash import (
    model, X_mean, X_std, decode_from_latent, create_heatmap,
)

# [J1, J1Δ, Jpmpm, J2], with Jzpm = 0
z = np.array([1.0, 0.2, 0.4, 0.5], dtype=np.float32)

x_dec = decode_from_latent(model, z, X_mean, X_std)
nbands = x_dec.shape[0] // 2
disp_mat = x_dec[:nbands, :]
inten_mat = np.clip(x_dec[nbands:, :], 0.0, None)

hs = np.linspace(0.0, 1.0, x_dec.shape[1])
energies = np.linspace(0.0, 10.0, 1001)

energies, Sperp = create_heatmap(
    disp_mat,
    inten_mat,
    Egrid=energies,
    sigma=0.1,
    normalize= True,
)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "font.size": 12,
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

# Match Julia's 340 × 300 pt figure.
fig, ax = plt.subplots(
    figsize=(340 / 72, 300 / 72),
    layout="constrained",
)

hm = ax.pcolormesh(
    hs,
    energies,
    Sperp.T,
    shading="nearest",
    cmap="viridis",
    vmin=0,
    vmax=max(float(np.max(np.abs(Sperp))), 1e-12),
    rasterized=True,
)

ax.set(
    title=r"$S_\perp(\mathbf{J})$",
    xlabel=r"$(h,h,0)$",
    ylabel=r"$\omega/J_1$",
    ylim=(0, 10),
    xticks=[0, 0.5, 1],
    yticks=np.arange(0, 9, 2),
)
ax.set_xticklabels(["0", "0.5", "1"])
fig.colorbar(hm, ax=ax, pad=0.04, fraction=0.05)

outfile = (
    Path(__file__).resolve().parents[1]
    / "plots"
    / "fig1_prediction.pdf"
)
outfile.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(outfile, dpi=300)
print(f"Wrote {outfile}")

plt.show()
