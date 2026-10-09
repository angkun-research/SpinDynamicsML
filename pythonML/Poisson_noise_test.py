import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pythonML.utils import add_poisson_noise_lswt

CSV_PATH = "./data/triangular_anisotropy_uniform_Js4_Bdep_Nk12_NJ30.csv"
TS = [100, 1000, 100000]
SAMPLE_INDEX = 0

# Load and filter as in Regression_multichannel.py.
df = pd.read_csv(CSV_PATH)
disp = np.array(df["disp"].apply(json.loads).tolist(), dtype=np.float32)
inten = np.array(df["data"].apply(json.loads).tolist(), dtype=np.float32)
Js = np.array(df["Js"].apply(json.loads).tolist(), dtype=np.float32)

X = np.stack([disp, inten], axis=1)
mask = (Js != 0).all(axis=1)
X, Js = X[mask], Js[mask]

# Keep one sample: shape (1, 2, Nq).
sample = X[SAMPLE_INDEX:SAMPLE_INDEX + 1]
q_index = np.arange(sample.shape[-1])

fig, axes = plt.subplots(
    2, len(TS), figsize=(4 * len(TS), 6),
    sharex=True, sharey="row", squeeze=False,
)

for col, T in enumerate(TS):
    noisy = add_poisson_noise_lswt(
        sample, T=T, fwhm=0.8, bgrate=1.0, seed=1234
    )

    for row in range(2):
        ax = axes[row, col]
        ax.plot(q_index, sample[0, row], "k-", lw=2, label="Original LSWT")
        ax.plot(q_index, noisy[0, row], "o--", ms=4, label="Poisson + fit")
        ax.grid(alpha=0.3)

    axes[0, col].set_title(f"T = {T:g}")
    axes[1, col].set_xlabel("q-point index")

axes[0, 0].set_ylabel("Dispersion (meV)")
axes[1, 0].set_ylabel("Integrated intensity")
axes[0, 0].legend()

fig.suptitle(
    f"LSWT sample {SAMPLE_INDEX}, Js = {Js[SAMPLE_INDEX].tolist()}"
)
fig.tight_layout()
OUTPUT_PATH = f"./plots/Poisson_noise_sample_{SAMPLE_INDEX}.png"
plt.savefig(OUTPUT_PATH, bbox_inches="tight")
plt.close()