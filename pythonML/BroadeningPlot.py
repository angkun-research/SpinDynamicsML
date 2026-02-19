import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import create_heatmap_from_flat
import os

CSV_PATH = './data/triangular_heisenberg_Js10_h20_qsBZ.csv'
EGRID_POINTS = 300
SIGMAS = [0.1, 0.5, 1.0,2.0]

# load dataframe and parse json columns
df = pd.read_csv(CSV_PATH)
for col in ['data', 'disp', 'Js']:
    df[col] = df[col].apply(json.loads)

# build common Egrid from full dispersion range
all_disp = np.concatenate(df['disp'].values)
E_min, E_max = all_disp.min(), all_disp.max()
Egrid = np.linspace(E_min, E_max, EGRID_POINTS)

# pick first sample
row = df.iloc[0]
flat_disp = row['disp']
flat_data = row['data']
Nq = len(flat_disp)
Nbands = 1

imgs = []
for s in SIGMAS:
    _, img = create_heatmap_from_flat(flat_disp, flat_data,
                                      Nbands=Nbands, Nq=Nq,
                                      Egrid=Egrid, sigma=s, normalize=True)
    imgs.append(img)

# plot side-by-side
fig, axes = plt.subplots(1, len(SIGMAS), figsize=(5 * len(SIGMAS), 4))
if len(SIGMAS) == 1:
    axes = [axes]
for ax, img, s in zip(axes, imgs, SIGMAS):
    im = ax.imshow(img.T, aspect='auto', origin='lower',
                   extent=[Egrid[0], Egrid[-1], 0, Nq])
    ax.set_title(f'sigma = {s}')
    ax.set_ylabel('Energy (meV)')
    ax.set_xlabel('q index')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()
#plt.show()
plt.savefig(os.path.join('../Desktop/', 'broadened_first_sample.png'), dpi=200)
plt.close()