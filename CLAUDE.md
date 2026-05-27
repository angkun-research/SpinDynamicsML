# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Constraints

- **Only modify files under `HPCtemp/`. Never write, edit, or create files anywhere else.**
- When running Python, always activate the project venv first: `source .venv/bin/activate`

## Project Overview

SpinDynamic is a physics + machine learning project that maps spin dynamical structure factors (SSF) back to Hamiltonian parameters (J1, J2, J3, ...). It has two main parts:

1. **Julia (SpinDynamics.jl)**: Uses the `Sunny` package to compute SSF via Linear Spin Wave Theory (LSWT) for various magnetic lattices
2. **Python (pythonML/)**: PyTorch models (autoencoders and regressors) that learn the inverse mapping from SSF → Hamiltonian parameters

**Data flow**: Given J parameters → Sunny computes SSF (dispersion + intensity) → NN predicts J back from SSF.

## Key Directories

- `src/SpinDynamics.jl` — Julia package stub (minimal, just the module header)
- `examples/*.jl` — LSWT computation scripts for specific lattice models
- `pythonML/` — PyTorch neural network models and training scripts
- `data/` — CSV datasets (SSF data) and `.pth` trained model checkpoints
- `HaoCode/` — Additional LSWT/Julia utility scripts
- `HPCtemp/` — Temporary scratch space (likely for HPC runs)

## Running the Code

### Julia (LSWT computation)

```bash
# From this directory, launch Julia with the project environment
julia --project=.

# Inside Julia:
using .SpinDynamics    # loads the package
using Sunny, CairoMakie # for any LSWT example script
include("examples/TriangularHeisenburg.jl")  # run a specific example
```

Examples in `examples/`:
- `TriangularHeisenburg.jl` — J1-Jn Heisenberg on triangular lattice, computes SSW spectrum
- `HoneycombHeisenburg.jl` — J1-J2-J3 on honeycomb lattice
- `CoRh2O2.jl` — Real material CoRh2O4 (cubic spinel, space group 227), LSWT + powder average
- `AnisotropyLSWT.jl` — Anisotropy terms in LSWT
- `PlotCompare.jl` — Visualization: compare predicted vs true spin wave dispersion
- `collectTriaData.jl` — Collects SSF data for triangular lattice training sets

### Python (NN training/inference)

```bash
# Activate the project's venv
source .venv/bin/activate
python pythonML/<script>.py
```

Dependencies: `numpy, pandas, torch, torchvision, tqdm, matplotlib, scikit-learn, h5py` (from `requirements.txt`)

Training scripts:
- `Regression_noise_test.py` — Main regression test: trains various models (MLP, Conv1D, Transformer, ConvTransformer) on triangular anisotropy data, tests robustness to noise. Switch `MODEL_NAME` to select architecture.
- `RegressionTest.py` — Quick regression benchmark using sklearn's LinearRegression on sweep data
- `AE_Hao_data.py` — Supervised autoencoder training on Hao's dataset (5J parameters)

## Neural Network Models

### Autoencoders (pythonML/AutoEncoderClass.py)
All use supervised latent loss: latent space directly encodes J1, J2, J3 with MSE loss to ground-truth J.

| Class | Input Format | Use Case |
|---|---|---|
| `FCautoencoder` | 1D vector | Baseline fully-connected AE |
| `ConvEncoder` | 1D vector → 2D image | Conv2d encoder with SiLU |
| `TransformerEncoder` | 1D vector → sequence | Transformer encoder/decoder |
| `Conv1DEncoder` | (B, C, L) 1D conv | Conv1D with pooling, handles 1D SSF curves |
| `ConvDownSampling` | (B, C, H, W) 2D | Deep conv encoder with 4 stride-2 conv layers |
| `SimpleConvAutoencoder` | (B, C, H, W) 2D | Shallow conv, no downsampling |
| `ViTAutoencoder` | (B, C, H, W) 2D | Vision transformer with [CLS] token for latent |
| `Conv1DAutoencoder` | (B, C, H, W) 2D | Dual 1D conv (row + column) pipeline |
| `ResNetEncoder` | (B, C, H, W) 2D | ResNet18 backbone, encoder-only (no decoder) |
| `FCBsatDecoder` | 1D vector | Decoder-only: predicts single Bsat value |
| `FCBsats3Decoder` | 1D vector | Decoder-only: predicts 3 Bsat values |

### Regression Models (pythonML/NeuralNetworkClass.py)
All accept input shape `(B, C, L)` where C=2 channels (dispersion + intensity), L = sequence length.

| Class | Architecture |
|---|---|
| `RegressionMLP` | Flatten → Linear → SiLU → Linear (output) |
| `ConvType1` | Conv1d(circular pad) → Flatten → Linear → output |
| `ConvType2` | Conv1d → AvgPool(2) → Flatten → Linear → output |
| `ConvType3` | Conv1d → Pool → Conv1d → Pool → Flatten → Linear → output |
| `TransformerType1` | Linear proj → TransformerEncoder → Flatten → Linear → output |
| `ConvTransformer` | Conv1d → SiLU → TransformerEncoder → Flatten → Linear → output |

### Shared Utilities
- `pythonML/utils.py` — `pick_best_device()` (selects GPU with most free RAM), `create_heatmap()` (SSF dispersion + intensity → 2D heatmap via Gaussian broadening)
- `pythonML/VisualRes.py`, `VisualImage.py`, `DataAnalysis.py` — Visualization and analysis helpers

## Data Format

Training data CSV files have columns: `data` (SSF intensity, JSON array), `disp` (SSF dispersion, JSON array), `Js` (true J parameters, JSON array). The `collect*.jl` scripts in `examples/` generate these.

Model checkpoints (`.pth`) contain `model_state`, `X_mean`, `X_std` for inference.

## Lattice Conventions

All Julia examples use the `Sunny` package. Common patterns:
- Lattice vectors: `lattice_vectors(a, a, 100a, 90, 90, 120)` for quasi-2D (long c-axis)
- Exchange bonds: `set_exchange!(sys, J_value, Bond(site1, site2, translation_vec))`
- SSF measurement: `ssf_perp(sys)` for perpendicular component
- Spin wave theory: `swt = SpinWaveTheory(sys; measure=ssf_perp(sys))`
- Spectrum: `intensities_bands(swt, path)` returns `res.disp` and `res.data`

## Important Notes

- The Julia package `src/SpinDynamics.jl` is essentially a stub (exports only `greet()`). All real Julia code is in `examples/` and `HaoCode/` as standalone scripts.
- Autoencoders constrain latent space to [-1,1] via `tanh` and use relative or absolute MSE for latent loss.
- Regression models output raw (unconstrained) values; no output activation is applied.
- The `Conv1DEncoder` class has both an old (commented out) and new implementation — the new one uses lazy decoder initialization via `pooled_length`.
- There are two overlapping `run_experiment` function definitions in `Regression_noise_test.py` — the second one (line 200) is the active version.

## Noise Test Results

### mlp
| Train | Sigma | Train Loss | Train RelErr | Test | Test Loss | Test RelErr |
|-------|-------|------------|--------------|------|-----------|-------------|
| clean | -- | 7.291272e-06 | 5.756120e-03 | clean | 7.173790e-03 | 1.570934e-01 |
| noisy | 0.01 | 1.589821e-04 | 2.868404e-02 | clean | 2.784182e-05 | 1.318918e-02 |
| noisy | 0.01 | 1.589821e-04 | 2.868404e-02 | noisy | 1.603404e-04 | 2.916673e-02 |
| noisy | 0.02 | 5.499984e-04 | 5.285325e-02 | clean | 4.837651e-05 | 1.616337e-02 |
| noisy | 0.02 | 5.499984e-04 | 5.285325e-02 | noisy | 5.409429e-04 | 5.238613e-02 |
| noisy | 0.03 | 1.117588e-03 | 7.341692e-02 | clean | 7.683837e-05 | 1.683364e-02 |
| noisy | 0.03 | 1.117588e-03 | 7.341692e-02 | noisy | 1.092549e-03 | 7.273339e-02 |

