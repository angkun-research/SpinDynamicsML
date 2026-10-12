# SpinDynamic

Infer spin Hamiltonian parameters from dynamical structure factors using machine learning. Julia scripts use Sunny to generate spectra; Python scripts train and evaluate models on those data.

- `HaoCode/`: triangular-lattice LSWT and Lanczos calculations, parameter sweeps, spectrum filtering, data export, and Fisher information analysis. Shared calculation functions are in `triangular_nlswt.jl`.
- `examples/`: lattice-specific spin-wave calculations and dataset generation for triangular, honeycomb, and spinel models.
- `pythonML/`: PyTorch autoencoders and regressors, preprocessing and noise utilities, training scripts, and prediction plots.
- `HPCtemp/`: experiment scripts for scalar/tensor spectra, noise robustness, training-set size, uncertainty estimates, and decoder visualization.
- `data/`: generated datasets and model checkpoints. `plots/`: saved figures.
- `Project.toml` / `Manifest.toml`: Julia dependency environment. `requirements.txt`: Python dependencies.

The code runs as standalone scripts. Use `julia --project=.` for Julia and the project `.venv` for Python; dataset paths and experiment settings are configured in individual scripts.
