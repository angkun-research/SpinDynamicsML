# SpinDynamicsML

**Machine learning for inferring physical model parameters from noisy scattering spectra.**

[**Preprint: Learning Spin Hamiltonians from Interacting Magnon Spectra**](https://arxiv.org/abs/2611.25321)

SpinDynamicsML combines Julia spin-dynamics simulations with PyTorch inverse models to recover microscopic interaction parameters from dynamical spin structure factors—the energy- and momentum-dependent spectra probed by neutron scattering. It also asks which parameters the available measurements can actually distinguish, how noise affects predictions, and how the choice of simulation model affects generalization.

The workflow spans physical modeling, synthetic dataset generation, spectral preprocessing, neural-network and classical regression benchmarks, predictive uncertainty, and learned forward reconstruction.

The benchmarks use synthetic neutron-scattering spectra.

## The problem

Recovering a physical model from measured spectra is an inverse problem: many candidate parameter sets may produce similar responses, and repeatedly running a many-body simulation during parameter fitting is expensive. A predictive model must account for both measurement ambiguity and the fidelity of its training simulations.

This project studies a frustrated triangular-lattice magnet with four or five unknown interaction parameters. It combines an analysis of identifiability with supervised regression and a learned forward model for checking and visualizing predictions.

## Approach

1. **Generate synthetic spectra.** Use linear spin-wave theory (LSWT) and the truncated Hilbert-space ladder (THL) method to simulate magnetic responses across a Hamiltonian parameter domain.
2. **Assess identifiability.** Compare measurement geometries and polarization channels, using Fisher-information matrices to diagnose locally poorly constrained parameter combinations.
3. **Build compact spectral representations.** Convert spectra into model inputs using adaptive energy discretization or Gaussian-mixture representations.
4. **Learn the inverse map.** Benchmark fully connected, Conv1D, transformer, and convolution-transformer networks alongside classical regression methods.
5. **Model noise and uncertainty.** Explore Gaussian noise augmentation, Poisson counting noise, and input-dependent predictive means and variances.
6. **Reconstruct the response.** Pass inferred parameters through a learned forward decoder for rapid spectral visualization and a self-consistency check before validation with the physical solver.

## Selected findings

- **Omitting magnon interactions can change the inferred exchange.** In the paper’s example, an LSWT-trained network inferred $J_{zz}/J_1=-0.32$ from a synthetic interacting spectrum generated at $J_{zz}/J_1=+0.25$, even though the LSWT dispersion at the inferred couplings closely matched the peak energies. [Preprint, Fig. 2](https://arxiv.org/html/2611.25321v1#S3.F2)
- **The momentum cut can determine whether a coupling’s sign is identifiable.** The unpolarized $(h,h,0)$ cut leaves the sign of $J_{z\pm}$ ambiguous, while the unpolarized $(2h,-h,1)$ cut can distinguish it. [Preprint, Sec. IV](https://arxiv.org/html/2611.25321v1#S4)
- **Conv1D had the lowest error among the four neural architectures tested.** In the four-parameter unpolarized THL benchmark, its mean absolute relative error was **2.04% on clean spectra** and **4.98% on spectra with Gaussian noise**. [Preprint, Table 2](https://arxiv.org/html/2611.25321v1#S5.T2)
- **The best classical baseline depends on the solver.** Support-vector regression had the lowest error among the classical methods in the THL benchmark, and Gaussian-process regression had the lowest error in the corresponding LSWT benchmark. [Preprint, Sec. V.6](https://arxiv.org/html/2611.25321v1#S5.SS6)
- **A learned decoder supports rapid exploration.** It reconstructs spectra from inferred couplings for visualization and a self-consistency check. A direct calculation with the physical solver can then test those couplings against the input spectrum. [Preprint, Sec. V.4](https://arxiv.org/html/2611.25321v1#S5.SS4)

## Code highlights

| Area | Selected files |
| --- | --- |
| Spin-dynamics calculations | [THL](JuliaLadder/triangular_ladder.jl), [truncated Hilbert-space exact diagonalization (THED)](JuliaCode/triangular_nlswt.jl) |
| Identifiability and Fisher-information analysis | [`JuliaLadder/FIM_check.jl`](JuliaLadder/FIM_check.jl) |
| Spectral discretization and preprocessing | [`JuliaCode/UtilDiscretizations.jl`](JuliaCode/UtilDiscretizations.jl), [`JuliaCode/filteredTensorGMM.jl`](JuliaCode/filteredTensorGMM.jl) |
| Neural regression architectures | [`pythonML/NeuralNetworkClass.py`](pythonML/NeuralNetworkClass.py) |
| Predictive uncertainty and forward decoding | [`pythonML/AutoEncoderClassNew.py`](pythonML/AutoEncoderClassNew.py), [`pythonML/AE_Lanczos_variance.py`](pythonML/AE_Lanczos_variance.py) |
| Classical regression benchmarks | [`pythonML/Classic_ML_Regression.py`](pythonML/Classic_ML_Regression.py) |
| Gaussian noise and Poisson counting experiments | [`pythonML/Regression_noise_test.py`](pythonML/Regression_noise_test.py), [`pythonML/Poisson_noise_test.py`](pythonML/Poisson_noise_test.py) |
| Cross-solver prediction comparison | [`pythonML/Load_LSWTNN_Solvers.py`](pythonML/Load_LSWTNN_Solvers.py) |
| Interactive spectral reconstruction | [THL dashboard](pythonML/Latent_THL_Decoder_dash.py), [THED dashboard](pythonML/Latent_Lanczos_Decoder_dash.py) |

Additional lattice-specific calculations and dataset-generation scripts are in [`examples/`](examples/). The repository is organized as standalone research scripts.

**Core tools:** Python, PyTorch, NumPy, pandas, scikit-learn, Julia, Sunny, and Makie. Interactive decoder visualizations use Dash and Plotly.

## Working with the code

Run scripts from the repository root unless a script specifies a different working directory. The THL environment in [`JuliaLadder/Project.toml`](JuliaLadder/Project.toml) and [`JuliaLadder/Manifest.toml`](JuliaLadder/Manifest.toml) uses the `corrections` revision of [Sunny](https://github.com/SunnySuite/Sunny.jl). The root [`Project.toml`](Project.toml) and [`Manifest.toml`](Manifest.toml) use the `non-perturbative2` branch of [Hao Zhang's Sunny fork](https://github.com/Hao-Phys/Sunny.jl). Python dependencies are listed in [`requirements.txt`](requirements.txt), with Dash and Plotly additionally needed for the interactive decoder scripts.

Dataset paths, spectral dimensions, model architectures, and experiment settings are configured within individual scripts. Generated datasets and trained checkpoints are not included in the repository, so training and evaluation scripts require the corresponding local data. These files provide implementation examples rather than a packaged command-line interface or a bundled end-to-end demo.

## Acknowledgments

Hao Zhang provided the original THED Julia code on which the project's many-body calculations build. Spin-dynamics calculations use the [Sunny](https://github.com/SunnySuite/Sunny.jl) ecosystem and the specialized fork linked above.

## Citation

Ang-Kun Wu, David A. Dahlbom, Kipton Barros, Cristian D. Batista, and Hao Zhang.
*Learning Spin Hamiltonians from Interacting Magnon Spectra* (2026).
[arXiv:2611.25321](https://arxiv.org/abs/2611.25321).
