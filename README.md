# SpinDynamicsML

**Machine learning for inferring physical model parameters from noisy scattering spectra.**

SpinDynamicsML combines Julia spin-dynamics simulations with PyTorch inverse models to recover microscopic interaction parameters from dynamical spin structure factors—the energy- and momentum-dependent spectra probed by neutron scattering. It also asks which parameters the available measurements can actually distinguish, how noise affects predictions, and how the choice of simulation model affects generalization.

The workflow spans physical modeling, synthetic dataset generation, spectral preprocessing, neural-network and classical regression benchmarks, predictive uncertainty, and learned forward reconstruction.

**Status:** Active research code accompanying a **manuscript in preparation**. The benchmarks use synthetic spectra; application to experimental measurements is a future direction.

## The problem

Recovering a physical model from measured spectra is an inverse problem: many candidate parameter sets may produce similar responses, and repeatedly running a many-body simulation during parameter fitting is expensive. A predictive model must account for both measurement ambiguity and the fidelity of its training simulations.

This project studies a frustrated triangular-lattice magnet with four or five unknown interaction parameters. It combines an analysis of identifiability with supervised regression and a learned forward model for checking and visualizing predictions.

## Approach

1. **Generate synthetic spectra.** Use linear spin-wave theory (LSWT) and truncated Hilbert exact diagonalization (THED) to simulate magnetic responses across a Hamiltonian parameter domain.
2. **Assess identifiability.** Compare measurement geometries and polarization channels, using Fisher-information matrices to diagnose locally poorly constrained parameter combinations.
3. **Build compact spectral representations.** Convert spectra into model inputs using adaptive energy discretization or Gaussian-mixture representations.
4. **Learn the inverse map.** Benchmark fully connected, Conv1D, transformer, and convolution-transformer networks alongside classical regression methods.
5. **Model noise and uncertainty.** Explore Gaussian noise augmentation, Poisson counting noise, and input-dependent predictive means and variances.
6. **Reconstruct the response.** Pass inferred parameters through a learned forward decoder for rapid spectral visualization and a self-consistency check before validation with the physical solver.

## Selected findings

- **Measurement design determines what can be inferred.** In the regimes studied, some in-plane unpolarized measurements cannot distinguish the sign of an off-diagonal exchange coupling. Polarized channels can resolve that ambiguity; out-of-plane measurements can improve local conditioning.
- **A compact convolutional model performs well.** Conv1D achieves the lowest error among the four neural architectures tested on the THED architecture benchmark, for both clean and noise-augmented data.
- **Classical baselines remain competitive.** Support-vector regression performs well on the THED benchmark, while Gaussian-process regression gives the lowest error on the LSWT benchmark. Model choice depends on the forward solver, data representation, and noise conditions.
- **Training-solver fidelity matters.** Cross-solver evaluation exposes errors when an LSWT-trained model receives THED spectra. Input-dependent predictive uncertainty does not account for this forward-model mismatch.
- **A learned decoder supports rapid exploration.** It reconstructs spectra from predicted parameters for screening and visualization; final estimates still require validation with the physical solver.

The current manuscript reports the following **clean-data neural architecture benchmark** for four-parameter inference from unpolarized THED spectra, using 12 momentum points, 100 retained energy points, and hidden width 64:

| Architecture | Test mean squared error | Test mean absolute relative error |
| --- | ---: | ---: |
| Fully connected | 9.3 × 10⁻⁴ | 5.6% |
| **Conv1D** | **2.4 × 10⁻⁴** | **2.9%** |
| Transformer | 3.0 × 10⁻⁴ | 3.7% |
| Convolution-transformer | 4.2 × 10⁻⁴ | 4.0% |

These results describe the specified synthetic-data benchmark and may be updated as the manuscript is finalized.

## Code highlights

| Area | Selected files |
| --- | --- |
| Spin-dynamics calculations and shared THED helpers | [`JuliaCode/triangular_nlswt.jl`](JuliaCode/triangular_nlswt.jl) |
| Identifiability and Fisher-information analysis | [`JuliaCode/FIM_check.jl`](JuliaCode/FIM_check.jl) |
| Spectral discretization and preprocessing | [`JuliaCode/UtilDiscretizations.jl`](JuliaCode/UtilDiscretizations.jl), [`JuliaCode/filteredTensorGMM.jl`](JuliaCode/filteredTensorGMM.jl) |
| Neural regression architectures | [`pythonML/NeuralNetworkClass.py`](pythonML/NeuralNetworkClass.py) |
| Predictive uncertainty and forward decoding | [`pythonML/AutoEncoderClassNew.py`](pythonML/AutoEncoderClassNew.py), [`pythonML/AE_Lanczos_variance.py`](pythonML/AE_Lanczos_variance.py) |
| Classical regression benchmarks | [`pythonML/Classic_ML_Regression.py`](pythonML/Classic_ML_Regression.py) |
| Gaussian noise and Poisson counting experiments | [`pythonML/Regression_noise_test.py`](pythonML/Regression_noise_test.py), [`pythonML/Poisson_noise_test.py`](pythonML/Poisson_noise_test.py) |
| Cross-solver prediction comparison | [`pythonML/Load_LSWTNN_Solvers.py`](pythonML/Load_LSWTNN_Solvers.py) |
| Interactive spectral reconstruction | [`pythonML/Latent_Lanczos_Decoder_dash.py`](pythonML/Latent_Lanczos_Decoder_dash.py) |

Additional lattice-specific calculations and dataset-generation scripts are in [`examples/`](examples/). The repository is organized as standalone research scripts.

**Core tools:** Python, PyTorch, NumPy, pandas, scikit-learn, Julia, Sunny, and Makie. Interactive decoder visualizations use Dash and Plotly.

## Working with the code

Run scripts from the repository root. Julia dependencies are recorded in [`Project.toml`](Project.toml) and [`Manifest.toml`](Manifest.toml); the project uses the `non-perturbative2` branch of [Hao Zhang's Sunny fork](https://github.com/Hao-Phys/Sunny.jl). Python dependencies are listed in [`requirements.txt`](requirements.txt), with Dash and Plotly additionally needed for the interactive decoder scripts.

Dataset paths, spectral dimensions, model architectures, and experiment settings are configured within individual scripts. Generated datasets and trained checkpoints are not included in the repository, so training and evaluation scripts require the corresponding local data. These files provide implementation examples rather than a packaged command-line interface or a bundled end-to-end demo.

## Acknowledgments

Hao Zhang provided the original THED Julia code on which the project's many-body calculations build. Spin-dynamics calculations use the [Sunny](https://github.com/SunnySuite/Sunny.jl) ecosystem and the specialized fork linked above.

## Associated manuscript

**Identifiability and Neural-Network Inference of Microscopic Hamiltonians from Dynamical Spin Structure Factors**

Ang-Kun Wu, David A. Dahlbom, Cristian D. Batista, and Hao Zhang.

**Manuscript in preparation.** An arXiv link will be added when available.
