import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'

# CSV_PATH = './data/triangular_anisotropy_uniform_Js4_B60_Nk250_NJ30.csv'
# choose from 0.0 to 0.1 with 0.01 increments
sigmas = np.arange(0.0, 0.11, 0.01)
mean_vars = np.array([1.544e-5, 3.698e-5, 6.501e-5, 1.486e-4, 2.153e-4, 3.405e-4, 4.950e-4, 7.261e-4, 6.726e-4, 8.820e-4, 1.066e-3])
mare = np.array([2.875e-3, 1.343e-2, 1.638e-2, 2.312e-2, 3.173e-2, 3.701e-2, 5.022e-2, 5.970e-2, 5.603e-2, 6.390e-2, 7.455e-2])

fig, ax = plt.subplots(figsize=(5.5, 4.5))

ax.plot(sigmas, mean_vars, color="#1f77b4", marker="o", linewidth=2.0,
        label=r"$\langle \sigma^2 \rangle$")
ax.plot(sigmas, mare, color="#d62728", marker="s", linewidth=2.0,
        label=r"MARE")

ax.set_xlabel(r"$\sigma_\epsilon$")
ax.set_ylabel("Error metric")
ax.set_yscale("log")
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()

plt.savefig("./plots/sigma_vs_metrics.pdf", bbox_inches="tight")
plt.close()