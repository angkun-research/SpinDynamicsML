import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'

# # CSV_PATH = './data/triangular_anisotropy_uniform_Js4_B60_Nk250_NJ30.csv'
# # choose from 0.0 to 0.1 with 0.01 increments
# sigmas = np.arange(0.0, 0.11, 0.01)
# # mean_vars = np.array([1.544e-5, 3.698e-5, 6.501e-5, 1.486e-4, 2.153e-4, 3.405e-4, 4.950e-4, 7.261e-4, 6.726e-4, 8.820e-4, 1.066e-3])
# # mare = np.array([2.875e-3, 1.343e-2, 1.638e-2, 2.312e-2, 3.173e-2, 3.701e-2, 5.022e-2, 5.970e-2, 5.603e-2, 6.390e-2, 7.455e-2])
# # CSV_PATH = './data/HaoResults/Hao_anisotropy_Js4_uniform_trace_Filter100_Bsats.csv'
# mean_vars = np.array([1.519e-4, 1.422e-4, 1.509e-4, 1.640e-4, 1.773e-4, 2.137e-4, 
#         2.642e-4, 2.231e-4, 2.441e-4, 2.465e-4, 2.773e-4])
# mare = np.array([2.594e-2, 3.045e-2, 3.243e-2, 3.633e-2, 3.974e-2, 4.345e-2, 
#         5.000e-2, 5.546e-2, 5.900e-2, 6.355e-2, 6.643e-2])

# fig, ax = plt.subplots(figsize=(5.5, 4.5))

# ax.plot(sigmas, mean_vars, color="#1f77b4", marker="o", linewidth=2.0,
#         label=r"$\langle \sigma^2 \rangle$")
# ax.plot(sigmas, mare, color="#d62728", marker="s", linewidth=2.0,
#         label=r"MARE")
# ax.set_ylim(1e-4, 1e-1)
# ax.set_xlabel(r"$\sigma_\epsilon$", fontsize=14)
# ax.set_ylabel("Error metric", fontsize=12)
# ax.set_yscale("log")
# ax.grid(alpha=0.3)
# # set tick label size
# ax.tick_params(axis='both', which='major', labelsize=12)
# ax.legend(loc="right", fontsize=12)
# fig.tight_layout()

# plt.savefig("./plots/sigma_vs_metrics_lanczos.pdf", bbox_inches="tight")
# plt.close()


# y_{true}=[ 1., -0.866667,  0.866667, -1.]
# \mu_{\sigma=0.0}=[ 1.008152, -0.874677,  0.864058, -1.004507]
# \sqrt{\mathrm{var}}_{\sigma=0.0}=[0.004378, 0.004239, 0.004361, 0.003883]
# \mu_{\sigma=0.1}=[ 0.99566 , -0.781094,  0.934481, -0.989526]
# \sqrt{\mathrm{var}}_{\sigma=0.1}=[0.007628, 0.031654, 0.049083, 0.015609]


#CSV_PATH = "./data/THED_noise_best20_validation_metrics.csv"
#CSV_PATH = "./data/THED_noise_last20_validation_metrics.csv"
CSV_PATH = "./data/THL_noise_best20_validation_metrics.csv"
#OUTPUT_PATH = "./plots/sigma_vs_metrics_THED.pdf"
OUTPUT_PATH = "./plots/sigma_vs_metrics_THL.pdf"
EXPECTED_EPOCHS_PER_SIGMA = 20

results = pd.read_csv(CSV_PATH)

required_columns = {"sigma", "epoch", "mean_variance","median_variance", "mare"}

summary = (
    results.groupby("sigma", as_index=False)
    .agg(
        mean_variance_mean=("mean_variance", "mean"),
        mean_variance_std=("mean_variance", "std"),
        median_variance_mean=("median_variance", "mean"),
        median_variance_std=("median_variance", "std"),
        mare_mean=("mare", "mean"),
        mare_std=("mare", "std"),
        mean_variance_max=("mean_variance", "max"),
    )
    .sort_values("sigma")
)

print("Final-20-epoch validation summary:")
print(summary.to_string(index=False))

sigmas = summary["sigma"].to_numpy()
mean_vars = summary["mean_variance_mean"].to_numpy()
mean_vars_std = summary["mean_variance_std"].to_numpy()
# mean_vars = summary["mean_variance_max"].to_numpy()
median_vars = summary["median_variance_mean"].to_numpy()
median_vars_std = summary["median_variance_std"].to_numpy()
mare = summary["mare_mean"].to_numpy()
mare_std = summary["mare_std"].to_numpy()

fig, ax = plt.subplots(figsize=(5.5, 4.5))
scale = 10**2
# ax.errorbar(sigmas,median_vars*scale,yerr=median_vars_std*scale,
#     color="#1f77b4",marker="o",markersize=5,
#     linewidth=2.0,capsize=3,label=r"$100\times$ MPV",
# )
ax.errorbar(sigmas,mean_vars*scale,yerr=mean_vars_std*scale,
    color="#1f77b4",marker="o",markersize=5,
    linewidth=2.0,capsize=3,label=r"$100\times\langle \sigma^2 \rangle$",
)
ax.errorbar(sigmas,mare,yerr=mare_std,
    color="#d62728",marker="s",markersize=3,
    linewidth=2.0,capsize=5,label="MARE",
)

ax.set_ylim(5e-3, 1e-1)
ax.set_xlabel(r"$\sigma_\epsilon$", fontsize=14)
ax.set_ylabel("Error metric", fontsize=12)
ax.set_yscale("log")
ax.grid(alpha=0.3, which="both")
ax.tick_params(axis="both", which="major", labelsize=12)
ax.legend(loc="best", fontsize=12)

fig.tight_layout()
plt.savefig(OUTPUT_PATH, bbox_inches="tight")
plt.close()

print(f"Saved figure to: {OUTPUT_PATH}")




# CSV_PATH = "./data/LSWT_gaussian_noise_best20_validation_metrics_epochs200.csv"
# CSV_PATH2 = "./data/LSWT_poisson_noise_best20_validation_metrics_epochs200.csv"
# OUTPUT_PATH = "./plots/sigma_vs_metrics_LSWT.pdf"

# def load_and_summarize(csv_path):
#     results = pd.read_csv(csv_path)
#     required_columns = {"sigma", "epoch", "mean_variance","median_variance", "mare"}

#     summary = (
#         results.groupby("sigma", as_index=False)
#         .agg(
#             mean_variance_mean=("mean_variance", "mean"),
#             mean_variance_std=("mean_variance", "std"),
#             median_variance_mean=("median_variance", "mean"),
#             median_variance_std=("median_variance", "std"),
#             mare_mean=("mare", "mean"),
#             mare_std=("mare", "std"),
#             mean_variance_max=("mean_variance", "max"),
#         )
#         .sort_values("sigma")
#     )

#     print("Final-20-epoch validation summary:")
#     print(summary.to_string(index=False))

#     sigmas = summary["sigma"].to_numpy()
#     mean_vars = summary["mean_variance_mean"].to_numpy()
#     mean_vars_std = summary["mean_variance_std"].to_numpy()
#     median_vars = summary["median_variance_mean"].to_numpy()
#     median_vars_std = summary["median_variance_std"].to_numpy()
#     mare = summary["mare_mean"].to_numpy()
#     mare_std = summary["mare_std"].to_numpy()
#     return sigmas, mean_vars, mean_vars_std, mare, mare_std

# sigmas, mean_vars, mean_vars_std, mare, mare_std = load_and_summarize(CSV_PATH)
# sigmas2, mean_vars2, mean_vars_std2, mare2, mare_std2 = load_and_summarize(CSV_PATH2)

# fig, ax = plt.subplots(figsize=(5.5, 4.5))
# scale = 1 #10**2
# ax.errorbar(sigmas,mean_vars,yerr=mean_vars_std*scale,
#     color="#1f77b4",marker="o",markersize=5,
#     linewidth=2.0,capsize=3,label=r"$\langle \sigma^2 \rangle$",
# )
# ax.errorbar(sigmas,mare,yerr=mare_std,
#     color="#d62728",marker="s",markersize=3,
#     linewidth=2.0,capsize=5,label="MARE",
# )
# ax.errorbar(sigmas2,mean_vars2,yerr=mean_vars_std2*scale,
#     color="#1f77b4",marker="^",markersize=5,linestyle="--",
#     linewidth=2.0,capsize=3,label=r"$\langle \sigma^2 \rangle$ (Poisson)",
# )
# ax.errorbar(sigmas2,mare2,yerr=mare_std2,
#     color="#d62728",marker="d",markersize=3,linestyle="--",
#     linewidth=2.0,capsize=5,label="MARE (Poisson)",
# )

# ax.set_ylim(1e-5, 1e-1)
# ax.set_xlabel(r"$\sigma_\epsilon$ (or $1/T$ Poisson)", fontsize=14)
# ax.set_ylabel("Error metric", fontsize=12)
# ax.set_yscale("log")
# ax.grid(alpha=0.3, which="both")
# ax.tick_params(axis="both", which="major", labelsize=12)
# ax.legend(loc="best", fontsize=12)

# fig.tight_layout()
# plt.savefig(OUTPUT_PATH, bbox_inches="tight")
# plt.close()

# print(f"Saved figure to: {OUTPUT_PATH}")