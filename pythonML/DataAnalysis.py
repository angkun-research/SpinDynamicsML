import json
import numpy as np
import pandas as pd
from utils import create_heatmap_from_flat
import matplotlib.pyplot as plt

# --- Load data (same logic as RegressionTest.py) ---
CSV_PATH = './data/triangular_heisenberg_sweep.csv'
df = pd.read_csv(CSV_PATH)

for col in ['data', 'disp']:
    df[col] = df[col].apply(json.loads)

# find max E and min E for setting Egrid later (optional)
all_disp = np.concatenate(df['disp'].values)
E_max = all_disp.max()
E_min = all_disp.min()
print(f"Dispersion energy range: {E_min} to {E_max} meV")

# Build feature matrix X and target J = [J1,J2,J3]
Nq = len(df.iloc[0]['disp']) # assuming 1 band per q-point
Nbands = 1
Egrid = np.linspace(E_min, E_max, 300) #None
images = []
for idx, row in df.iterrows():
    flat_disp = row['disp']
    flat_data = row['data']
    # ensure correct lengths
    if len(flat_disp) != len(flat_data):
        raise ValueError(f"disp/data length mismatch at row {idx}")
    # create_heatmap_from_flat returns (Egrid, img) where img shape = (Nq, len(Egrid))
    _, img = create_heatmap_from_flat(flat_disp, flat_data, Nbands=Nbands, Nq=Nq, Egrid=Egrid, sigma=0.2, normalize=True)
    # flatten the 2D heatmap into a 1D feature vector
    images.append(img.ravel())

X = np.vstack(images).astype(np.float32)
y = df[['J1', 'J2', 'J3']].values.astype(np.float32)

print(X.shape) # (8000,30000)
print(y.shape) # (8000,3)

# # Principal Component Analysis (PCA) for dimensionality reduction
# from sklearn.decomposition import PCA
# pca = PCA(n_components=200)  # Reduce to 200 dimensions
# X_reduced = pca.fit_transform(X)
# # plot eigenvalues with log y scale
# plt.figure()
# plt.scatter(np.arange(1, 201), pca.explained_variance_, marker='o', s=3) 
# plt.yscale('log')
# plt.xlabel('Principal Component')
# plt.ylabel('Eigenvalue')
# plt.title('PCA Eigenvalues')
# plt.grid()
# plt.show()

# --- 1. Efficient Vectorized Correlation Calculation ---

print("Calculating pixel-wise correlations...")

# A. Center the data (subtract mean)
X_mean = X.mean(axis=0)
y_mean = y.mean(axis=0)
X_centered = X - X_mean
y_centered = y - y_mean

# B. Calculate Covariance (Dot product)
# This results in a (30000, 3) matrix
covariance = np.dot(X_centered.T, y_centered) / (X.shape[0] - 1)

# C. Calculate Standard Deviations
X_std = X.std(axis=0, ddof=1)
y_std = y.std(axis=0, ddof=1)

# Handle pixels with zero variance (always 0 intensity) to avoid division by zero
# We set their correlation to 0
epsilon = 1e-9
X_std[X_std < epsilon] = epsilon

# D. Calculate Pearson Correlation Coefficient
# We use broadcasting to divide (30000, 3) by (30000, 1) and (1, 3)
correlation_matrix = covariance / (X_std[:, None] * y_std[None, :])

# correlation_matrix is now (30000, 3).
# Col 0 = Correlation with J1, Col 1 = J2, Col 2 = J3

# --- 2. Reshape and Visualize ---

# Dimensions for reshaping (from your loop)
NE = len(Egrid)  # Should be 300 based on your code
# Nq is already defined in your code

feature_names = ['J1', 'J2', 'J3']

plt.figure(figsize=(15, 5))

for i in range(3):
    # Extract the column for the i-th J parameter
    j_corr_flat = correlation_matrix[:, i]
    
    # Reshape back to the original image dimensions (Nq, NE)
    j_corr_img = j_corr_flat.reshape(Nq, NE)
    print(j_corr_img.shape)  # Should be (Nq, 300)
    # Plotting
    plt.subplot(1, 3, i+1)
    
    # Transpose might be needed depending on how you want axes: 
    # currently x-axis=Energy, y-axis=Momentum
    plt.imshow(j_corr_img.T, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1,
               extent=[0,Nq, E_min, E_max], origin='lower')
    
    plt.colorbar(label='Pearson Correlation')
    plt.title(f'Sensitivity Map for {feature_names[i]}')
    plt.ylabel('Energy (meV)')
    plt.xlabel('Momentum Index (q)')

plt.tight_layout()
plt.show()