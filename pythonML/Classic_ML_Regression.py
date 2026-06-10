import os
# Propagate warning suppression to joblib worker processes (n_jobs=-1)
os.environ['PYTHONWARNINGS'] = 'ignore'
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, MultiTaskElasticNet
from sklearn.cross_decomposition import PLSRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.multioutput import MultiOutputRegressor

warnings.filterwarnings('ignore') # Suppress warnings for cleaner output

# ------ Config ------
CSV_PATH    = './data/triangular_anisotropy_uniform_Js4_B50_Nk20_NJ30.csv'
SEED        = 1234
TRAIN_RATIO = 0.9
GPR_MAX_TRAIN = 2000

# ------ Model registry ------
def build_models(seed):
    # Wide noise bounds: simulated data is nearly noise-free, so the optimal
    # noise_level can be far below the default lower bound (1e-5).
    rbf_kernel = (RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1e4))
                  + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-12, 1e1)))
    return {
        'Ridge':            Ridge(alpha=1.0),
        'ElasticNet':       MultiTaskElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000),
        'PLS':              PLSRegression(n_components=20),
        'KernelRidge':      KernelRidge(alpha=0.1, kernel='rbf', gamma=None),
        'SVR':              MultiOutputRegressor(SVR(kernel='rbf', C=10.0, epsilon=0.01), n_jobs=-1),
        'GPR':              MultiOutputRegressor(
                                GaussianProcessRegressor(kernel=rbf_kernel, n_restarts_optimizer=2,
                                                         normalize_y=True,
                                                         random_state=seed), n_jobs=-1),
        'RandomForest':     RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=seed),
        'GradientBoosting': MultiOutputRegressor(
                                GradientBoostingRegressor(n_estimators=200, max_depth=4,
                                                          random_state=seed), n_jobs=-1),
        'kNN':              KNeighborsRegressor(n_neighbors=10, n_jobs=-1),
    }

# ------ Data utilities ------
def load_raw_data(csv_path):
    df = pd.read_csv(csv_path)
    for col in ['data', 'disp', 'Js']:
        df[col] = df[col].apply(json.loads)
    disp  = np.array(df['disp'].tolist(),  dtype=np.float32)
    inten = np.array(df['data'].tolist(),  dtype=np.float32)
    X = np.concatenate([disp, inten], axis=1)
    y = np.array(df['Js'].tolist(), dtype=np.float32)
    y[:, 1] *= y[:, 0]
    return X, y

def split_indices(n, train_ratio=0.9, seed=1234):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_train = int(train_ratio * n)
    return perm[:n_train], perm[n_train:]

def rel_error(y_pred, y_true):
    return np.mean(np.abs(y_pred - y_true) / (np.abs(y_true) + 1e-12))

def mse(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)

# ------ Main ------
np.random.seed(SEED)

X, y = load_raw_data(CSV_PATH)
train_idx, test_idx = split_indices(len(X), TRAIN_RATIO, SEED)
X_train, y_train = X[train_idx], y[train_idx]
X_test,  y_test  = X[test_idx],  y[test_idx]

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

print(f"Dataset: {CSV_PATH}")
print(f"Train: {len(y_train)}, Test: {len(y_test)}, Input dim: {X_train_s.shape[1]}\n")
print(f"{'Model':<20} {'Train MSE':>14} {'Train RelErr':>13} {'Test MSE':>14} {'Test RelErr':>13}")
print(f"{'-'*20} {'-'*14} {'-'*13} {'-'*14} {'-'*13}")

for name, model in build_models(SEED).items():
    if name == 'GPR' and len(y_train) > GPR_MAX_TRAIN:
        rng = np.random.default_rng(SEED)
        idx = rng.choice(len(y_train), GPR_MAX_TRAIN, replace=False)
        X_fit, y_fit = X_train_s[idx], y_train[idx]
    else:
        X_fit, y_fit = X_train_s, y_train

    model.fit(X_fit, y_fit)

    tr_mse = mse(model.predict(X_fit),    y_fit)
    tr_rel = rel_error(model.predict(X_fit),    y_fit)
    te_mse = mse(model.predict(X_test_s), y_test)
    te_rel = rel_error(model.predict(X_test_s), y_test)

    gpr_note = f" (cap {GPR_MAX_TRAIN})" if name == 'GPR' and len(y_train) > GPR_MAX_TRAIN else ""
    print(f"{name+gpr_note:<20} {tr_mse:>14.6e} {tr_rel:>13.6e} {te_mse:>14.6e} {te_rel:>13.6e}")