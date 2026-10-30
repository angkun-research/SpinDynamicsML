import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pythonML.NeuralNetworkClass import (RegressionMLP, ConvType1, ConvType2, ConvType3,
                                         ConvType4, TransformerType1, ConvTransformer)
from pythonML.utils import apply_standardizer

# Evaluate a trained regression checkpoint (Regression_multichannel.py) on a single
# Hamiltonian parameter set computed by two solvers:
#   LSWT : data/LSWT_Js4_Bsat_Nk12_test_1.csv  (Save_Single_LSWT.jl)
#   THED : data/THED_Js4_Nk12_Ne1.csv          (Save_Single_THED_as_LSWT.jl)
#
# Usage:
#   source .venv/bin/activate
#   python pythonML/Load_LSWTNN_Solvers.py [--model conv1] [--hidden 64]

MODELS = {
    'mlp': RegressionMLP, 'conv1': ConvType1, 'conv2': ConvType2, 'conv3': ConvType3,
    'conv4': ConvType4, 'transformer1': TransformerType1, 'conv_transformer': ConvTransformer,
}
PARAM_NAMES = ['J1', 'Delta*J1', 'Jpmpm', 'J2']

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', default='./data/Conv1D_anisotropy_LSWT_scalar_Js4_hidden64_Nk60_Bdep_unitless.pth') 
# Conv1D_anisotropy_LSWT_scalar_Js4_hidden64_Nk12_Bdep.pth')
parser.add_argument('--model', default='conv1', choices=MODELS.keys())
parser.add_argument('--hidden', type=int, default=64)
parser.add_argument('--lswt_csv', default='./data/LSWT_Js4_Bdep_Nk60_test_unitless.csv')
parser.add_argument('--thed_csv', default='./data/THL_Js4_Nk60_Ne1.csv')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_solver_csv(path):
    """Returns X (N, 2, L) = [disp, intensity] and y (N, 4) with Delta -> Delta*J1 (as in training)."""
    df = pd.read_csv(path)
    for col in ['data', 'disp', 'Js']:
        df[col] = df[col].apply(json.loads)
    disp = np.array(df['disp'].tolist(), dtype=np.float32)
    inten = np.array(df['data'].tolist(), dtype=np.float32)
    y = np.array(df['Js'].tolist(), dtype=np.float32)
    y[:, 1] *= y[:, 0]
    #return np.stack([disp, inten], axis=1), y
    return np.stack([disp], axis=1), y


def load_model(ckpt_path, model_name, hidden_dim, in_channels, input_length, output_dim):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = MODELS[model_name](in_channels=in_channels, input_length=input_length,
                               hidden_dim=hidden_dim, output_dim=output_dim).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    return model, ckpt['X_mean'], ckpt['X_std']


@torch.no_grad()
def predict(model, X, X_mean, X_std):
    X_in = apply_standardizer(X, X_mean, X_std).astype(np.float32)
    return model(torch.from_numpy(X_in).to(device)).cpu().numpy()


solvers = {'LSWT': args.lswt_csv, 'THED': args.thed_csv}
data = {name: load_solver_csv(path) for name, path in solvers.items()}
print(f"Loaded solver CSVs: {list(data.keys())}")
X_lswt, _ = data['LSWT']
X_thl, _ = data['THED']
#X_thl[:, 1, :] = X_lswt[:, 1, :].copy()
print(X_lswt)
print(X_thl)
print(X_lswt - X_thl)

# Checkpoint's saved X_mean has shape (1, C, L), which fixes the input shape
ckpt_meta = torch.load(args.ckpt, map_location='cpu', weights_only=False)
_, C, L = np.asarray(ckpt_meta['X_mean']).shape
out_dim = 4
model, X_mean, X_std = load_model(args.ckpt, args.model, args.hidden, C, L, out_dim)
print(f"Loaded {args.ckpt} ({args.model}, hidden={args.hidden}, in=({C},{L}), out={out_dim})\n")

for name, (X, y) in data.items():
    if X.shape[1:] != (C, L):
        print(f"[{name}] skipped: input shape {X.shape[1:]} != expected {(C, L)}")
        continue
    pred = predict(model, X, X_mean, X_std)
    print(f"=== {name} ({solvers[name]}, {len(y)} sample(s)) ===")
    for i in range(len(y)):
        print(f"  sample {i}:")
        print(f"    {'param':<10} {'true':>12} {'pred':>12} {'abs err':>12} {'rel err':>10}")
        for k, p in enumerate(PARAM_NAMES):
            ae = abs(pred[i, k] - y[i, k])
            re = ae / (abs(y[i, k]) + 1e-12)
            print(f"    {p:<10} {y[i, k]:>12.6f} {pred[i, k]:>12.6f} {ae:>12.6f} {re:>10.4%}")
    mse = float(np.mean((pred - y) ** 2))
    mare = float(np.mean(np.abs(pred - y) / (np.abs(y) + 1e-12)))
    print(f"  MSE={mse:.6e}  MARE={mare:.6e}\n")
