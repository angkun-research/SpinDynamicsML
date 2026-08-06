import os
import json
import numpy as np
import pandas as pd
import torch
from dash import Dash, dcc, html, Input, Output
import plotly.graph_objects as go

# import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from AutoEncoderClassNew import Conv1Dminimal
from utils import create_heatmap

# -----------------------
# Config
# -----------------------
CKPT_PATH = "./data/ConvChannel_anisotropy_LSWT_Js4_Nk250_sigma0.0.pth"

LATENT_DIM = 4
HIDDEN_DIM = 64
HAS_VARIANCE = True
IF_DECODE = True
Lk = 250

SLIDER_MARKS = {
    -1.0: "-1",
    -0.5: "-0.5",
     0.0: "0",
     0.5: "0.5",
     1.0: "1",
}

E_MIN = 0.0
E_MAX = 20.0
E_BINS = 300
SIGMA_DSSF = 0.08

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------
# Data helpers
# -----------------------
def to_numpy(x):
    if isinstance(x, np.ndarray):
        return x.astype(np.float32)
    if torch.is_tensor(x):
        return x.detach().cpu().numpy().astype(np.float32)
    return np.asarray(x, dtype=np.float32)

def build_model(input_length, in_channels=2):
    model = Conv1Dminimal(
        in_channels=in_channels,
        input_length=input_length,
        hidden_dim=HIDDEN_DIM,
        latent_dim=LATENT_DIM,
        has_variance=HAS_VARIANCE,
        ifdecode=IF_DECODE,
    )
    return model

def load_model_and_stats(ckpt_path, input_length):
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model = build_model(input_length=input_length, in_channels=2).to(DEVICE)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()
    X_mean = to_numpy(ckpt["X_mean"])   # shape (1, 2, L) or (1, 2, 1)
    X_std = to_numpy(ckpt["X_std"])
    return model, X_mean, X_std

@torch.no_grad()
def decode_from_latent(model, z_user, X_mean, X_std):
    # z_user shape: (latent_dim,)
    z = torch.tensor(z_user, dtype=torch.float32, device=DEVICE).unsqueeze(0)  # (1, latent_dim)
    x_recon_n = model.decode(z).detach().cpu().numpy()  # normalized recon, shape (1,2,L)
    x_recon_raw = x_recon_n * X_std + X_mean
    return x_recon_raw[0]  # (2, L)

# -----------------------
# Initialize
# -----------------------
model, X_mean, X_std = load_model_and_stats(CKPT_PATH, input_length=Lk)
k = np.linspace(0.0, 1.0, Lk)

app = Dash(__name__)
app.title = "Latent Decoder Explorer"

param_labels = [r"$J_1$", r"$J_1\Delta$", r"$J_{\pm\pm}$", r"$J_2$"]

slider_blocks = []
for i in range(LATENT_DIM):
    slider_blocks.append(
        html.Div([
            dcc.Markdown(
                param_labels[i],
                mathjax=True,
                style={"marginBottom": "4px"}
            ),
            dcc.Slider(
                id=f"z{i+1}",
                min=-1.0,
                max=1.0,
                step=0.01,
                marks=SLIDER_MARKS,
                value=float(np.clip(0.0, -1.0, 1.0)),
                tooltip={"placement": "bottom", "always_visible": True},
                included=False,
            )
        ], style={"marginBottom": "20px"})
    )

app.layout = html.Div([
    html.H2("Interactive Latent-Space Decoder (DSSF)"),
    html.Div(slider_blocks, style={"maxWidth": "900px"}),
    html.Div(id="latent-text", style={"marginBottom": "10px", "fontFamily": "monospace"}),
    dcc.Graph(id="dssf-graph"),
], style={"padding": "20px"})

@app.callback(
    Output("dssf-graph", "figure"),
    Output("latent-text", "children"),
    Input("z1", "value"),
    Input("z2", "value"),
    Input("z3", "value"),
    Input("z4", "value"),
)
def update_plots(z1, z2, z3, z4):
    z = np.array([z1, z2, z3, z4], dtype=np.float32)
    x_dec = decode_from_latent(model, z, X_mean, X_std)

    # channel 0 = dispersion, channel 1 = intensity
    disp_1d = x_dec[0]
    inten_1d = np.clip(x_dec[1], 0.0, None) # clip negative intensity to zero

    # create DSSF image from one branch (Nbands=1, Nq=L)
    disp_mat = disp_1d[None, :]
    inten_mat = inten_1d[None, :]
    Egrid = np.linspace(E_MIN, E_MAX, E_BINS)
    Egrid, img = create_heatmap(
        disp_mat, inten_mat, Egrid=Egrid, sigma=SIGMA_DSSF, normalize=True
    )

    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        x=k,
        y=Egrid,
        z=img.T,   # shape: (len(Egrid), L)
        colorscale="Turbo",
        colorbar={"title": "I(q,ω)"},
    ))

    fig.update_layout(
        title="Decoded Dynamic Structure Factor",
        xaxis_title="(H,H,0)",
        yaxis_title="Energy (meV)",
        template="plotly_white",
        height=520
    )

    txt = "Current latent vector: " + np.array2string(z, precision=4, separator=", ")
    return fig, txt

if __name__ == "__main__":
    app.run(debug=True)