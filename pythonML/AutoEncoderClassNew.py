from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm


class FCautoencoder(nn.Module):
    """
    Standalone FC autoencoder with optional variance head.
    Forward returns predictions only; no loss is computed here.
    """
    def __init__(
        self,
        input_dim,
        hidden_dim=128,
        latent_dim=3,
        ifdecode=False,
        has_variance=False,
        var_eps=1e-6,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.ifdecode = ifdecode
        self.has_variance = has_variance
        self.var_eps = var_eps

        latent_out_dim = latent_dim * 2 if has_variance else latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, latent_out_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim),
        )
    def encode(self, x, ):
        """
        Encodes the input into the latent space.
        Args:
            x (torch.Tensor): Input data
        Returns:
            torch.Tensor: Encoded data in the latent space
        """
        z_raw = self.encoder(x)
        return z_raw

    def decode(self, z):
        """
        Decodes the data from the latent space to the original input space.
        Args:
            z (torch.Tensor): Data in the latent space
        Returns:
            torch.Tensor: Reconstructed data in the original input space
        """
        return self.decoder(z)

    def _split_latent(self, z_raw):
        if not self.has_variance:
            mu = z_raw
            return mu, None

        mu_raw = z_raw[:, :self.latent_dim]
        var_raw = z_raw[:, self.latent_dim:]
        mu = mu_raw
        var = F.softplus(var_raw) + self.var_eps # ensure positivity, safer than exp
        return mu, var

    def forward(self, x):
        z_raw = self.encoder(x)
        mu, var = self._split_latent(z_raw)

        if self.ifdecode:
            recon_x = self.decoder(mu)
        else:
            recon_x = None

        # Return plain dict only (no loss inside model)
        return {
            "z_raw": z_raw,  # either latent_dim or 2*latent_dim
            "mu": mu,        # always latent_dim
            "var": var,      # None if has_variance=False
            "x_recon": recon_x,
        }


def _latent_loss(
    out_dict,
    target,
    latent_dim,
    has_variance=False,
    latent_loss_type="mse",    # "mse" or "relative" or "gaussian_nll"
    var_eps=1e-6,
):
    target = target.to(dtype=out_dict["mu"].dtype, device=out_dict["mu"].device)
    mu = out_dict["mu"]

    mse = F.mse_loss(mu, target, reduction="sum")
    if has_variance:
        if out_dict["var"] is None:
            raise ValueError("has_variance=True but model did not return var.")
        var = out_dict["var"] + var_eps
        loss_latent = F.gaussian_nll_loss(mu, target, var, reduction="sum")
        rel = ((mu - target) ** 2) / (target.abs() ** 2 + 1e-12)
        return loss_latent, mse, rel.sum(), var.mean()

    rel = ((mu - target) ** 2) / (target.abs() ** 2 + 1e-12)
    rel_sum = rel.sum()

    if latent_loss_type == "relative":
        return rel_sum, mse, rel_sum, None

    # default
    return mse, rel_sum, None, None


def _recon_loss(out_dict, x, ifdecode=True, mag_recon_loss=1.0):
    if (not ifdecode) or (out_dict["x_recon"] is None):
        return torch.tensor(0.0, device=x.device, dtype=x.dtype)
    return F.mse_loss(out_dict["x_recon"], x, reduction="mean") * mag_recon_loss


def train(
    model,
    dataloader,
    optimizer,
    prev_updates,
    device="cpu",
    latent_dim=3,
    has_variance=False,
    latent_loss_type="mse",    # "mse", "relative", "gaussian_nll"
    ifdecode=True,
    mag_recon_loss=1.0,
    var_eps=1e-6,
):
    model.train()
    last_loss = None
    last_recon = None
    last_latent = None
    last_rel = None
    last_var_mean = None
    last_mse = None

    for batch_idx, (data, target) in enumerate(tqdm(dataloader)):
        n_upd = prev_updates + batch_idx
        data = data.to(device)
        target = target.to(device)

        optimizer.zero_grad()

        out_dict = model(data)
        loss_recon = _recon_loss(
            out_dict,
            data,
            ifdecode=ifdecode,
            mag_recon_loss=mag_recon_loss,
        )
        loss_latent, loss_mse, rel_loss, var_mean = _latent_loss(
            out_dict,
            target,
            latent_dim=latent_dim,
            has_variance=has_variance,
            latent_loss_type=latent_loss_type,
            var_eps=var_eps,
        )
        loss = loss_recon + loss_latent
        loss.backward()

        #torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # clip gradients to avoid exploding gradients
        optimizer.step()

        last_loss = loss
        last_recon = loss_recon
        last_latent = loss_latent
        last_mse = loss_mse
        last_rel = rel_loss
        last_var_mean = var_mean

    print(
        f"====> Train set loss: {last_loss.item():.4f}, "
        f"Recon Loss: {last_recon.item():.4f}, "
        f"Latent Loss: {last_latent.item():.4f}"
        f"Latent MSE: {last_mse.item():.4f}"
    )
    print(f"====> Relative Latent Loss: {last_rel.item():.6f}")
    if last_var_mean is not None:
        print(f"====> Mean Predicted Variance: {last_var_mean.item():.6e}")

    return prev_updates + len(dataloader)


def test(
    model,
    dataloader,
    cur_step,
    device="cpu",
    latent_dim=3,
    has_variance=False,
    latent_loss_type="mse",    # "mse", "relative", "gaussian_nll"
    ifdecode=True,
    mag_recon_loss=1.0,
    var_eps=1e-6,
):
    model.eval()

    test_loss = 0.0
    test_recon = 0.0
    test_latent = 0.0
    test_rel = 0.0
    test_var_mean = 0.0
    test_mse = 0.0
    n_batches = 0

    with torch.no_grad():
        for data, target in tqdm(dataloader, desc="Testing"):
            data = data.to(device)
            target = target.to(device)

            out_dict = model(data)
            loss_recon = _recon_loss(
                out_dict,
                data,
                ifdecode=ifdecode,
                mag_recon_loss=mag_recon_loss,
            )
            loss_latent, loss_mse, rel_loss, var_mean = _latent_loss(
                out_dict,
                target,
                latent_dim=latent_dim,
                has_variance=has_variance,
                latent_loss_type=latent_loss_type,
                var_eps=var_eps,
            )

            loss = loss_recon + loss_latent

            test_loss += loss.item()
            test_recon += loss_recon.item()
            test_latent += loss_latent.item()
            test_mse += loss_mse.item()
            test_rel += rel_loss.item()
            if var_mean is not None:
                test_var_mean += var_mean.item()
            n_batches += 1

    test_loss /= n_batches
    test_recon /= n_batches
    test_latent /= n_batches
    test_mse /= n_batches
    test_rel /= n_batches
    if has_variance:
        test_var_mean /= n_batches

    print(
        f"====> Test set loss: {test_loss:.4f}, "
        f"Recon Loss: {test_recon:.4f}, "
        f"Latent Loss: {test_latent:.4f}, "
        f"Latent MSE: {test_mse:.4f}"
    )
    print(f"====> Relative Latent Loss: {test_rel:.6f}")
    if has_variance:
        print(f"====> Mean Predicted Variance: {test_var_mean:.6e}")
