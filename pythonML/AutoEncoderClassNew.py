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
        var_eps=1e-10,
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
            nn.Linear(hidden_dim, latent_out_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
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
    has_variance=False,
    latent_loss_type="mse",    # "mse" or "relative" or "gaussian_nll"
    var_eps=1e-10,
):
    target = target.to(dtype=out_dict["mu"].dtype, device=out_dict["mu"].device)
    mu = out_dict["mu"]

    mse = F.mse_loss(mu, target, reduction="sum")
    if has_variance:
        if out_dict["var"] is None:
            raise ValueError("has_variance=True but model did not return var.")
        var = out_dict["var"] + var_eps
        loss_latent = F.gaussian_nll_loss(mu, target, var, reduction="sum")
        #rel = ((mu - target) ** 2) / (target.abs() ** 2 + 1e-12)
        rel = (mu - target).abs() / (target.abs() + 1e-12)
        return loss_latent, mse, rel.sum(), var.mean()

    #rel = ((mu - target) ** 2) / (target.abs() ** 2 + 1e-12)
    rel = (mu - target).abs() / (target.abs() + 1e-12)
    rel_sum = rel.sum()

    if latent_loss_type == "relative":
        return rel_sum, mse, rel_sum, None

    # default
    return mse, mse, rel_sum, None


def _recon_loss(out_dict, x, ifdecode=True, mag_recon_loss=1.0):
    if (not ifdecode) or (out_dict["x_recon"] is None):
        return torch.tensor(0.0, device=x.device, dtype=x.dtype)
    return F.mse_loss(out_dict["x_recon"], x, reduction="mean") * mag_recon_loss


def train(
    model,
    dataloader,
    optimizer,
    latent_dim,
    batch_size,
    device="cpu",
    has_variance=False,
    latent_loss_type="mse",    # "mse", "relative", "gaussian_nll"
    ifdecode=True,
    mag_recon_loss=1.0,
    var_eps=1e-10,
):
    model.train()
    last_loss = None
    last_recon = None
    last_latent = None
    last_rel = None
    last_var_mean = None
    last_mse = None
    total_rel_error = 0.0
    total_mse = 0.0

    for batch_idx, (data, target) in enumerate(tqdm(dataloader)):
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
        total_rel_error += rel_loss.item()
        total_mse += loss_mse.item()

    mean_rel_error = total_rel_error / len(dataloader) / latent_dim / batch_size
    #mean_rel_error = mean_rel_error ** 0.5
    total_mse = total_mse / len(dataloader) / latent_dim / batch_size
    print(
        f"====> Train set loss: {last_loss.item():.4f}, "
        f"Recon Loss: {last_recon.item():.4f}, "
        f"Latent Loss: {last_latent.item():.4f}"
        f"Latent TSE: {last_mse.item():.4f}"
    )
    print(f"====> Relative Latent Loss: {last_rel.item():.6f}")
    print(f"====> Total Latent MSE: {total_mse:.6f}")
    print(f"====> Mean Relative Error: {mean_rel_error:.6f}")
    if last_var_mean is not None:
        print(f"====> Mean Predicted Variance: {last_var_mean.item():.6e}")

    all_out_dict = {
        "train_loss": last_loss.item(),
        "train_recon": last_recon.item(),
        "train_latent": last_latent.item(),
        "train_mse": total_mse, #last_mse.item(),
        "train_rel": last_rel.item(),
        "train_var_mean": last_var_mean.item() if last_var_mean is not None else None,
        "train_mare": mean_rel_error,
    }

    return all_out_dict


def test(
    model,
    dataloader,
    latent_dim,
    batch_size,
    device="cpu",
    has_variance=False,
    latent_loss_type="mse",    # "mse", "relative", "gaussian_nll"
    ifdecode=True,
    mag_recon_loss=1.0,
    var_eps=1e-10,
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
    #test_rmse = (test_rel / latent_dim / batch_size) ** 0.5  # RMSE of relative error
    test_mare = test_rel / latent_dim / batch_size
    test_mse = test_mse / latent_dim / batch_size
    print(
        f"====> Test set loss: {test_loss:.4f}, "
        f"Recon Loss: {test_recon:.4f}, "
        f"Latent Loss: {test_latent:.4f}, "
        f"Latent TSE: {test_mse:.4f}"
    )
    print(f"====> Relative Latent Loss: {test_rel:.6f}")
    print(f"====> Total Latent MSE: {test_mse:.6f}")
    print(f"====> Mean Relative Error: {test_mare:.6f}")
    if has_variance:
        print(f"====> Mean Predicted Variance: {test_var_mean:.6e}")

    all_out_dict = {
        "test_loss": test_loss,
        "test_recon": test_recon,
        "test_latent": test_latent,
        "test_mse": test_mse,
        "test_rel": test_rel,
        "test_var_mean": test_var_mean if has_variance else None,
        "test_mare": test_mare,
    }
    return all_out_dict


class Conv1Dminimal(nn.Module):
    """
    Simplest Conv1D with optional variance head.
    Forward returns predictions only; no loss is computed here.
    """
    def __init__(
        self,
        in_channels,
        input_length,
        kernel_size=3,
        hidden_dim=64,
        latent_dim=3,
        ifdecode=False,
        has_variance=False,
        var_eps=1e-10,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.ifdecode = ifdecode
        self.has_variance = has_variance
        self.var_eps = var_eps

        latent_out_dim = latent_dim * 2 if has_variance else latent_dim

        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=kernel_size,padding=kernel_size//2), 
            nn.SiLU(),
            nn.Flatten(),  # flatten before the final linear layer
            nn.Linear(hidden_dim * input_length, hidden_dim), # adjust input size for linear
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_out_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * input_length), # adjust output size for linear
            nn.Unflatten(1, (hidden_dim, input_length)),  # unflatten to (batch_size, hidden_dim, input_length)
            nn.ConvTranspose1d(hidden_dim, in_channels, kernel_size=kernel_size,padding=kernel_size//2),
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