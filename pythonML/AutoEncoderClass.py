# use variational autoencoder (VAE) to generate new data and study latent space
# adapted to supervised latent-space (J1,J2,J3) training

from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
import numpy as np


@dataclass
class AutoEncoderOutput:
    """
    Output container for supervised autoencoder.
    Fields:
      z: latent tensor (J1,J2,J3)
      x_recon: reconstructed input
      loss: total loss (recon + latent)
      loss_recon: reconstruction loss
      latent_loss: supervised latent loss (MSE between z and true J)
    """
    z: torch.Tensor
    x_recon: torch.Tensor
    loss: Optional[torch.Tensor] = None
    loss_recon: Optional[torch.Tensor] = None
    latent_loss: Optional[torch.Tensor] = None

class FCautoencoder(nn.Module):
    """
    Fully Connected Autoencoder (supervised latent) class.
    Args:
        input_dim (int): dimension of input data
        hidden_dim (int): dimension of hidden layers
        latent_dim (int): dimension of latent space (should be 3 for J1,J2,J3)
    """
    def __init__(self,input_dim, hidden_dim=1024,latent_dim=3):
        super(FCautoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(), # Swish activation function
            nn.Linear(hidden_dim,hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.SiLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 8),
            nn.SiLU(),
            nn.Linear(hidden_dim // 8, latent_dim),
        )
        # Latent space is J1, J2, J3
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 8),
            nn.SiLU(),
            nn.Linear(hidden_dim // 8, hidden_dim // 4),
            nn.SiLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim),
        )
    
    def encode(self, x, eps: float = 1e-8):
        """
        Encodes the input into the latent space.
        Args:
            x (torch.Tensor): Input data
        Returns:
            torch.Tensor: Encoded data in the latent space
        """
        z = self.encoder(x)
        z = nn.Tanh()(z)  # constrain latent space to [-1,1]
        return z

    def decode(self, z):
        """
        Decodes the data from the latent space to the original input space.
        Args:
            z (torch.Tensor): Data in the latent space
        Returns:
            torch.Tensor: Reconstructed data in the original input space
        """
        return self.decoder(z)

    def forward(self, x, y: Optional[torch.Tensor] = None, compute_loss: bool = True):
        """
        Perform a forward pass.
        Args:
            x (torch.Tensor): Input data (will be flattened inside)
            y (Optional[torch.Tensor]): True latent J parameters (shape [batch, 3]) for supervised latent loss
            compute_loss (bool): Whether to compute the loss
        Returns:
            VAEOutput: output dataclass with losses
        """
        # ensure flattened
        # if x.dim() > 2:
        #     x = x.view(x.size(0), -1)

        # y is required — validate / reshape
        if y is None:
            raise ValueError("y (target J parameters) must be provided to forward().")
        # collapse extra dims and keep only first latent_dim entries
        if y.dim() == 1:
            y = y.view(-1, self.latent_dim)
        elif y.dim() > 2:
            y = y.view(y.size(0), -1)[:, :self.latent_dim]
        else:
            # y.dim() == 2 -> ensure correct width
            if y.size(1) < self.latent_dim:
                raise ValueError(f"y must have at least {self.latent_dim} values per sample, got {y.size(1)}")

        if y.size(1) != self.latent_dim:
            # if more columns present, truncate to latent_dim
            y = y[:, :self.latent_dim].contiguous()

        z = self.encode(x)
        recon_x = self.decode(z)

        if not compute_loss:
            return AutoEncoderOutput(
                z = z,
                x_recon = recon_x,
                loss = None,
                loss_recon = None,
                latent_loss = None,
            )

        # reconstruction loss (keep previous offset if desired)
        loss_recon = F.mse_loss(recon_x, x, reduction='mean') #reduction='sum')
        # supervised latent loss: compare z to provided true J parameters if available,
        target_J = y.to(z.dtype).to(z.device)
        loss_latent = F.mse_loss(z, target_J, reduction='sum')

        loss = loss_recon + loss_latent

        return AutoEncoderOutput(
            z = z,
            x_recon = recon_x,
            loss = loss,
            loss_recon = loss_recon,
            latent_loss = loss_latent,
        )


def train(model, dataloader, optimizer, prev_updates, writer=None,device='cpu'):
    """
    Args:
        model(nn.Module): The model to train
        dataloader (torch.utils.data.DataLoader): The data loader yields (data, target_J)
        optimizer (torch.optim.Optimizer): The optimizer to update the model parameters
    """
    model.train()
    for batch_idx, (data, target) in enumerate(tqdm(dataloader)):
        n_upd = prev_updates + batch_idx
        data = data.to(device)
        # flatten if needed
        if data.dim() > 2:
            data = data.view(data.size(0), -1)
        target = target.to(device)
        optimizer.zero_grad()
        output = model(data, target, compute_loss=True)
        loss = output.loss
        loss.backward()
        if n_upd % 100 == 0:
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            if writer is not None:
                global_step = n_upd
                writer.add_scalar('Loss/train', loss.item(), global_step)
                writer.add_scalar('Loss/train/Recon', output.loss_recon.item(), global_step)
                writer.add_scalar('Loss/train/Latent', output.latent_loss.item(), global_step)
                writer.add_scalar('GradNorm/train', total_norm, global_step)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    print(f'====> Train set loss: {loss.item():.4f}, Recon Loss: {output.loss_recon.item():.4f}, Latent Loss: {output.latent_loss.item():.4f}')
    return prev_updates + len(dataloader)

def test(model, dataloader, cur_step, writer=None,device='cpu'):
    """
    Args:
        model (nn.Module): The model to test
        dataloader (torch.utils.data.DataLoader): The data loader yields (data, target_J)
        cur_step (int): The current step
    """
    model.eval()
    test_loss = 0.0
    test_recon_loss = 0.0
    test_kl_loss = 0.0
    with torch.no_grad():
        for data, target in tqdm(dataloader, desc='Testing'):
            data = data.to(device)
            if data.dim() > 2:
                data = data.view(data.size(0), -1)
            target = target.to(device)
            output = model(data, target, compute_loss=True)
            test_loss += output.loss.item()
            test_recon_loss += output.loss_recon.item()
            test_kl_loss += output.latent_loss.item()
    
    test_loss /= len(dataloader)
    test_recon_loss /= len(dataloader)
    test_kl_loss /= len(dataloader)
    print(f'====> Test set loss: {test_loss:.4f}, Recon Loss: {test_recon_loss:.4f}, Latent Loss: {test_kl_loss:.4f}')
    if writer is not None:
        writer.add_scalar('Loss/test', test_loss, global_step=cur_step)
        writer.add_scalar('Loss/test/Recon', test_recon_loss, global_step=cur_step)
        writer.add_scalar('Loss/test/Latent', test_kl_loss, global_step=cur_step)