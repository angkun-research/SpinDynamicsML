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


class ConvEncoder(nn.Module):
    """
    Convolutional Autoencoder (supervised latent) class.
    Args:
        input_dim (int): dimension of input data
        hidden_dim (int): dimension of hidden layers
        latent_dim (int): dimension of latent space (should be 3 for J1,J2,J3)
    """
    def __init__(self,input_dim, hidden_dim=1024,latent_dim=3, kernel_size=3):
        super(ConvEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.kernel_size = kernel_size
        dim1 = int(np.sqrt(hidden_dim))
        while hidden_dim % dim1 != 0:
            dim1 -= 1
        dim1p = hidden_dim // dim1
        outpadding = (kernel_size - 1) % 2

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(), # Swish activation function
            nn.Unflatten(1, (1, dim1, dim1p)), # Reshape to (batch_size, channels=1, height=dim1, width=dim1p)
            nn.Conv2d(1, 4, kernel_size=kernel_size, padding='same'),  # Conv2d layer
            nn.SiLU(),
            nn.Flatten(),  # Flatten back to 1D
            nn.Linear(dim1 * dim1p * 4, hidden_dim // 2),  # Adjust dimensions for Conv2d output
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, latent_dim),
        )
        # Latent space is J1, J2, J3
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, dim1 * dim1p * 4),  # Adjust dimensions for Conv2d output
            nn.SiLU(),
            nn.Unflatten(1, (4, dim1, dim1p)),  # Reshape to (batch_size, channels=4, height=dim1, width=dim1p)
            nn.ConvTranspose2d(4, 1, kernel_size=kernel_size, padding=kernel_size//2, output_padding=outpadding),  # Transpose Conv2d layer
            nn.Flatten(),  # Flatten back to 1D
            nn.Linear(dim1 * dim1p, input_dim),  # Final output layer
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


class TransformerEncoder(nn.Module):
    """
    Transformer-based Autoencoder (supervised latent) class.
    Args:
        input_dim (int): dimension of input data
        hidden_dim (int): dimension of hidden layers
        latent_dim (int): dimension of latent space (should be 3 for J1,J2,J3)
    """
    def __init__(self,input_dim, hidden_dim=1024,latent_dim=3, n_layers=2, n_heads=4):
        super(TransformerEncoder, self).__init__()
        self.latent_dim = latent_dim
        d_feature = int(np.sqrt(hidden_dim))
        while hidden_dim % d_feature != 0:
            d_feature -= 1
        d_sequence = hidden_dim // d_feature
        self.d_feature = d_feature
        self.d_sequence = d_sequence
        # Transformer encoder - decreasing dimensions like FC version
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_feature, nhead=n_heads, dim_feedforward=hidden_dim,
            dropout=0.1, activation='gelu', batch_first=True
        )

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(), # Swish activation function
            nn.Unflatten(1, (d_sequence, d_feature)), # Reshape to (batch_size, d_sequence, d_feature)
            nn.TransformerEncoder(encoder_layer, num_layers=n_layers),
            nn.SiLU(),
            nn.Flatten(),  # Flatten back to 1D
            nn.Linear(d_sequence * d_feature, latent_dim),  
        )
        # Latent space is J1, J2, J3
        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_feature, nhead=n_heads, dim_feedforward=hidden_dim,
            dropout=0.1, activation='gelu', batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        # Fully connected layers for decoding
        self.fc_decoder = nn.Sequential(
            nn.Linear(latent_dim, d_sequence * d_feature),
            nn.SiLU(),
        )
        self.output_layer = nn.Sequential(
            nn.Flatten(),  # Flatten back to 1D
            nn.Linear(d_sequence * d_feature, input_dim),  # Final output layer
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
        #return self.decoder(z)  # (batch, input_dim)
        batch_size = z.shape[0]
        # FC decoder
        decoded = self.fc_decoder(z) # (batch_size, d_model * matrix_rows)
        decoded = decoded.view(batch_size,self.d_sequence, self.d_feature) # (batch_size, d_model, matrix_rows)
        target_seq = torch.zeros_like(decoded, device=z.device)
        # Create memory for transformer decoder
        memory = decoded
        #target_seq = torch.zeros(batch_size, self.matrix_rows, self.d_model, device=z.device)
        decoded_features = self.transformer_decoder(target_seq, memory) # (batch, matrix_rows, d_model)
        x = self.output_layer(decoded_features)
        return x

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







