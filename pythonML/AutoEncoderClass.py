# use variational autoencoder (VAE) to generate new data and study latent space
# adapted to supervised latent-space (J1,J2,J3) training

from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
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
    relative_loss: Optional[torch.Tensor] = None

class FCautoencoder(nn.Module):
    """
    Fully Connected Autoencoder (supervised latent) class.
    Args:
        input_dim (int): dimension of input data
        hidden_dim (int): dimension of hidden layers
        latent_dim (int): dimension of latent space (should be 3 for J1,J2,J3)
    """
    def __init__(self,input_dim, hidden_dim=1024,latent_dim=3,mag_recon_loss=1.0, ifdecode=False, relative_loss=False):
        super(FCautoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.mag_recon_loss = mag_recon_loss
        self.ifdecode = ifdecode
        self.relative_loss = relative_loss
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
        #z = torch.tanh(z ** 3)
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

        z = self.encode(x)
        recon_x = self.decode(z)
        if self.ifdecode:
            recon_x = self.decode(z)
        else:
            recon_x = torch.zeros_like(x)  # set to zero tensor of same shape as x

        if not compute_loss:
            return AutoEncoderOutput(z=z, x_recon=recon_x, loss=None, loss_recon=None, latent_loss=None)

        if self.ifdecode:
            loss_recon = F.mse_loss(recon_x, x, reduction='mean') * self.mag_recon_loss
        else:
            loss_recon = torch.tensor(0.0, device=x.device)


        # reconstruction loss (keep previous offset if desired)
        #loss_recon = F.mse_loss(recon_x, x, reduction='mean')*self.mag_recon_loss #reduction='sum')
        #loss_recon = F.mse_loss(recon_x, x, reduction='sum') 
        # supervised latent loss: compare z to provided true J parameters if available,
        target_J = y.to(z.dtype).to(z.device)
        tsl = F.mse_loss(z, target_J, reduction='sum')
        rel_err = ((z - target_J)**2) / ((target_J.abs())**2)
        rel_err = rel_err.sum()
        if self.relative_loss:
            loss_latent = rel_err
        else:
            loss_latent = tsl

        loss = loss_recon + loss_latent

        return AutoEncoderOutput(
            z = z,
            x_recon = recon_x,
            loss = loss,
            loss_recon = loss_recon,
            latent_loss = loss_latent,
            relative_loss = rel_err,
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
        # if data.dim() > 2:
        #     data = data.view(data.size(0), -1)
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
    # if has loss.relative_loss, print it
    if hasattr(output, 'relative_loss'):
        print(f'====> Relative Latent Loss: {output.relative_loss.item():.6f}')
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
            # if data.dim() > 2:
            #     data = data.view(data.size(0), -1)
            target = target.to(device)
            output = model(data, target, compute_loss=True)
            test_loss += output.loss.item()
            test_recon_loss += output.loss_recon.item()
            test_kl_loss += output.latent_loss.item()
    
    test_loss /= len(dataloader)
    test_recon_loss /= len(dataloader)
    test_kl_loss /= len(dataloader)
    print(f'====> Test set loss: {test_loss:.4f}, Recon Loss: {test_recon_loss:.4f}, Latent Loss: {test_kl_loss:.4f}')
    # if has loss.relative_loss, print it
    if hasattr(output, 'relative_loss'):
        print(f'====> Relative Latent Loss: {output.relative_loss.item():.6f}')
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
    def __init__(self,input_dim, hidden_dim=1024,latent_dim=3, kernel_size=3,mag_recon_loss=1.0):
        super(ConvEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.mag_recon_loss = mag_recon_loss
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
        loss_recon = F.mse_loss(recon_x, x, reduction='mean') * self.mag_recon_loss #reduction='sum')
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
    def __init__(self,input_dim, hidden_dim=1024,latent_dim=3, n_layers=2, n_heads=4,mag_recon_loss=1.0):
        super(TransformerEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.mag_recon_loss = mag_recon_loss
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
        loss_recon = F.mse_loss(recon_x, x, reduction='mean') * self.mag_recon_loss #reduction='sum')
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


# class Conv1DEncoder(nn.Module):
#     """
#     Convolutional Autoencoder (supervised latent) class.
#     Args:
#         input_dim (int): dimension of input data
#         hidden_dim (int): dimension of hidden layers
#         latent_dim (int): dimension of latent space (should be 3 for J1,J2,J3)
#     """
#     def __init__(self,input_dim,in_channels=1, hidden_dim=128,latent_dim=3, 
#                  kernel_size=3,mag_recon_loss=1.0,relative_loss=False, ifdecode=False):
#         super(Conv1DEncoder, self).__init__()
#         self.latent_dim = latent_dim
#         self.input_dim = input_dim
#         self.mag_recon_loss = mag_recon_loss
#         self.relative_loss = relative_loss
#         self.kernel_size = kernel_size
#         self.ifdecode = ifdecode
#         pad = (kernel_size-1) // 2
#         k1, k2 = self.kernel_size, 3
#         p1, p2 = (k1 - 1) // 2, (k2 - 1) // 2

#         self.encoder = nn.Sequential(
#             #nn.Conv1d(in_channels, hidden_dim, kernel_size=k1, padding=p1),  # first conv: k=5
#             nn.Conv1d(in_channels, hidden_dim, kernel_size=k1,padding=p1,padding_mode='circular'),  # periodic padding to avoid edge effects
#             nn.SiLU(),# Swish activation function
#             nn.Conv1d(hidden_dim, hidden_dim, kernel_size=k2, padding=p2,padding_mode='circular'),    # second conv: k=3
#             nn.SiLU(),
#             nn.AdaptiveAvgPool1d(1),  # Ensure fixed output, shape (B, hidden_dim, 1)
#             nn.Flatten(),  # Flatten back to 1D
#             nn.Linear(hidden_dim, hidden_dim//2),  # Adjust dimensions for Conv2d output
#             nn.SiLU(),
#             nn.Linear(hidden_dim//2, latent_dim),
#             nn.Tanh(),  # constrain latent space to [-1,1]
#         )
#         # Latent space is J1, J2, J3
#         self.decoder = nn.Sequential(
#             nn.Linear(latent_dim, hidden_dim),
#             nn.SiLU(),
#             nn.Linear(hidden_dim, hidden_dim*input_dim),  # Adjust dimensions for Conv2d output
#             nn.SiLU(),
#             nn.Unflatten(1, (hidden_dim, input_dim)),  # Reshape to (batch_size, channels=hidden_dim, width=1)
#             #nn.Linear(hidden_dim, input_dim),  # expand to (B, hidden_dim, input_dim)
#             nn.ConvTranspose1d(hidden_dim, in_channels, kernel_size=kernel_size, padding=pad,output_padding=0),#pad-1),  # Transpose Conv2d layer
#             nn.Flatten(),  # Flatten back to 1D
#         )
    
#     def encode(self, x, eps: float = 1e-8):
#         """
#         Encodes the input into the latent space.
#         Args:
#             x (torch.Tensor): Input data
#         Returns:
#             torch.Tensor: Encoded data in the latent space
#         """
#         #print("Encoded x shape:", x.shape)
#         z = self.encoder(x)
#         return z

#     def decode(self, z):
#         """
#         Decodes the data from the latent space to the original input space.
#         Args:
#             z (torch.Tensor): Data in the latent space
#         Returns:
#             torch.Tensor: Reconstructed data in the original input space
#         """
#         recon_x = self.decoder(z)
#         # reshape to (B, input_dim)
#         #recon_x = recon_x.view(recon_x.size(0), -1)
#         return recon_x
    
#     def forward(self, x, y: Optional[torch.Tensor] = None, compute_loss: bool = True):
#         # convert from (B, input_dim) to (B, in_channels, input_dim)
#         x_shape = x.view(x.size(0), 1, x.size(1))
#         # encode / decode
#         z = self.encode(x_shape)
#         recon_x = self.decode(z)

#         if self.ifdecode:
#             recon_x = self.decode(z)
#         else:
#             recon_x = torch.zeros_like(x)  # set to zero tensor of same shape as x
#         if self.ifdecode:
#             loss_recon = F.mse_loss(recon_x, x, reduction='mean') * self.mag_recon_loss
#         else:
#             loss_recon = torch.tensor(0.0, device=x.device)

#         if not compute_loss:
#             return AutoEncoderOutput(z=z, x_recon=recon_x, loss=None, loss_recon=None, latent_loss=None)

#         #loss_recon = F.mse_loss(recon_x, x, reduction='mean') * self.mag_recon_loss
#         target_J = y.to(z.dtype).to(z.device)
#         loss_latent = F.mse_loss(z, target_J, reduction='sum')
#         loss = loss_recon + loss_latent

#         tsl = F.mse_loss(z, target_J, reduction='sum')
#         rel_err = ((z - target_J)**2) / ((target_J.abs())**2)
#         rel_err = rel_err.sum()
#         if self.relative_loss:
#             loss_latent = rel_err
#         else:
#             loss_latent = tsl

#         loss = loss_recon + loss_latent

#         return AutoEncoderOutput(
#             z = z,
#             x_recon = recon_x,
#             loss = loss,
#             loss_recon = loss_recon,
#             latent_loss = loss_latent,
#             relative_loss = rel_err,
#         )

class Conv1DEncoder(nn.Module):
    """
    Convolutional Autoencoder (supervised latent) class.
    """
    def __init__(self,in_channels=1, hidden_dim=128,latent_dim=3, 
                 kernel_size=3,mag_recon_loss=1.0,relative_loss=False, ifdecode=False):
        super(Conv1DEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.mag_recon_loss = mag_recon_loss
        self.relative_loss = relative_loss
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.ifdecode = ifdecode
        self.input_length = None # Will be built on first forward pass
        self.decoder = None  # Will be built on first forward pass
    
        pad = (kernel_size-1) // 2
        k1, k2 = self.kernel_size, 3
        p1, p2 = (k1 - 1) // 2, (k2 - 1) // 2
        # Store for lazy decoder initialization
        self.pad = pad
        self.k1, self.k2 = k1, k2
        self.p1, self.p2 = p1, p2
        
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=k1,padding=p1,padding_mode='circular'),  # periodic padding to avoid edge effects
            nn.SiLU(),# Swish activation function
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=k2, padding=p2,padding_mode='circular'),    # second conv: k=3
            nn.SiLU(),
            nn.AdaptiveAvgPool1d(1),  # Ensure fixed output, shape (B, hidden_dim, 1)
            nn.Flatten(),  # Flatten back to 1D
            nn.Linear(hidden_dim, hidden_dim//2),  # Adjust dimensions for Conv2d output
            nn.SiLU(),
            nn.Linear(hidden_dim//2, latent_dim),
            nn.Tanh(),  # constrain latent space to [-1,1]
        )

    def _build_decoder(self, input_length):
        """Lazily build decoder with known input_length."""
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim * input_length),
            nn.SiLU(),
            nn.Unflatten(1, (self.hidden_dim, input_length)),
            nn.ConvTranspose1d(self.hidden_dim, self.hidden_dim, kernel_size=self.k2, padding=self.p2),
            # circular padding in ConvTranspose1d is not directly supported, maybe add manual circular padding in forward pass
            nn.SiLU(),
            nn.ConvTranspose1d(self.hidden_dim, self.in_channels, kernel_size=self.k1, padding=self.p1),
        )
    
    def encode(self, x, eps: float = 1e-8):
        """
        Encodes the input into the latent space.
        Args:
            x (torch.Tensor): Input data
        Returns:
            torch.Tensor: Encoded data in the latent space
        """
        #print("Encoded x shape:", x.shape)
        z = self.encoder(x)
        return z

    def decode(self, z):
        """
        Decodes the data from the latent space to the original input space.
        Args:
            z (torch.Tensor): Data in the latent space
        Returns:
            torch.Tensor: Reconstructed data in the original input space
        """
        recon_x = self.decoder(z)
        return recon_x
    
    def forward(self, x, y: Optional[torch.Tensor] = None, compute_loss: bool = True):
        # convert from (B, input_dim) to (B, in_channels, input_dim)
        x_shape = x if x.dim() == 3 else x.view(x.size(0), 2, -1)
        
        # Initialize decoder on first forward pass
        if self.input_length is None:
            self.input_length = x_shape.size(2)
            self._build_decoder(self.input_length)

        # encode / decode
        z = self.encode(x_shape)
        recon_x = self.decode(z)

        if self.ifdecode:
            recon_x = self.decode(z)
        else:
            recon_x = torch.zeros_like(x)  # set to zero tensor of same shape as x
        if self.ifdecode:
            loss_recon = F.mse_loss(recon_x, x, reduction='mean') * self.mag_recon_loss
        else:
            loss_recon = torch.tensor(0.0, device=x.device)

        if not compute_loss:
            return AutoEncoderOutput(z=z, x_recon=recon_x, loss=None, loss_recon=None, latent_loss=None)

        #loss_recon = F.mse_loss(recon_x, x, reduction='mean') * self.mag_recon_loss
        target_J = y.to(z.dtype).to(z.device)
        loss_latent = F.mse_loss(z, target_J, reduction='sum')
        loss = loss_recon + loss_latent

        tsl = F.mse_loss(z, target_J, reduction='sum')
        rel_err = ((z - target_J)**2) / ((target_J.abs())**2)
        rel_err = rel_err.sum()
        if self.relative_loss:
            loss_latent = rel_err
        else:
            loss_latent = tsl

        loss = loss_recon + loss_latent

        return AutoEncoderOutput(
            z = z,
            x_recon = recon_x,
            loss = loss,
            loss_recon = loss_recon,
            latent_loss = loss_latent,
            relative_loss = rel_err,
        )



class ConvDownSampling(nn.Module):
    def __init__(self, D_K, D_E, in_channels=1, base_channels=32, latent_dim=3,kernel_size=3, mag_recon_loss=1.0):
        super(ConvDownSampling, self).__init__()
        self.input_h = D_K
        self.input_w = D_E
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.mag_recon_loss = mag_recon_loss
        outpadding = (kernel_size - 1) % 2

        # Encoder conv stack: three downsampling conv blocks (stride=2)
        self.encoder = nn.Sequential(
            # Input: (B, C, 100,300) -> Output: (B, 32, 50, 150)
            nn.Conv2d(in_channels, base_channels, kernel_size=kernel_size, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(0.1), # Leaky is often better for gradients in physics nets
            
            # Input: (B, 32, 50, 150) -> Output: (B, 64, 25, 75)
            nn.Conv2d(base_channels, base_channels*2, kernel_size=kernel_size, stride=2, padding=1),
            nn.BatchNorm2d(base_channels*2),
            nn.LeakyReLU(0.1),
            
            # Input: (B, 64, 25, 75) -> Output: (B, 128, 13, 38)
            nn.Conv2d(base_channels*2, base_channels*4, kernel_size=kernel_size, stride=2, padding=1),
            nn.BatchNorm2d(base_channels*4),
            nn.LeakyReLU(0.1),
            
            # Input: (B, 128, 13, 38) -> Output: (B, 256, 7, 19)
            nn.Conv2d(base_channels*4, base_channels*8, kernel_size=kernel_size, stride=2, padding=1),
            nn.BatchNorm2d(base_channels*8),
            nn.LeakyReLU(0.1),
        )
        # Calculate feature map size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, self.in_channels, self.input_h, self.input_w)
            dummy_out = self.encoder(dummy)
            self.feature_c = dummy_out.shape[1]
            self.feature_h = dummy_out.shape[2]
            self.feature_w = dummy_out.shape[3]
            self.flat_dim = self.feature_c * self.feature_h * self.feature_w

        # Bottleneck
        self.to_params = nn.Linear(self.flat_dim, latent_dim)
        self.from_params = nn.Linear(latent_dim, self.flat_dim)
        
        # --- DECODER ---
        self.decoder = nn.Sequential(
            # Input: (B, 256, 7, 19) -> Output: (B, 128, 13, 38)
            nn.ConvTranspose2d(base_channels*8, base_channels*4, kernel_size=kernel_size, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(base_channels*4),
            nn.LeakyReLU(0.1),
            # Input: (B, 128, 13, 38) -> Output: (B, 64, 25, 75)
            nn.ConvTranspose2d(base_channels*4, base_channels*2, kernel_size=kernel_size, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(base_channels*2),
            nn.LeakyReLU(0.1),
            # Input: (B, 64, 25, 75) -> Output: (B, 32, 50, 150)
            nn.ConvTranspose2d(base_channels*2, base_channels, kernel_size=kernel_size, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(0.1),
            # Input: (B, 32, 50, 150) -> Output: (B, 1, 100, 300)
            nn.ConvTranspose2d(base_channels, in_channels, kernel_size=kernel_size, stride=2, padding=1, output_padding=1),
            nn.Tanh() 
        )


    def encode(self, x):
        # 1. Encode
        features = self.encoder(x)
        features_flat = torch.flatten(features, start_dim=1)
        
        # 2. Bottleneck (Latent Space = Hamiltonian Parameters)
        z = self.to_params(features_flat)
        z = nn.Tanh()(z)  # constrain latent space to [-1,1]
        return z

    def decode(self, z):
        # 3. Decode
        reconstruction_input = self.from_params(z)
        
        # Reshape back to (B, Channels, H, W) for the ConvTranspose layers
        reconstruction_input = reconstruction_input.view(
            -1, self.feature_c, self.feature_h, self.feature_w
        )
        
        reconstruction = self.decoder(reconstruction_input)
        # 4. FIX: Interpolate if dimensions don't match (e.g. 112 vs 100, 292 vs 300)
        if reconstruction.shape[-2:] != (self.input_h, self.input_w):
             reconstruction = F.interpolate(
                 reconstruction, 
                 size=(self.input_h, self.input_w), 
                 mode='bilinear', 
                 align_corners=False
             )
        return reconstruction

    def forward(self, x, y: Optional[torch.Tensor] = None, compute_loss: bool = True):
        if x.dim() == 2:
            x = x.view(-1, self.in_channels, self.input_h, self.input_w)
        # encode / decode
        z = self.encode(x)
        recon_x = self.decode(z)

        if not compute_loss:
            return AutoEncoderOutput(z=z, x_recon=recon_x, loss=None, loss_recon=None, latent_loss=None)

        loss_recon = F.mse_loss(recon_x, x, reduction='mean') * self.mag_recon_loss
        target_J = y.to(z.dtype).to(z.device)
        loss_latent = F.mse_loss(z, target_J, reduction='sum')
        loss = loss_recon + loss_latent

        return AutoEncoderOutput(z=z, x_recon=recon_x, loss=loss, loss_recon=loss_recon, latent_loss=loss_latent)




class SimpleConvAutoencoder(nn.Module):
    def __init__(self, D_K, D_E, in_channels=1, base_channels=32, latent_dim=3,kernel_size=3, mag_recon_loss=1.0):
        super(SimpleConvAutoencoder, self).__init__()
        self.input_h = D_K
        self.input_w = D_E
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.mag_recon_loss = mag_recon_loss
        self.base_channels = base_channels

        # We use padding=1 with kernel=3 to keep size EXACTLY the same.
        self.encoder = nn.Sequential(
            # Layer 1: (B, 1, 100, 300) -> (B, 16, 100, 300)
            nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(0.1),
            
            # Layer 2: (B, 16, 100, 300) -> (B, 32, 100, 300)
            nn.Conv2d(base_channels, base_channels*2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(base_channels*2),
            nn.LeakyReLU(0.1)
        )
        # Calculate Flattened Size
        # Since we didn't downsample, it's just H * W * Channels
        self.flat_dim = (base_channels * 2) * self.input_h * self.input_w
        # Bottleneck (Huge Linear Layer)
        self.to_params = nn.Linear(self.flat_dim, latent_dim)
        self.from_params = nn.Linear(latent_dim, self.flat_dim)
        
        # --- DECODER (Symmetric) ---
        # Note: We use Conv2d here too! No need for Transpose because size is constant.
        self.decoder = nn.Sequential(
            # Layer 1: (B, 32, 100, 300) -> (B, 16, 100, 300)
            nn.Conv2d(base_channels*2, base_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(0.1),
            
            # Layer 2: (B, 16, 100, 300) -> (B, 1, 100, 300)
            nn.Conv2d(base_channels, in_channels, kernel_size=3, stride=1, padding=1),
            # Final activation to match data range (e.g., Tanh for -1 to 1)
            nn.Tanh() 
        )


    def encode(self, x):
        # 1. Encode
        features = self.encoder(x)
        features_flat = torch.flatten(features, start_dim=1)
        
        # 2. Bottleneck (Latent Space = Hamiltonian Parameters)
        z = self.to_params(features_flat)
        z = nn.Tanh()(z)  # constrain latent space to [-1,1]
        return z

    def decode(self, z):
        # 3. Decode
        reconstruction_input = self.from_params(z)
        
        # Reshape back to (B, Channels, H, W) for the ConvTranspose layers
        reconstruction_input = reconstruction_input.view(
            -1, self.base_channels * 2, self.input_h, self.input_w
        )
        
        reconstruction = self.decoder(reconstruction_input)
        return reconstruction

    def forward(self, x, y: Optional[torch.Tensor] = None, compute_loss: bool = True):
        if x.dim() == 2:
            x = x.view(-1, self.in_channels, self.input_h, self.input_w)
        # encode / decode
        z = self.encode(x)
        recon_x = self.decode(z)

        if not compute_loss:
            return AutoEncoderOutput(z=z, x_recon=recon_x, loss=None, loss_recon=None, latent_loss=None)

        loss_recon = F.mse_loss(recon_x, x, reduction='mean') * self.mag_recon_loss
        target_J = y.to(z.dtype).to(z.device)
        loss_latent = F.mse_loss(z, target_J, reduction='sum')
        loss = loss_recon + loss_latent

        return AutoEncoderOutput(z=z, x_recon=recon_x, loss=loss, loss_recon=loss_recon, latent_loss=loss_latent)



class ViTAutoencoder(nn.Module):
    def __init__(self, 
                 D_K, D_E, 
                 patch_size=10, 
                 in_channels=1, 
                 embed_dim=128, 
                 latent_dim=3, 
                 num_heads=4, 
                 num_layers=4,
                mag_recon_loss=1.0):
        super().__init__()
        
        self.mag_recon_loss = mag_recon_loss
        self.img_h, self.img_w = D_K, D_E
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.in_channels = in_channels
        
        # Calculate number of patches
        # 100/10 = 10, 300/10 = 30 -> 10 * 30 = 300 patches
        self.grid_h = self.img_h // patch_size
        self.grid_w = self.img_w // patch_size
        self.num_patches = self.grid_h * self.grid_w
        
        # --- ENCODER ---
        # 1. Patch Embedding (We use a Conv2d with stride=patch_size to do this fast)
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # 2. Positional Embedding (Learnable locations)
        # We add 1 for the [CLS] token (the token that will hold the parameters)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        
        # 3. Transformer Encoder Blocks
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 4. Bottleneck: [CLS] token -> Hamiltonian Params
        self.to_params = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, latent_dim)
        )
        
        # --- DECODER ---
        # 1. Expand Latent -> Sequence
        # We map the 3 params back to a full sequence of 300 patches
        self.from_params = nn.Linear(latent_dim, self.num_patches * embed_dim)
        
        # 2. Decoder Positional Embedding
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        
        # 3. Transformer Decoder Blocks
        decoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer_decoder = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)
        
        # 4. Un-patchify: Project embeddings back to pixel values
        # Output dim = patch_height * patch_width * channels (e.g., 10*10*1 = 100 pixels per patch)
        self.pixel_pred = nn.Linear(embed_dim, patch_size * patch_size * in_channels)

    def encode(self, x):
        B = x.shape[0]
        # --- ENCODE ---
        # 1. Create Patches: (B, C, H, W) -> (B, Embed_Dim, Grid_H, Grid_W)
        x_patches = self.patch_embed(x)
        # 2. Flatten to Sequence: (B, Embed_Dim, Num_Patches) -> (B, Num_Patches, Embed_Dim)
        x_patches = x_patches.flatten(2).transpose(1, 2)
        # 3. Append [CLS] token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x_seq = torch.cat((cls_tokens, x_patches), dim=1)
        # 4. Add Position Embeddings
        x_seq = x_seq + self.pos_embed
        # 5. Apply Transformer
        x_encoded = self.transformer_encoder(x_seq)
        # 6. Extract [CLS] token for Latent Variables
        cls_out = x_encoded[:, 0]
        z = self.to_params(cls_out) # This is your J1, J2, etc.
        z = nn.Tanh()(z)  # constrain latent space to [-1,1]
        return z
    
    def decode(self, z):
        B = z.shape[0]
        # --- DECODE ---
        # 1. Map parameters back to sequence
        decoder_input = self.from_params(z)
        decoder_input = decoder_input.view(B, self.num_patches, self.embed_dim)
        # 2. Add Decoder Positional Embeddings
        decoder_input = decoder_input + self.decoder_pos_embed
        # 3. Apply Transformer Decoder
        x_decoded = self.transformer_decoder(decoder_input)
        # 4. Project to Pixels
        # (B, Num_Patches, Pixels_Per_Patch)
        patches_reconstructed = self.pixel_pred(x_decoded)
        # 5. Reassemble Patches into Image (Unpatchify)
        # This requires some dimension shuffling
        # Current: (B, Grid_H * Grid_W, P*P*C)
        # Target:  (B, C, H, W)
        # Split pixels back to (P, P)
        patches_reconstructed = patches_reconstructed.view(
            B, self.grid_h, self.grid_w, self.patch_size, self.patch_size
        )
        # Permute to (B, Grid_H, P, Grid_W, P) to align rows/cols
        # Then combine dimensions
        recon_image = torch.einsum('bhwqp->bhqwp', patches_reconstructed)
        recon_image = recon_image.reshape(B, 1, self.img_h, self.img_w)
        return recon_image

    def forward(self, x, y=None, compute_loss=True):
        if x.dim() == 2:
            x = x.view(-1, self.in_channels, self.img_h, self.img_w)
        # encode / decode
        z = self.encode(x)
        recon_x = self.decode(z)
        
        if not compute_loss:
            return AutoEncoderOutput(z=z, x_recon=recon_x, loss=None, loss_recon=None, latent_loss=None)

        loss_recon = F.mse_loss(recon_x, x, reduction='mean') * self.mag_recon_loss
        target_J = y.to(z.dtype).to(z.device)
        loss_latent = F.mse_loss(z, target_J, reduction='sum')
        loss = loss_recon + loss_latent

        return AutoEncoderOutput(z=z, x_recon=recon_x, loss=loss, loss_recon=loss_recon, latent_loss=loss_latent)




class Conv1DAutoencoder(nn.Module):
    """
    Two independent 1D-conv pipelines (rows and columns). Each pipeline:
      - encodes sequences with Conv1d and AdaptiveAvgPool1d(1) -> per-row / per-col descriptors
      - descriptors are concatenated and mapped to latent (to_params)
    Decoder (reversed):
      - map latent -> concatenated descriptors (from_params)
      - split into row / column descriptor sets
      - expand each descriptor to a full 1D sequence via a small linear head
      - refine with Conv1d and project to pixel values
      - combine row- and column-based reconstructions (average) -> final image
    """
    def __init__(self, D_K, D_E, in_channels=1, base_channels=32, latent_dim=3,kernel_size=3, 
                 mag_recon_loss=1.0, ifdecode = True):
        super(Conv1DAutoencoder, self).__init__()
        self.input_h = D_K
        self.input_w = D_E
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.mag_recon_loss = mag_recon_loss
        self.base_channels = base_channels
        self.ifdecode = ifdecode

        pad = kernel_size // 2
        # conv that runs across each row (convolve over width / energy axis)
        self.conv_row = nn.Conv1d(in_channels, base_channels, kernel_size=kernel_size, padding=pad)
        # conv that runs across each column (convolve over height / momentum axis)
        self.conv_col = nn.Conv1d(in_channels, base_channels, kernel_size=kernel_size, padding=pad)

        # pool each 1D output to a single descriptor per row/column
        self.pool = nn.AdaptiveAvgPool1d(1)

        # combined feature size: H * row_channels + W * col_channels
        self.combined_dim = self.input_h * base_channels + self.input_w * base_channels

        # bottleneck / decoder heads
        self.to_params = nn.Linear(self.combined_dim, latent_dim)
        # decode back to flattened image
        self.from_params = nn.Linear(latent_dim, self.combined_dim)

        # Decoder expansion heads: pooled descriptor -> full 1D sequence
        # For rows: expand each (base_channels) descriptor to base_channels * W then reshape to (base_channels, W)
        self.row_expand = nn.Linear(base_channels, base_channels * self.input_w)
        # For columns: expand to base_channels * H
        self.col_expand = nn.Linear(base_channels, base_channels * self.input_h)

        # small conv refinement after expansion (keeps dims constant)
        self.conv_row_refine = nn.Conv1d(base_channels, base_channels, kernel_size=kernel_size, padding=pad)
        self.conv_col_refine = nn.Conv1d(base_channels, base_channels, kernel_size=kernel_size, padding=pad)

        # project channel dimension down to image channels (usually 1)
        self.to_pixel_row = nn.Conv1d(base_channels, in_channels, kernel_size=1)
        self.to_pixel_col = nn.Conv1d(base_channels, in_channels, kernel_size=1)

        self.act = nn.LeakyReLU(0.1)
        self.final_act = nn.Tanh()


    def encode(self, x):
        # expected x: (B, C, H, W)
        B = x.shape[0]
        # Row-wise conv: treat each row as a sequence of length W
        # reshape -> (B * H, C, W)
        row_in = x.view(B * self.input_h, self.in_channels, self.input_w)
        row_feat = self.conv_row(row_in)                   # (B*H, row_ch, W)
        row_feat = self.act(row_feat)
        row_pooled = self.pool(row_feat).squeeze(-1)       # (B*H, row_ch)
        row_pooled = row_pooled.view(B, self.input_h * row_pooled.shape[1])  # (B, H*row_ch)

        # Column-wise conv: treat each column as a sequence of length H
        col_in = x.permute(0, 1, 3, 2).contiguous()        # (B, C, W, H)
        col_in = col_in.view(B * self.input_w, self.in_channels, self.input_h)
        col_feat = self.conv_col(col_in)                   # (B*W, col_ch, H)
        col_feat = self.act(col_feat)
        col_pooled = self.pool(col_feat).squeeze(-1)       # (B*W, col_ch)
        col_pooled = col_pooled.view(B, self.input_w * col_pooled.shape[1])  # (B, W*col_ch)

        # concatenate row and column descriptors
        combined = torch.cat([row_pooled, col_pooled], dim=1)  # (B, combined_dim)
        z = self.to_params(combined)
        z = torch.tanh(z)
        return z

    def decode(self, z):
        B = z.shape[0]
        # Map latent back to concatenated descriptors
        recon_combined = self.from_params(z)  # (B, combined_dim)

        # Split into row and column descriptor flats
        row_flat = recon_combined[:, : (self.input_h * self.base_channels)]
        col_flat = recon_combined[:, (self.input_h * self.base_channels):]

        # Row path: (B, H*base_ch) -> (B*H, base_ch) -> expand -> (B*H, base_ch, W)
        # use reshape to avoid non-contiguous .view() errors
        row_pooled = row_flat.reshape(B * self.input_h, self.base_channels)
        row_seq = self.row_expand(row_pooled)                       # (B*H, base_ch * W)
        row_seq = row_seq.view(B * self.input_h, self.base_channels, self.input_w)  # (B*H, base_ch, W)
        row_seq = self.conv_row_refine(row_seq)                     # refine
        row_seq = self.act(row_seq)
        row_pixels = self.to_pixel_row(row_seq)                     # (B*H, in_ch, W)
        # collapse channel dim (in_ch expected = 1)
        row_pixels = row_pixels.view(B, self.input_h, self.in_channels, self.input_w)  # (B, H, C, W)
        # bring to (B, C, H, W)
        row_image = row_pixels.permute(0, 2, 1, 3).contiguous().view(B, self.in_channels, self.input_h, self.input_w)

        # Column path: (B, W*base_ch) -> (B*W, base_ch) -> expand -> (B*W, base_ch, H)
        col_pooled = col_flat.reshape(B * self.input_w, self.base_channels)
        col_seq = self.col_expand(col_pooled)                       # (B*W, base_ch * H)
        col_seq = col_seq.view(B * self.input_w, self.base_channels, self.input_h)   # (B*W, base_ch, H)
        col_seq = self.conv_col_refine(col_seq)
        col_seq = self.act(col_seq)
        col_pixels = self.to_pixel_col(col_seq)                     # (B*W, in_ch, H)
        col_pixels = col_pixels.view(B, self.input_w, self.in_channels, self.input_h)  # (B, W, C, H)
        # permute to (B, C, H, W)
        col_image = col_pixels.permute(0, 2, 3, 1).contiguous()     # (B, C, H, W)

        # Combine row and column reconstructions (simple average)
        recon = 0.5 * (row_image + col_image)
        recon = self.final_act(recon)
        return recon

    def forward(self, x, y: Optional[torch.Tensor] = None, compute_loss: bool = True):
        if x.dim() == 2:
            x = x.view(-1, self.in_channels, self.input_h, self.input_w)
        # encode / decode
        z = self.encode(x)
        if self.ifdecode:
            recon_x = self.decode(z)
        else:
            recon_x = torch.zeros_like(x)  # set to zero tensor of same shape as x

        if not compute_loss:
            return AutoEncoderOutput(z=z, x_recon=recon_x, loss=None, loss_recon=None, latent_loss=None)

        if self.ifdecode:
            loss_recon = F.mse_loss(recon_x, x, reduction='mean') * self.mag_recon_loss
        else:
            loss_recon = torch.tensor(0.0, device=x.device)
        target_J = y.to(z.dtype).to(z.device)
        loss_latent = F.mse_loss(z, target_J, reduction='sum')
        loss = loss_recon + loss_latent

        return AutoEncoderOutput(z=z, x_recon=recon_x, loss=loss, loss_recon=loss_recon, latent_loss=loss_latent)


class ResNetEncoder(nn.Module):
    """
    ResNet-based Autoencoder (supervised latent).
    Uses a ResNet backbone as encoder (up to final conv block) and a mirrored
    ConvTranspose decoder. Latent space is constrained with Tanh to [-1,1].
    Args:
        D_K, D_E: input height and width
        in_channels: input channels (usually 1)
        latent_dim: size of latent (should be 3 for J1,J2,J3)
        pretrained: whether to use pretrained ResNet weights for encoder
        mag_recon_loss: multiplier for reconstruction loss
    """
    def __init__(self, D_K, D_E, in_channels=1, latent_dim=3,
                 pre_fc_out_channels: Optional[int]=None):
        super(ResNetEncoder, self).__init__()
        self.input_h = D_K
        self.input_w = D_E
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.pre_fc_out_channels = pre_fc_out_channels

        if pre_fc_out_channels is not None:
            in_flat = in_channels * D_K * D_E
            out_flat = pre_fc_out_channels * 28*28 #D_K * D_E
            self.pre_fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_flat, out_flat),
                nn.ReLU(),
                nn.Unflatten(1, (pre_fc_out_channels, 28, 28)),
            )
            conv_in_channels = pre_fc_out_channels
        else:
            conv_in_channels = in_channels
        
        # Load resnet18 and keep layers up to layer4 (exclude avgpool and fc)
        resnet = models.resnet18(weights=None)
        # adapt first conv if in_channels != 3
        if in_channels != 3:
            # replace conv1 to accept in_channels
            orig = resnet.conv1
            resnet.conv1 = nn.Conv2d(conv_in_channels, orig.out_channels,
                                     kernel_size=orig.kernel_size,
                                     stride=orig.stride, # 2
                                     padding=orig.padding,
                                     bias=orig.bias is not None)

        # encoder backbone: conv1, bn1, relu, maxpool, layer1..layer4
        self.encoder_backbone = nn.Sequential(
            resnet.conv1, # (B, 1,100,300) -> (B, 64, 50,150)
            resnet.bn1, # (B, 64, 50,150)
            resnet.relu, # (B, 64, 50,150)
            resnet.maxpool, # (B, 64, 25,75)
            resnet.layer1, # (B, 64, 25,75)
            resnet.layer2, # (B, 128, 13,38)
            resnet.layer3, # (B, 256, 7,19)
            resnet.layer4, # (B, 512, 4,10)
        )

        # run a dummy input to determine feature map shape
        with torch.no_grad():
            if self.pre_fc_out_channels is not None:
                dummy = torch.zeros(1, self.pre_fc_out_channels, 28, 28)
            else:
                dummy = torch.zeros(1, in_channels, D_K, D_E)
            self.encoder_backbone.eval()
            feat = self.encoder_backbone(dummy)
            self.feature_c = feat.shape[1]
            self.feature_h = feat.shape[2]
            self.feature_w = feat.shape[3]
            self.flat_dim = self.feature_c * self.feature_h * self.feature_w

        # bottleneck
        self.to_params = nn.Linear(self.flat_dim, latent_dim)

    def encode(self, x):
        if x.dim() == 2:
            x = x.view(-1, self.in_channels, self.input_h, self.input_w)
        if self.pre_fc_out_channels is not None:
            x = self.pre_fc(x)
        features = self.encoder_backbone(x)  # (B, C, H', W')
        flat = features.view(features.size(0), -1)
        z = self.to_params(flat)
        z = torch.tanh(z)
        return z

    def forward(self, x, y: Optional[torch.Tensor] = None, compute_loss: bool = True):
        # require supervised latent y (consistent with other classes)
        if y is None:
            raise ValueError("y (target J parameters) must be provided to forward().")
        # prepare input
        if x.dim() == 2:
            x_in = x.view(-1, self.in_channels, self.input_h, self.input_w)
        else:
            x_in = x
        z = self.encode(x_in)
        if not compute_loss:
            return AutoEncoderOutput(z=z, x_recon=None, loss=None, loss_recon=None, latent_loss=None)

        target_J = y.to(z.dtype).to(z.device)
        loss_latent = F.mse_loss(z, target_J, reduction='sum')
        loss = loss_latent
        loss_recon = torch.tensor(0.0, device=x.device)

        return AutoEncoderOutput(z=z, x_recon=None, loss=loss, loss_recon=loss_recon, latent_loss=loss_latent)
    

class FCBsatDecoder(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim=1024,
        latent_dim=3,
        mag_recon_loss=1.0,
        ifdecode=True,
        relative_loss=False,
        bsat_index=-1,
    ):
        super(FCBsatDecoder, self).__init__()
        self.latent_dim = latent_dim
        self.mag_recon_loss = mag_recon_loss
        self.ifdecode = ifdecode
        self.relative_loss = relative_loss
        self.bsat_index = bsat_index

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.SiLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 8),
            nn.SiLU(),
            nn.Linear(hidden_dim // 8, latent_dim),
        )

        # Decoder now predicts only Bsat (shape [B,1])
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 8),
            nn.SiLU(),
            nn.Linear(hidden_dim // 8, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def encode(self, x, eps: float = 1e-8):
        z = self.encoder(x)
        z = torch.tanh(z)
        return z

    def decode(self, z):
        return self.decoder(z)

    def forward(
        self,
        x,
        y: Optional[torch.Tensor] = None,
        bsat_target: Optional[torch.Tensor] = None,
        compute_loss: bool = True,
    ):
        z = self.encode(x)

        if self.ifdecode:
            recon_bsat = self.decode(z)  # [B,1]
        else:
            recon_bsat = torch.zeros((x.size(0), 1), device=x.device, dtype=x.dtype)

        if not compute_loss:
            return AutoEncoderOutput(z=z, x_recon=recon_bsat, loss=None, loss_recon=None, latent_loss=None)

        if bsat_target is None:
            # fallback: use Bsat feature from input
            bsat_target = x[:, self.bsat_index].unsqueeze(1)

        bsat_target = bsat_target.to(z.dtype).to(z.device)

        if self.ifdecode:
            loss_recon = F.mse_loss(recon_bsat, bsat_target, reduction='sum') * self.mag_recon_loss
        else:
            loss_recon = torch.tensor(0.0, device=x.device)

        target_J = y.to(z.dtype).to(z.device)
        tsl = F.mse_loss(z, target_J, reduction='sum')
        rel_err = ((z - target_J) ** 2) / ((target_J.abs()) ** 2 + 1e-12)
        rel_err = rel_err.sum()
        loss_latent = rel_err if self.relative_loss else tsl

        loss = loss_recon + loss_latent

        return AutoEncoderOutput(
            z=z,
            x_recon=recon_bsat,
            loss=loss,
            loss_recon=loss_recon,
            latent_loss=loss_latent,
            relative_loss=rel_err,
        )
    

class FCBsats3Decoder(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim=1024,
        latent_dim=3,
        mag_recon_loss=1.0,
        ifdecode=True,
        relative_loss=False,
        bsat_dim=3,                       # NEW
        bsat_slice=slice(-3, None),       # NEW: where Bsat vector lives in x
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.mag_recon_loss = mag_recon_loss
        self.ifdecode = ifdecode
        self.relative_loss = relative_loss
        self.bsat_dim = bsat_dim
        self.bsat_slice = bsat_slice

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.SiLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4), nn.SiLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 8), nn.SiLU(),
            nn.Linear(hidden_dim // 8, latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 8), nn.SiLU(),
            nn.Linear(hidden_dim // 8, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, bsat_dim),
        )

    def encode(self, x):
        return torch.tanh(self.encoder(x))

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x, y=None, bsat_target=None, compute_loss=True):
        z = self.encode(x)

        if self.ifdecode:
            recon_bsat = self.decode(z)   # [B, bsat_dim]
        else:
            recon_bsat = torch.zeros((x.size(0), self.bsat_dim), device=x.device, dtype=x.dtype)

        if not compute_loss:
            return AutoEncoderOutput(z=z, x_recon=recon_bsat, loss=None, loss_recon=None, latent_loss=None)

        if bsat_target is None:
            # CHANGED: fallback to vector slice
            bsat_target = x[:, self.bsat_slice]   # [B, bsat_dim]

        bsat_target = bsat_target.to(z.dtype).to(z.device)
        if bsat_target.dim() == 1:
            bsat_target = bsat_target.unsqueeze(1)

        if self.ifdecode:
            loss_recon = F.mse_loss(recon_bsat, bsat_target, reduction="sum") * self.mag_recon_loss
        else:
            loss_recon = torch.tensor(0.0, device=x.device)

        target_J = y.to(z.dtype).to(z.device)
        tsl = F.mse_loss(z, target_J, reduction="sum")
        rel_err = ((z - target_J) ** 2) / ((target_J.abs()) ** 2 + 1e-12)
        rel_err = rel_err.sum()
        loss_latent = rel_err if self.relative_loss else tsl
        loss = loss_recon + loss_latent

        return AutoEncoderOutput(
            z=z, x_recon=recon_bsat, loss=loss,
            loss_recon=loss_recon, latent_loss=loss_latent, relative_loss=rel_err,
        )