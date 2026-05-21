from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

@dataclass
class RegressionOutput:
    y_pred: torch.Tensor
    loss: Optional[torch.Tensor] = None


def train(model, dataloader, optimizer, prev_updates, writer=None, device='cpu'):
    model.train()
    running_loss = 0.0
    running_relative_loss = 0.0
    number_of_samples = 0.0

    for batch_idx, (data, target) in enumerate(tqdm(dataloader)):
        n_upd = prev_updates + batch_idx
        data = data.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        pred = model(data)
        loss = F.mse_loss(pred, target, reduction='mean')
        relative_loss = (pred - target).abs() / (target.abs() + 1e-12)
        relative_loss = relative_loss.mean()
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # gradient clipping to prevent exploding gradients
        optimizer.step()

        running_loss += loss.item()*data.size(0)  # accumulate total loss for the batch
        running_relative_loss += relative_loss.item()*data.size(0)  # accumulate total relative loss for the batch
        number_of_samples += data.size(0)  # accumulate total number of samples processed

        if n_upd % 100 == 0 and writer is not None:
            writer.add_scalar('Loss/train', loss.item(), n_upd)

    avg_batch_loss = running_loss / number_of_samples
    avg_relative_loss = running_relative_loss / number_of_samples
    print(f'====> Train set loss: {avg_batch_loss:.6e}, relative loss: {avg_relative_loss:.6e}')
    return prev_updates + len(dataloader)


def test(model, dataloader, cur_step, writer=None, device='cpu'):
    model.eval()
    running_loss = 0.0
    running_relative_loss = 0.0
    number_of_samples = 0.0

    with torch.no_grad():
        for data, target in tqdm(dataloader, desc='Testing'):
            data = data.to(device)
            target = target.to(device)
            pred = model(data)
            loss = F.mse_loss(pred, target, reduction='mean')
            relative_loss = (pred - target).abs() / (target.abs() + 1e-12)
            relative_loss = relative_loss.mean()
            running_loss += loss.item()*data.size(0)
            running_relative_loss += relative_loss.item()*data.size(0)
            number_of_samples += data.size(0)

    avg_batch_loss = running_loss / number_of_samples
    avg_relative_loss = running_relative_loss / number_of_samples
    print(f'====> Test set loss: {avg_batch_loss:.6e}, relative loss: {avg_relative_loss:.6e}')

    if writer is not None:
        writer.add_scalar('Loss/test', avg_batch_loss, global_step=cur_step)


class RegressionMLP(nn.Module):
    def __init__(self, in_channels, input_length, hidden_dim=256, output_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),  # flatten the input
            nn.Linear(in_channels * input_length, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
            # nn.Linear(hidden_dim, hidden_dim//2),
            # nn.SiLU(),
            # nn.Linear(hidden_dim//2, output_dim),
        )

    def forward(self, x):
        x = self.net(x)
        #x = torch.tanh(x.pow(3))  # constrain outputs to [-1, 1]
        return x

class ConvType1(nn.Module):
    def __init__(self, in_channels, input_length, hidden_dim=256, output_dim=4,kernel_size=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=kernel_size,padding=kernel_size//2,padding_mode='circular'), 
            nn.SiLU(),
            nn.Flatten(),  # flatten before the final linear layer
            nn.Linear(hidden_dim * input_length, hidden_dim), # adjust input size for linear
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        x = self.net(x)
        #x = torch.tanh(x.pow(3))  # constrain outputs to [-1, 1]
        return x

class ConvType2(nn.Module):
    def __init__(self, in_channels, input_length, hidden_dim=256, output_dim=4,kernel_size=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=kernel_size,padding=kernel_size//2,padding_mode='circular'), 
            nn.SiLU(),
            nn.AvgPool1d(kernel_size=2, stride=2), 
            nn.Flatten(),  # flatten before the final linear layer
            nn.Linear(hidden_dim * (input_length // 2), hidden_dim), # adjust input size for linear
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        x = self.net(x)
        #x = torch.tanh(x.pow(3))  # constrain outputs to [-1, 1]
        return x


class ConvType3(nn.Module):
    def __init__(self, in_channels, input_length, hidden_dim=256, output_dim=4,kernel_size=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=kernel_size,padding=kernel_size//2,padding_mode='circular'), 
            nn.SiLU(),
            nn.AvgPool1d(kernel_size=2, stride=2), 
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size,padding=kernel_size//2,padding_mode='circular'), 
            nn.SiLU(),
            nn.AvgPool1d(kernel_size=2, stride=2), 
            nn.Flatten(),  # flatten before the final linear layer
            nn.Linear(hidden_dim * (input_length // 4), hidden_dim), # adjust input size for linear
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        x = self.net(x)
        #x = torch.tanh(x.pow(3))  # constrain outputs to [-1, 1]
        return x


class TransformerType1(nn.Module):
    def __init__(self, in_channels, input_length, hidden_dim=256, output_dim=4,kernel_size=3):
        super().__init__()
        self.input_proj = nn.Linear(in_channels, hidden_dim)

        self.transformer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 2,
            activation='relu',
            batch_first=True,
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_dim * input_length, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        x = x.transpose(1, 2)  # Transformer expects input shape (seq_len, batch_size, d_model), so we transpose from (batch_size, d_model, seq_len)
        x = self.input_proj(x)     # [B, 20, hidden_dim]
        x = self.transformer(x)    # [B, 20, hidden_dim]
        x = self.fc(x)            # [B, output_dim]
        return x

class ConvTransformer(nn.Module):
    def __init__(self, in_channels, input_length, hidden_dim=256, output_dim=4,kernel_size=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(
                in_channels,
                hidden_dim,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                padding_mode='circular',
            ),
            nn.SiLU(),
        )
        self.transformer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 2,
            activation='relu',
            batch_first=True,
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_dim * input_length, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        x = self.conv(x)          # [B, hidden_dim, L]
        x = x.transpose(1, 2)     # [B, L, hidden_dim]
        x = self.transformer(x)   # [B, L, hidden_dim]
        x = self.fc(x) 
        #x = torch.tanh(x.pow(3))  # constrain outputs to [-1, 1]
        return x