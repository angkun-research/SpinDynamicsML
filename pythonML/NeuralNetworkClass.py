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


def train(model, dataloader, optimizer, prev_updates, writer=None, twoheads=False, device='cpu'):
    model.train()
    running_loss = 0.0
    running_relative_loss = 0.0
    number_of_samples = 0.0

    for batch_idx, (data, target) in enumerate(tqdm(dataloader)):
        n_upd = prev_updates + batch_idx
        data = data.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        if not twoheads:
            pred = model(data)
            loss = F.mse_loss(pred, target, reduction='mean')
            relative_loss = (pred - target).abs() / (target.abs() + 1e-12)
            relative_loss = relative_loss.mean()
        else:
            cls_logits, reg_pred = model(data)
            y_cls = target[:, 0]
            y_cls = ((y_cls + 1) / 2).long()   # -1/+1 -> 0/1
            y_reg = target[:, 1:]
            loss_cls = F.cross_entropy(cls_logits, y_cls) # implicitly applies softmax to cls_logits
            loss_reg = F.mse_loss(reg_pred, y_reg, reduction='mean')
            loss = loss_cls + loss_reg
            # Keep relative error comparable to your old code:
            # classification is compared as -1/+1 prediction, regression as raw values
            cls_pred01 = torch.argmax(cls_logits, dim=1)
            cls_pred_pm1 = 2 * cls_pred01 - 1
            pred_full = torch.cat([cls_pred_pm1.unsqueeze(1).float(), reg_pred], dim=1)
            relative_loss = (pred_full - target).abs() / (target.abs() + 1e-12)
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


def test(model, dataloader, cur_step, writer=None, twoheads=False, device='cpu'):
    model.eval()
    running_loss = 0.0
    running_relative_loss = 0.0
    number_of_samples = 0.0

    with torch.no_grad():
        for data, target in tqdm(dataloader, desc='Testing'):
            data = data.to(device)
            target = target.to(device)
            if not twoheads:
                pred = model(data)
                loss = F.mse_loss(pred, target, reduction='mean')
                relative_loss = (pred - target).abs() / (target.abs() + 1e-12)
                relative_loss = relative_loss.mean()
            else:
                cls_logits, reg_pred = model(data)
                y_cls = target[:, 0]
                y_cls = ((y_cls + 1) / 2).long()   # -1/+1 -> 0/1
                y_reg = target[:, 1:]
                loss_cls = F.cross_entropy(cls_logits, y_cls) # implicitly applies softmax to cls_logits
                loss_reg = F.mse_loss(reg_pred, y_reg, reduction='mean')
                loss = loss_cls + loss_reg
                cls_pred01 = torch.argmax(cls_logits, dim=1)
                cls_pred_pm1 = 2 * cls_pred01 - 1
                pred_full = torch.cat([cls_pred_pm1.unsqueeze(1).float(), reg_pred], dim=1)
                relative_loss = (pred_full - target).abs() / (target.abs() + 1e-12)
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

class ConvType4(nn.Module):
    def __init__(self, in_channels, input_length, hidden_dim=256, output_dim=4,kernel_size=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=kernel_size,padding=kernel_size//2,padding_mode='circular'), 
            nn.SiLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size,padding=kernel_size//2,padding_mode='circular'), 
            nn.SiLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size,padding=kernel_size//2,padding_mode='circular'),
            nn.SiLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Flatten(),  # flatten before the final linear layer
            nn.Linear(hidden_dim * (input_length // 8), hidden_dim), # adjust input size for linear
            nn.SiLU(),
            # +1 output: first 2 for classification logits (class 0/1), remaining (output_dim - 1) for regression
            nn.Linear(hidden_dim, output_dim + 1)
        )
    def forward(self, x):
        x = self.net(x)                    # shape: [B, output_dim + 1]

        cls_logits = x[:, :2]            # class 0/1 logits
        reg_out = x[:, 2:]               # shape: [B, output_dim - 1]
    
        #cls_probs = torch.softmax(cls_logits, dim=1)
        #cls_out = torch.argmax(cls_probs, dim=1)  # predicted class labels (0 or 1)
        #cls_out = 2*cls_out - 1  # convert to -1 or +1
        #final_out = torch.cat([cls_out.unsqueeze(1).float(), reg_out], dim=1)  # shape: [B, output_dim]
        return cls_logits, reg_out


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