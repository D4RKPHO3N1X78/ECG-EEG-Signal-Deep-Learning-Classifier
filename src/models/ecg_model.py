import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock1D(nn.Module):
    """
    1D Residual Convolutional Block with BatchNorm, GELU activation, and Skip Connection.
    """
    def __init__(self, channels: int, kernel_size: int = 5, dropout: float = 0.2):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(channels)
        self.act1 = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(channels)
        self.act2 = nn.GELU()
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.dropout1(out)
        out = self.bn2(self.conv2(out))
        out = self.dropout2(out)
        out += residual
        return self.act2(out)


class ECGResNet1DBiLSTM(nn.Module):
    """
    Deep Learning Model for ECG Signal Classification (1D ResNet + Bi-LSTM + Self-Attention).
    Input shape: (batch_size, num_channels, sequence_length) e.g., (B, 1, 200) or (B, 12, 1000)
    """
    ECG_CLASSES = [
        "Normal Sinus Rhythm (NSR)",
        "Atrial Fibrillation (AFIB)",
        "Premature Ventricular Contraction (PVC)",
        "Supraventricular Tachycardia (SVT)",
        "ST-Elevation Myocardial Infarction (STEMI)"
    ]

    def __init__(self, in_channels: int = 1, num_classes: int = 5, hidden_dim: int = 64):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        
        # Stem Convolution
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )
        
        # Conv Stage 1
        self.stage1_proj = nn.Conv1d(32, 64, kernel_size=1)
        self.res1 = ResBlock1D(64)
        
        # Conv Stage 2
        self.stage2_proj = nn.Conv1d(64, 128, kernel_size=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.res2 = ResBlock1D(128)
        
        # Bidirectional LSTM Layer
        self.bilstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )
        
        # Self-Attention Layer
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, C, L)
        if x.ndim == 2:
            x = x.unsqueeze(1) # Add channel dim (B, 1, L)
            
        feat = self.stem(x)
        
        feat = self.stage1_proj(feat)
        feat = self.res1(feat)
        
        feat = self.stage2_proj(feat)
        feat = self.pool2(feat)
        feat = self.res2(feat) # Shape: (B, 128, L')
        
        # Permute for LSTM (B, L', 128)
        lstm_in = feat.permute(0, 2, 1)
        lstm_out, _ = self.bilstm(lstm_in) # (B, L', hidden_dim*2)
        
        # Attention Pooling
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1) # (B, L', 1)
        context = torch.sum(attn_weights * lstm_out, dim=1) # (B, hidden_dim*2)
        
        logits = self.classifier(context)
        return logits
