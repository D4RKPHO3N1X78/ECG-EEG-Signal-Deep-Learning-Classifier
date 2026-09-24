import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv2dWithConstraint(nn.Conv2d):
    """
    2D Convolution with max-norm constraint on weights for regularization.
    """
    def __init__(self, *args, max_norm: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_norm = max_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            with torch.no_grad():
                self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return super().forward(x)


class EEGNetTransformer(nn.Module):
    """
    EEGNet + Transformer Hybrid Architecture for Multi-Channel EEG Abnormality Classification.
    Input shape: (batch_size, 1, num_channels, sequence_length) or (batch_size, num_channels, sequence_length)
    """
    EEG_CLASSES = [
        "Normal EEG (Baseline Alpha/Beta)",
        "Epileptic Seizure Activity (Ictal Spikes)",
        "Focal Spike & Waveform Abnormality",
        "Slow-Wave Encephalopathy (Delta/Theta)",
        "Artifact Interference (Ocular/EMG)"
    ]

    def __init__(self, num_channels: int = 8, num_classes: int = 5, samples: int = 500, F1: int = 8, D: int = 2, F2: int = 16):
        super().__init__()
        self.num_channels = num_channels
        self.num_classes = num_classes
        
        # Block 1: Temporal Conv + Depthwise Spatial Conv
        self.conv1 = nn.Conv2d(1, F1, (1, 64), padding=(0, 32), bias=False)
        self.bn1 = nn.BatchNorm2d(F1)
        self.depthwise = Conv2dWithConstraint(F1, F1 * D, (num_channels, 1), groups=F1, bias=False, max_norm=1.0)
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.act1 = nn.ELU()
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(0.25)

        # Block 2: Separable Conv
        self.separable = nn.Sequential(
            nn.Conv2d(F1 * D, F2, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F2, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(0.25)
        )

        # Temporal Transformer Stage
        transformer_dim = F2
        encoder_layer = nn.TransformerEncoderLayer(d_model=transformer_dim, nhead=4, dim_feedforward=64, dropout=0.2, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

        # Final Classifier
        self.classifier = nn.Sequential(
            nn.Linear(F2, 32),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (B, C, L) or (B, 1, C, L)
        if x.ndim == 3:
            x = x.unsqueeze(1) # Add channel dim -> (B, 1, C, L)
            
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.depthwise(x)
        x = self.bn2(x)
        x = self.act1(x)
        x = self.pool1(x)
        x = self.drop1(x)

        x = self.separable(x) # (B, F2, 1, L')
        
        # Reshape for Transformer: (B, L', F2)
        x = x.squeeze(2).permute(0, 2, 1)
        x = self.transformer(x)
        
        # Global Average Pooling over temporal dimension
        x = torch.mean(x, dim=1) # (B, F2)
        
        logits = self.classifier(x)
        return logits
