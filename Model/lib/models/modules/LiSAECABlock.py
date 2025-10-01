"""
@author   :   andredalwin    
@modified :   chuasharmaine
@DateTime :   2025/10/01
@Version  :   1.0
"""

import torch
import torch.nn as nn

class ECABlock(nn.Module):
    def __init__(self, channels, k_size=3, dim="2d"):
        super(ECABlock, self).__init__()
        self.dim = dim
        if dim == "3d":
            self.avg_pool = nn.AdaptiveAvgPool3d(1)
        elif dim == "2d":
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        else:
            raise ValueError(f"Invalid dimension '{dim}' for ECABlock")

        # ECA uses a 1D conv over the channel dimension
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size,
                              padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Global pooling
        y = self.avg_pool(x)  # shape: (B, C, 1, 1) or (B, C, 1, 1, 1)
        y = y.squeeze(-1).squeeze(-1) if self.dim == "2d" else y.squeeze(-1).squeeze(-1).squeeze(-1)  # (B, C)

        # Channel attention
        y = y.unsqueeze(1)               # (B, 1, C)
        y = self.conv(y)                 # (B, 1, C)
        y = self.sigmoid(y)              # (B, 1, C)
        y = y.squeeze(1)                 # (B, C)

        # Reshape for broadcast
        if self.dim == "3d":
            y = y.view(y.size(0), y.size(1), 1, 1, 1)
        else:
            y = y.view(y.size(0), y.size(1), 1, 1)

        return x * y