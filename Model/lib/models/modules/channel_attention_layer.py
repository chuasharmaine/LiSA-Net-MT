import torch
import torch.nn as nn

def conv3x3(in_planes, out_planes, stride=1, bias=False, group=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, groups=group, bias=bias)


class ECA_Conv_Block(nn.Module):
    """
    ECA-based convolution block (replacement for SE_Conv_Block).
    https://github.com/BangguWu/ECANet/blob/master/models/eca_module.py

    This block enhances feature maps using Efficient Channel Attention (ECA),
    while keeping parameter overhead low. It follows a bottleneck-style design:
    - Conv1 (3x3) -> BN -> ReLU
    - Conv2 (3x3, channel expansion) -> BN
    - Apply ECA attention on Conv2 output
    - Residual connection (with projection if input/output channels differ)
    - Conv3 (3x3, channel reduction) -> BN -> ReLU
    - Optional Dropout

    Args:
        inplanes (int): Number of input channels
        planes (int): Base number of output channels (before expansion)
        stride (int): Stride for Conv1 and optional downsampling
        downsample (nn.Module, optional): Unused here (kept for compatibility)
        drop_out (bool): Whether to apply 2D dropout at the end
        k_size (int): Kernel size for ECA 1D convolution
    Returns:
        out (Tensor): Transformed feature map, shape (B, planes, H, W)
        y (Tensor): ECA attention weights, shape (B, planes*2, 1, 1)
    """

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, drop_out=False, k_size=3):
        super(ECA_Conv_Block, self).__init__()

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes * 2)
        self.bn2 = nn.BatchNorm2d(planes * 2)
        self.conv3 = conv3x3(planes * 2, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.stride = stride
        self.dropout = drop_out

        # ECA attention
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.eca_conv = nn.Conv1d(
            1, 1, kernel_size=k_size,
            padding=(k_size - 1) // 2, bias=False
        )
        self.sigmoid = nn.Sigmoid()

        self.downchannel = None
        if inplanes != planes:
            self.downchannel = nn.Sequential(nn.Conv2d(inplanes, planes * 2, kernel_size=1, stride=stride, bias=False),
                                             nn.BatchNorm2d(planes * 2),)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downchannel is not None:
            residual = self.downchannel(x)

        # ECA channel attention
        y = self.global_avg_pool(out)                  # (B, C, 1, 1)
        y = y.squeeze(-1).transpose(-1, -2)           # (B, 1, C)
        y = self.eca_conv(y)                          # 1D conv across channels
        y = self.sigmoid(y).transpose(-1, -2).unsqueeze(-1)  # reshape back to (B, C, 1, 1)
        out = out * y.expand_as(out)                  # apply attention weights

        out += residual
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)
        if self.dropout:
            out = nn.Dropout2d(0.5)(out)

        return out, y