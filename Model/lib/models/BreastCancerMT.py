"""
Carlos Aumente-Maestro, Jorge Díez, Beatriz Remeseiro,
"A Multi-Task Framework for Breast Cancer Segmentation and Classification in Ultrasound Imaging"

https://github.com/caumente/multi_task_breast_cancer

Notes:
- This is a modified and adapted version of the original implementation.
- Changes were made to integrate the model into the LiSA-Net training pipeline.
- Modifications include adjustments to the model structure, data handling, and training flow to support compatibility with our framework.
- This implementation is used for research and comparison purposes only and does not represent the official code.
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Standard convolutional block with batch norm and ReLU"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                             stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class DoubleConvBlock(nn.Module):
    """Two consecutive conv blocks"""
    def __init__(self, in_channels, out_channels):
        super(DoubleConvBlock, self).__init__()
        self.conv1 = ConvBlock(in_channels, out_channels, 3, 1, 1)
        self.conv2 = ConvBlock(out_channels, out_channels, 3, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class EncoderBlock(nn.Module):
    """Encoder block: double conv + max pool"""
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.double_conv = DoubleConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        conv_out = self.double_conv(x)
        pool_out = self.pool(conv_out)
        return pool_out, conv_out


class DecoderBlock(nn.Module):
    """Decoder block: upsample + concat skip + double conv"""
    def __init__(self, in_channels, out_channels, skip_channels=None):
        super(DecoderBlock, self).__init__()
        if skip_channels is None:
            skip_channels = out_channels
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.double_conv = DoubleConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, x, skip=None):
        x = self.upsample(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.double_conv(x)
        return x


class ClassificationHead(nn.Module):
    """Classification head with global average pooling and FC layers"""
    def __init__(self, in_channels, out_channels, dropout=0.5):
        super(ClassificationHead, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, out_channels)
        )

    def forward(self, x):
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


class BreastCancerMT(nn.Module):
    """
    Multitask U-Net for simultaneous segmentation and classification
    
    Args:
        in_channels (int): Number of input channels (default: 1)
        seg_out_channels (int): Number of segmentation output channels (default: 2)
        cls_out_channels (int): Number of classification output channels (default: 4)
    """
    def __init__(self, in_channels=1, seg_out_channels=2, cls_out_channels=4):
        super(BreastCancerMultitaskNet, self).__init__()
        
        # Encoder
        self.enc1 = EncoderBlock(in_channels, 64)
        self.enc2 = EncoderBlock(64, 128)
        self.enc3 = EncoderBlock(128, 256)
        self.enc4 = EncoderBlock(256, 512)
        
        # Bottleneck
        self.bottleneck = DoubleConvBlock(512, 1024)
        
        # Segmentation Decoder
        self.dec4 = DecoderBlock(1024, 512, skip_channels=512)
        self.dec3 = DecoderBlock(512, 256, skip_channels=256)
        self.dec2 = DecoderBlock(256, 128, skip_channels=128)
        self.dec1 = DecoderBlock(128, 64, skip_channels=64)
        self.seg_head = nn.Conv2d(64, seg_out_channels, kernel_size=1)
        
        # Classification Head
        self.cls_head = ClassificationHead(1024, cls_out_channels)

    def forward(self, x):
        # Encoder with skip connections
        x1, skip1 = self.enc1(x)
        x2, skip2 = self.enc2(x1)
        x3, skip3 = self.enc3(x2)
        x4, skip4 = self.enc4(x3)
        
        # Bottleneck - shared representation
        bottleneck = self.bottleneck(x4)
        
        # Segmentation path
        x = self.dec4(bottleneck, skip4)
        x = self.dec3(x, skip3)
        x = self.dec2(x, skip2)
        x = self.dec1(x, skip1)
        seg_out = self.seg_head(x)
        
        # Classification path
        cls_out = self.cls_head(bottleneck)
        
        # Return dict
        return {
            "segmentation": seg_out,
            "classification": cls_out
        }


if __name__ == '__main__':
    import torch
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    print("\n" + "="*60)
    print("Testing BreastCancerMultitaskNet")
    print("="*60)
    
    # Test with batch size 2, 1 channel, 128x128 image
    x = torch.randn(2, 1, 128, 128).to(device)
    
    # Multitask model
    model = BreastCancerMultitaskNet(
        in_channels=1,
        seg_out_channels=2,
        cls_out_channels=4
    ).to(device)
    
    with torch.no_grad():
        output = model(x)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Output type: {type(output)}")
    print(f"Output keys: {list(output.keys())}")
    for key, val in output.items():
        print(f"  {key}: {val.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params/1e6:.2f}M")