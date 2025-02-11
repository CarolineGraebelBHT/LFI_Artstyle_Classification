import torch
import torch.nn as nn

class BottleneckBlock(nn.Module):
    """
    Bottleneck Residual Block for ResNet-50 (1x1 → 3x3 → 1x1 convolutions)
    """

    def __init__(self, in_channels, out_channels, stride=1, expansion=4):
        super(BottleneckBlock, self).__init__()

        mid_channels = in_channels // expansion  # Correct channel calculation

        # 1x1 Conv (reduce dimensionality)
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)

        # 3x3 Conv (main transformation)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)

        # 1x1 Conv (restore dimensionality)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        # Skip connection (shortcut)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x  # Store original input for skip connection
        out = torch.relu(self.bn1(self.conv1(x)))
        out = torch.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        out += self.shortcut(residual)  # Add skip connection
        return torch.relu(out)


class ResNet50(nn.Module):
    """
    Full ResNet-50 Architecture (Convolution → Bottleneck Blocks → Fully Connected Layer)
    """

    def __init__(self, num_classes):
        super(ResNet50, self).__init__()

        self.init_conv = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.init_bn = nn.BatchNorm2d(64)
        self.init_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet-50 Layer Structure (3, 4, 6, 3 blocks per stage)
        self.layer1 = self._make_layer(64, 64, num_blocks=3, stride=1)
        self.layer2 = self._make_layer(256, 128, num_blocks=4, stride=2)
        self.layer3 = self._make_layer(512, 256, num_blocks=6, stride=2)
        self.layer4 = self._make_layer(1024, 512, num_blocks=3, stride=2)

        # Adaptive Average Pooling
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully Connected Layer
        self.fc = nn.Linear(2048, num_classes)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        """
        Creates a full ResNet layer using multiple Bottleneck Blocks.
        - First block applies stride and adjusts shortcut.
        - Remaining blocks keep the same dimensions.
        """
        expanded_channels = out_channels * 4  # ResNet expands channels
        layers = [BottleneckBlock(in_channels, expanded_channels, stride=stride)]
        for _ in range(1, num_blocks):
            layers.append(BottleneckBlock(expanded_channels, expanded_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = torch.relu(self.init_bn(self.init_conv(x)))
        x = self.init_pool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)
