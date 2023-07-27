"""
MobileNetV2 with slight modification to support variable input image size
"""

import torch
from torch import nn


def to_nearest_multiple_of(num, multiple):
    return round(num / multiple) * multiple


class DepthWiseConv(nn.Module):
    def __init__(self, num_channels, kernel_size=3, stride=1):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(
            num_channels,
            num_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            groups=num_channels,
            bias=False,
        )

    def forward(self, x):
        # (B, C, H, W) -> (B, C, H//stride, W//stride)
        x = self.conv(x)
        return x


class PointWiseConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        # (B, Cin, H, W) -> (B, Cout, H, W)
        x = self.conv(x)
        return x


class InvertedResidualConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expansion_factor):
        super().__init__()
        self.stride = stride
        self.expansion_factor = expansion_factor
        self.res_connect = (in_channels == out_channels) and (stride == 1)
        expanded_channels = int(in_channels * expansion_factor)

        if self.expansion_factor != 1:
            self.pw_conv1 = PointWiseConv(in_channels, expanded_channels)
            self.bn1 = nn.BatchNorm2d(expanded_channels)
            self.relu1 = nn.ReLU6()

        self.dw_conv2 = DepthWiseConv(expanded_channels, stride=stride)
        self.bn2 = nn.BatchNorm2d(expanded_channels)
        self.relu2 = nn.ReLU6()

        self.pw_conv3 = PointWiseConv(expanded_channels, out_channels)
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # (B, Cin, H, W) -> (B, Cout, H/stride, W/stride)
        initial = x
        if self.expansion_factor != 1:
            x = self.relu1(self.bn1(self.pw_conv1(x)))
        x = self.relu2(self.bn2(self.dw_conv2(x)))
        x = self.bn3(self.pw_conv3(x))

        if self.res_connect:
            x += initial
        return x


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expansion_factor, repeat):
        super().__init__()
        layers = [
            InvertedResidualConv(in_channels, out_channels, stride, expansion_factor)
        ]
        for i in range(1, repeat):
            layers.append(
                InvertedResidualConv(out_channels, out_channels, 1, expansion_factor)
            )

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        # (B, Cin, H, W) -> (B, Cout, H/stride, W/stride)
        x = self.block(x)
        return x


class MobileNetV2(nn.Module):
    def __init__(self, num_class, alpha=1.0):
        super().__init__()
        if alpha <= 0:
            raise ValueError("width multiplier must be positive")
        self.num_class, self.alpha = num_class, alpha
        multiple = 8

        expansion_factors = [1, 6, 6, 6, 6, 6, 6]
        # somehow alpha doesn't apply to the 2nd and 3rd channel
        num_channels = [
            to_nearest_multiple_of(c * alpha, multiple)
            for c in (32, 16, 24, 32, 64, 96, 160, 320)
        ]
        final_channel = to_nearest_multiple_of(1280 * alpha, multiple)
        repeats = [1, 2, 3, 4, 3, 3, 1]
        strides = [1, 2, 2, 2, 1, 2, 1]
        assert (
            len(expansion_factors)
            == (len(num_channels) - 1)
            == len(repeats)
            == len(strides)
        )

        # expand number of channels
        self.initial = nn.Sequential(
            nn.Conv2d(
                3, num_channels[0], kernel_size=3, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(num_channels[0]),
            nn.ReLU6(),
        )

        # break down conv into inverted residual conv
        bottlenecks = []
        for expansion_factor, in_channels, out_channels, repeat, stride in zip(
            expansion_factors, num_channels[:-1], num_channels[1:], repeats, strides
        ):
            bottlenecks.append(
                Bottleneck(in_channels, out_channels, stride, expansion_factor, repeat)
            )
        self.bottlenecks = nn.Sequential(*bottlenecks)

        # classifier
        self.final = nn.Sequential(
            PointWiseConv(num_channels[-1], final_channel),
            nn.BatchNorm2d(final_channel),
            nn.ReLU6(),
            nn.AdaptiveAvgPool2d(1),  # for variable input size
            # nn.AvgPool2d(final_res),  # for same input size
            nn.Flatten(),
            nn.Dropout(p=0.001),
            nn.Linear(final_channel, num_class),
        )

    def forward(self, x):
        # (B, 3, H, W) -> (B, num_class)
        # no softmax implemented
        x = self.initial(x)
        x = self.bottlenecks(x)
        x = self.final(x)
        return x

    def __repr__(self):
        return f"MobileNetV2({self.num_class}, alpha={self.alpha})"


def test():
    from torchinfo import summary

    batch_size = 5
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = MobileNetV2(4).to(device)
    # summary(net, input_size=(batch_size, 3, 128, 128))

    # verify it works for variable input size
    y = net(torch.randn((batch_size, 3, 125, 250)).to(device))
    y = net(torch.randn((batch_size, 3, 320, 180)).to(device))

    # the minimum input size is 33x33 if the batch size = 1
    for image_size in ((33, 33), (32, 32)):
        try:
            y = net(torch.randn((1, 3, *image_size)).to(device))
            print(f"Model supports image size with {image_size}")
        except:
            print(f"Model does not support image size <= {image_size}")


if __name__ == "__main__":
    test()
