import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # residual branch：负责学习 F(x)
        self.residual_branch = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        )

        if stride == 1 and in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        identity = self.shortcut(x)

        out = self.residual_branch(x)

        out = out + identity
        out = self.relu(out)

        return out


class ResNetCIFAR(nn.Module):
    def __init__(self, n=3, num_classes=10):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )

        self.stage1 = self._make_stage(16, 16, n, stride=1)
        self.stage2 = self._make_stage(16, 32, n, stride=2)
        self.stage3 = self._make_stage(32, 64, n, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def _make_stage(self, in_channels, out_channels, num_blocks, stride):
        blocks = []
        # 新 stage 的第一个 block
        blocks.append(BasicBlock(in_channels, out_channels, stride=stride))

        # 同一个 stage 中剩下的 block
        for _ in range(1, num_blocks):
            blocks.append(BasicBlock(out_channels, out_channels, stride=1))

        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.stem(x)
        # print(x.shape)
        x = self.stage1(x)
        # print(x.shape)
        x = self.stage2(x)
        # print(x.shape)
        x = self.stage3(x)
        # print(x.shape)

        x = self.avgpool(x)
        # print(x.shape)
        x = torch.flatten(x, 1)
        # print(x.shape)
        x = self.fc(x)
        # print(x.shape)
        return x
