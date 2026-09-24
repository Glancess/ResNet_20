# ResNet-20 on CIFAR-10

<img width="306" height="196" alt="ResNet architecture" src="https://github.com/user-attachments/assets/b013a646-b994-4b7e-b084-23be1a74f84c" />

基于论文 [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)，使用 PyTorch 从零实现适用于 CIFAR-10 的 ResNet-20。

这个项目以理解网络结构和完整训练流程为主，代码拆分简单，适合学习：

- Residual Learning：`H(x) = F(x) + x`
- BasicBlock 的残差分支与 shortcut 分支
- Identity Shortcut 与 Projection Shortcut
- CIFAR ResNet 的 stage 设计
- Batch Normalization
- CIFAR-10 的训练、验证与测试流程
- TensorBoard 指标记录

## 模型结构

CIFAR-10 图像尺寸为 `3 × 32 × 32`。创建模型时使用：

```python
model = ResNetCIFAR(n=3, num_classes=10)
```

对于 CIFAR ResNet：

```text
depth = 6n + 2
```

当 `n = 3` 时：

```text
depth = 6 × 3 + 2 = 20
```

完整结构如下：

| 部分 | 结构 | 输出尺寸 |
| --- | --- | --- |
| Input | CIFAR-10 RGB 图像 | `3 × 32 × 32` |
| Stem | `3×3 Conv, 16` + BN + ReLU | `16 × 32 × 32` |
| Stage 1 | BasicBlock × 3，16 channels | `16 × 32 × 32` |
| Stage 2 | BasicBlock × 3，32 channels，第一个 block stride=2 | `32 × 16 × 16` |
| Stage 3 | BasicBlock × 3，64 channels，第一个 block stride=2 | `64 × 8 × 8` |
| Pool | Global Average Pooling | `64 × 1 × 1` |
| Classifier | Linear `64 → 10` | `10` |

## BasicBlock

每个 BasicBlock 的残差分支为：

```text
3×3 Conv → BN → ReLU → 3×3 Conv → BN
```

然后与 shortcut 相加：

```text
                         ┌──────── shortcut ────────┐
                         │                           │
x → Conv → BN → ReLU → Conv → BN → Add → ReLU → output
                         │             ↑
                         └─────────────┘
```

对应关系为：

```text
output = ReLU(F(x) + shortcut(x))
```

当前代码采用两种 shortcut：

- 输入、输出形状相同时：`nn.Identity()`
- channel 数变化或需要下采样时：`1×1 Conv + BatchNorm`

## 数据集划分

项目使用 torchvision 提供的 CIFAR-10：

| 数据集 | 样本数量 | 来源 |
| --- | ---: | --- |
| Train | 40,000 | 官方训练集前 40,000 张 |
| Validation | 10,000 | 官方训练集后 10,000 张 |
| Test | 10,000 | 官方测试集 |

图像会进行 `ToTensor()` 和 CIFAR-10 标准化：

```python
mean = (0.4914, 0.4822, 0.4465)
std = (0.2470, 0.2435, 0.2616)
```

数据默认下载到项目目录下的 `data/`。

## 训练配置

| 配置 | 当前值 |
| --- | --- |
| Random seed | `42` |
| Batch size | `128` |
| Epochs | `164` |
| Loss | CrossEntropyLoss |
| Optimizer | SGD |
| Initial learning rate | `0.1` |
| Momentum | `0.9` |
| Weight decay | `1e-4` |
| LR milestones | `82`, `123` |
| LR gamma | `0.1` |
| Metrics | Top-1、Top-5 Accuracy |

训练时优先使用 CUDA，其次使用 Apple MPS，最后使用 CPU。

## 项目结构

```text
ResNet_20/
├── Model/
│   └── ResNet_20.py       # BasicBlock 和 ResNetCIFAR
├── dataset/
│   ├── dataset.py         # CIFAR-10、数据划分和预处理
│   └── loader.py          # Train / Validation / Test DataLoader
├── engine/
│   ├── train.py           # 单个 epoch 的训练逻辑
│   └── evaluate.py        # 验证与测试逻辑
├── utils/
│   └── metric.py          # Top-k Accuracy
├── main.py                # 训练入口
└── README.md
```

## 环境安装

建议使用 Python 3.9 或更高版本。

```bash
pip install torch torchvision numpy tensorboard
```

## 开始训练

克隆仓库：

```bash
git clone https://github.com/Glancess/ResNet_20.git
cd ResNet_20
```

启动训练：

```bash
python main.py
```

程序会依次完成：

```text
训练集训练
    ↓
验证集评估
    ↓
保存验证集表现最好的模型
    ↓
训练结束后加载最佳模型
    ↓
在测试集上进行一次最终评估
```

最佳模型保存为：

```text
best_resnet20_cifar10.pth
```

## TensorBoard

训练过程记录以下指标：

- Train / Validation Loss
- Train / Validation Top-1 Accuracy
- Train / Validation Top-5 Accuracy
- Learning Rate

启动 TensorBoard：

```bash
tensorboard --logdir runs
```

然后在浏览器中打开：

```text
http://localhost:6006
```

本项目的日志目录为：

```text
runs/resnet20_cifar10
```

## 训练结果

当前仓库暂未提交完整训练结果。训练完成后可以将结果补充到这里：

| 指标 | 结果 |
| --- | ---: |
| Best Validation Top-1 | 待补充 |
| Final Test Top-1 | 待补充 |

## 当前实现说明

- 当前训练预处理只包含 `ToTensor()` 和 `Normalize()`，尚未加入随机裁剪或随机翻转等数据增强。
- 验证集和测试集不会参与参数更新。
- 测试集只在完整训练结束、重新加载最佳验证模型后评估一次。
- 该项目以清楚展示 ResNet-20 的结构和训练流程为主要目标。

## 参考资料

- Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385), CVPR 2016.
- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
