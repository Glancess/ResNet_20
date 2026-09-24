<img width="306" height="196" alt="image" src="https://github.com/user-attachments/assets/b013a646-b994-4b7e-b084-23be1a74f84c" />
# ResNet-20 on CIFAR-10

基于论文 **Deep Residual Learning for Image Recognition**，从零实现 CIFAR-10 版本的 ResNet-20。

本项目主要用于理解 ResNet 的核心设计，包括：

- Residual Learning：\(H(x)=F(x)+x\)
- BasicBlock
- Identity / Projection Shortcut
- Stage 设计
- Batch Normalization
- CIFAR-10 上的 ResNet-20 结构

## Model

CIFAR-10 输入尺寸为 `32×32`，网络包含 3 个 residual stages：

```text
Input: 3×32×32

Stem
3×3 Conv, 16

Stage 1
BasicBlock × 3
16 channels, 32×32

Stage 2
BasicBlock × 3
32 channels, 16×16

Stage 3
BasicBlock × 3
64 channels, 8×8

Global Average Pooling
↓
FC
↓
10 classes
