import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


from engine.train import train_one_epoch
from engine.evaluate import evaluate

from Model.ResNet_20 import ResNetCIFAR

from torch.utils.tensorboard import SummaryWriter

from dataset.loader import get_loaders

# =========================
# config
# =========================

SEED = 42
BATCH_SIZE = 128
NUM_WORKERS = 4

EPOCHS = 164
LR = 0.1
WEIGHT_DECAY = 1e-4
MOMENTUM = 0.9
TOP_K = (1, 5)

# =========================
# seed
# =========================


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =========================
# main
# =========================


def main():
    set_seed(SEED)

    # ---------- device ----------
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("device:", device)

    # ---------- dataset ----------
    train_loader, val_loader, test_loader = get_loaders(
        batch_size=BATCH_SIZE,
    )

    # ---------- model ----------
    # n=3 -> CIFAR ResNet-20
    model = ResNetCIFAR(n=3, num_classes=10).to(device)

    print(model)

    # ---------- sanity check ----------
    x = torch.randn(2, 3, 32, 32).to(device)

    with torch.no_grad():
        y = model(x)

    print("input shape :", x.shape)
    print("output shape:", y.shape)

    # 应该：
    # input  -> [2, 3, 32, 32]
    # output -> [2, 10]

    # ---------- loss ----------
    criterion = nn.CrossEntropyLoss()

    # ---------- optimizer ----------
    optimizer = torch.optim.SGD(
        model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY
    )

    # ---------- scheduler ----------
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[82, 123], gamma=0.1
    )

    # ---------- tensorboard ----------
    writer = SummaryWriter(log_dir="runs/resnet20_cifar10")

    best_val_acc = 0.0

    # =========================
    # train
    # =========================

    for epoch in range(EPOCHS):

        print(f"\n========== Epoch " f"{epoch + 1}/{EPOCHS} ==========")

        # ---------- train ----------

        (
            train_loss,
            train_top1,
            train_top5,
        ) = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            topk=TOP_K,
        )

        # ---------- validation ----------
        val_loss, val_top1, val_top5 = evaluate(
            model=model,
            data_loader=val_loader,
            criterion=criterion,
            device=device,
            topk=TOP_K,
        )

        # 这里根据你 engine 返回格式改

        # ---------- tensorboard ----------
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)

        writer.add_scalar("Accuracy/train_top1", train_top1, epoch)
        writer.add_scalar("Accuracy/val_top1", val_top1, epoch)

        writer.add_scalar("Accuracy/train_top5", train_top5, epoch)
        writer.add_scalar("Accuracy/val_top5", val_top5, epoch)

        writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)

        print(f"Train Loss: {train_loss:.4f} | " f"Train Acc: {train_top1:.2f}%")

        print(f"Val Loss: {val_loss:.4f} | " f"Val Acc: {val_top1:.2f}%")

        # ---------- save best ----------
        if val_top1 > best_val_acc:
            best_val_acc = val_top1

            torch.save(model.state_dict(), "best_resnet20_cifar10.pth")

            print(f"Best model saved: " f"{best_val_acc:.2f}%")

        scheduler.step()

    # =========================
    # final test
    # =========================

    model.load_state_dict(torch.load("best_resnet20_cifar10.pth", map_location=device))

    test_loss, test_top1, _ = evaluate(
        model=model,
        data_loader=test_loader,
        criterion=criterion,
        device=device,
        topk=TOP_K,
    )

    print("\n========== Final Test ==========")
    print(f"Test Loss: {test_loss:.4f} | " f"Test Acc: {test_top1:.2f}%")

    writer.close()


if __name__ == "__main__":
    main()
