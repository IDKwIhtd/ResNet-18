import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import ConcatDataset
import matplotlib.pyplot as plt
from torchvision.datasets import CIFAR100

from models.hybrid import CustomResNetWithTransformer, Block
from utils.early_stopping import EarlyStopping


def get_merged_dataset():
    transform_train = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    # CIFAR-10 + CIFAR-100 (train)
    cifar10_train = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_train
    )
    cifar100_train = torchvision.datasets.CIFAR100(
        root="./data", train=True, download=True, transform=transform_train
    )

    # CIFAR-10 + CIFAR-100 (test)
    cifar10_test = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_test
    )
    cifar100_test = torchvision.datasets.CIFAR100(
        root="./data", train=False, download=True, transform=transform_test
    )

    # CIFAR-100 기준 클래스 이름 → 라벨 딕셔너리
    cifar100_class_to_idx = {
        name: idx for idx, name in enumerate(cifar100_train.classes)
    }
    common_classes = set(cifar10_train.classes).intersection(set(cifar100_class_to_idx))

    # CIFAR-10 → CIFAR-100 라벨로 변환 (train)
    mapped_cifar10_train = [
        (img, cifar100_class_to_idx[cifar10_train.classes[label]])
        for img, label in cifar10_train
        if cifar10_train.classes[label] in common_classes
    ]

    # CIFAR-10 → CIFAR-100 라벨로 변환 (test)
    mapped_cifar10_test = [
        (img, cifar100_class_to_idx[cifar10_test.classes[label]])
        for img, label in cifar10_test
        if cifar10_test.classes[label] in common_classes
    ]

    # 병합된 학습/테스트셋 생성
    trainset = mapped_cifar10_train + list(cifar100_train)
    testset = mapped_cifar10_test + list(cifar100_test)

    return trainset, testset


def train():
    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    train_data, testset = get_merged_dataset()

    trainloader = torch.utils.data.DataLoader(
        train_data, batch_size=128, shuffle=True, num_workers=2
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2
    )

    num_classes = len(CIFAR100(root="./data", train=False).classes)

    model = CustomResNetWithTransformer(
        Block, [2, 2, 2, 2], num_classes=num_classes, dropout_prob=0.7
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, verbose=True
    )  # mode: val_loss가 최소화되는 방향으로 조정, factor: lr을 반으로 줄임, patience: val_loss가 2epoch 동안 줄지 않으면 적용, verbose: 줄어들 때마다 출력
    checkpoint_path = "checkpoint_hybrid.pth"
    best_val_loss = float("inf")
    early_stopping = EarlyStopping(
        patience=5, verbose=True, path="checkpoint_hybrid.pth"
    )

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print("체크포인트에서 이어서 학습 시작!")

    # if os.path.exists(checkpoint_path):
    #     checkpoint = torch.load(checkpoint_path, map_location=device)
    #     if "model_state_dict" in checkpoint:
    #         model.load_state_dict(checkpoint["model_state_dict"])
    #         optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    #         print("모델과 옵티마이저 상태 불러옴!")
    #     else:
    #         # 예전 방식: state_dict만 저장되어 있었던 경우
    #         model.load_state_dict(checkpoint)
    #         print("model_state_dict 키 없음 → 모델 state_dict만 불러옴 (이전 방식)")

    train_losses, train_accuracies, val_losses = [], [], []
    num_epochs = 10

    for epoch in range(num_epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / len(trainloader)
        train_acc = 100 * correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        val_loss /= len(testloader)
        val_losses.append(val_loss)

        scheduler.step(val_loss)

        print(
            f"[Epoch {epoch+1}] Train Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%, Val Loss: {val_loss:.4f}"
        )

        early_stopping(val_loss, model)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                checkpoint_path,
            )
            print(f"Val Loss 개선됨. 체크포인트 저장됨 → {checkpoint_path}")

        if early_stopping.early_stop:
            print("Early stopping. Best model 복원 중...")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            break

    # 그래프 저장
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.plot(range(1, len(train_losses) + 1), train_losses, "b-o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")

    plt.subplot(1, 3, 2)
    plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, "g-o")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training Accuracy")

    plt.subplot(1, 3, 3)
    plt.plot(range(1, len(val_losses) + 1), val_losses, "r-o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Validation Loss")

    plt.tight_layout()
    plt.savefig("loss_accuracy_hybrid.png")
    plt.show()


if __name__ == "__main__":
    train()
