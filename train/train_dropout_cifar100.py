import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from models.resnet18_dropout import (
    CustomResNet,
    Block,
)
from utils.early_stopping import EarlyStopping


def train():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # 데이터 변환 정의
    transform_train = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    transform_test = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    # 데이터셋 로드
    batch_size = 128
    trainset = torchvision.datasets.CIFAR100(
        root="./data", train=True, download=True, transform=transform_train
    )
    num_classes = len(trainset.classes)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2
    )

    testset = torchvision.datasets.CIFAR100(
        root="./data", train=False, download=True, transform=transform_test
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2
    )

    # 모델 및 학습 설정
    model = CustomResNet(Block, [2, 2, 2, 2], num_classes=num_classes).to(device)
    if os.path.exists("checkpoint.pth"):
        state_dict = torch.load("checkpoint.pth")
        state_dict.pop("fc.weight", None)
        state_dict.pop("fc.bias", None)
        model.load_state_dict(state_dict, strict=False)
        print("CIFAR-10 모델에서 feature extractor만 가져옴")

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    early_stopping = EarlyStopping(
        patience=5, verbose=True, path="checkpoint._cifar100.pth"
    )

    num_epochs = 10
    train_losses, train_accuracies, val_losses = [], [], []

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

        # 검증 손실 계산
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

        print(
            f"[Epoch {epoch+1}] Train Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%, Val Loss: {val_loss:.4f}"
        )
        # 조기 종료 체크
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break
    # 가장 좋은 모델 불러오기
    model.load_state_dict(torch.load("checkpoint._cifar100.pth"))

    # 손실 및 정확도 그래프 저장
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
    plt.savefig("loss_accuracy_dropout_cifar100.png")
    plt.show()


if __name__ == "__main__":
    train()
