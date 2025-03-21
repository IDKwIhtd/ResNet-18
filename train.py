import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import matplotlib.pyplot as plt
from model import CustomResNet, Block  # 모델 불러오기


def train():
    # device 설정 (M1/M2 Mac MPS 사용)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # 데이터 변환
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    # CIFAR-10 데이터셋 로드
    batch_size = 128
    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2
    )

    # 모델 및 하이퍼파라미터 설정
    model = CustomResNet(Block, [2, 2, 2, 2], num_classes=10).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 학습
    num_epochs = 10
    train_losses, train_accuracies = [], []

    for epoch in range(num_epochs):
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

        train_losses.append(running_loss / len(trainloader))
        train_accuracies.append(100 * correct / total)
        print(
            f"Epoch {epoch+1}: Loss = {train_losses[-1]:.4f}, Accuracy = {train_accuracies[-1]:.2f}%"
        )

    # 모델 가중치 저장
    torch.save(model.state_dict(), "resnet18.pth")

    # 손실 & 정확도 그래프 저장
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), train_losses, "b-o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")

    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), train_accuracies, "g-o")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training Accuracy")

    plt.savefig("loss_accuracy.png")  # 그래프 저장
    plt.show()


# main 문 추가 (macOS multiprocessing 문제 해결)
if __name__ == "__main__":
    train()
