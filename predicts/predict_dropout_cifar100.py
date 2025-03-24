import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from models.resnet18_dropout import CustomResNet, Block  # CIFAR-100 학습된 모델

# 디바이스 설정
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# CIFAR-100 클래스 수
num_classes = 100

# 모델 정의 및 학습된 파라미터 불러오기
model = CustomResNet(Block, [2, 2, 2, 2], num_classes=num_classes).to(device)
model.load_state_dict(torch.load("checkpoint._cifar100.pth"))
model.eval()

# 이미지 변환 정의
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

# CIFAR-100 테스트셋 불러오기
testset = torchvision.datasets.CIFAR100(
    root="./data", train=False, download=True, transform=transform
)
testloader = torch.utils.data.DataLoader(testset, batch_size=5, shuffle=False)


# 정규화 해제 함수
def unnormalize(img):
    img = img * 0.5 + 0.5
    return np.clip(img, 0, 1)


# 무작위 샘플 예측 및 시각화
num_samples = 5
indices = torch.randperm(len(testset))[:num_samples]

images = torch.stack([testset[i][0] for i in indices]).to(device)
labels = torch.tensor([testset[i][1] for i in indices]).to(device)

with torch.no_grad():
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)

fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
classes = testset.classes

for i in range(num_samples):
    img = images[i].cpu().numpy().transpose(1, 2, 0)
    img = unnormalize(img)
    axes[i].imshow(img)
    axes[i].set_title(f"GT: {classes[labels[i]]}\nPred: {classes[predicted[i]]}")
    axes[i].axis("off")

plt.savefig("predicts/predictions_dropout_cifar100.png")
plt.show()

# 전체 테스트 정확도 측정
correct = 0
total = 0

with torch.no_grad():
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"CIFAR-100 Test Accuracy: {accuracy:.2f}%")
