import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from model import CustomResNet, Block  # 모델 불러오기

# device 설정 (MPS 사용 가능하면 MPS, 그렇지 않으면 CPU 사용)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# 모델 로드
model = CustomResNet(Block, [2, 2, 2, 2], num_classes=10).to(device)
model.load_state_dict(torch.load("resnet18.pth"))
model.eval()

# 데이터 변환
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

# CIFAR-10 데이터 로드
testset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform
)
testloader = torch.utils.data.DataLoader(testset, batch_size=5, shuffle=False)


# 정규화 해제 함수 정의
def unnormalize(img):
    """정규화된 이미지를 원래 범위로 변환"""
    img = img * 0.5 + 0.5  # (x * std + mean)
    return np.clip(img, 0, 1)  # 값 범위를 [0,1]로 제한


# 테스트셋에서 무작위 이미지 5장 추출
num_samples = 5
indices = torch.randperm(len(testset))[:num_samples]

# 선택된 인덱스로부터 이미지와 라벨 로드
images = torch.stack([testset[i][0] for i in indices]).to(device)
labels = torch.tensor([testset[i][1] for i in indices]).to(device)

# 모델 예측
with torch.no_grad():
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)

# 시각화
fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
classes = testset.classes

for i in range(num_samples):
    img = images[i].cpu().numpy().transpose(1, 2, 0)
    img = unnormalize(img)
    axes[i].imshow(img)
    axes[i].set_title(f"GT: {classes[labels[i]]}\nPred: {classes[predicted[i]]}")
    axes[i].axis("off")

plt.savefig("predictions.png")
plt.show()


# 전체 테스트셋 정확도 측정
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
print(f"Test Accuracy: {accuracy:.2f}%")
