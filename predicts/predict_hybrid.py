import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, ConcatDataset, DataLoader

from models.hybrid import CustomResNetWithTransformer, Block

# 디바이스 설정
device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available() else "cpu"
)
print(f"Using device: {device}")

# 결과 디렉토리 생성
os.makedirs("predicts", exist_ok=True)


# 정규화 해제 함수
def unnormalize(img):
    img = img * 0.5 + 0.5
    return np.clip(img, 0, 1)


# 리스트 → Dataset 형태로 감싸기 위한 클래스
class SimpleListDataset(Dataset):
    def __init__(self, data):
        self.data = data  # [(img, label), ...]

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


# 데이터 변환 정의 (CIFAR-100 기준)
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

# CIFAR-100 클래스 기준
cifar100_base = torchvision.datasets.CIFAR100(root="./data", train=False)
class_name_to_idx = {name: i for i, name in enumerate(cifar100_base.classes)}
classes = cifar100_base.classes
num_classes = len(classes)

# CIFAR-10 → CIFAR-100 기준 라벨로 매핑
cifar10_test = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform
)
mapped_cifar10_test = [
    (img, class_name_to_idx[cifar10_test.classes[label]])
    for img, label in cifar10_test
    if cifar10_test.classes[label] in class_name_to_idx
]
mapped_cifar10_test_ds = SimpleListDataset(mapped_cifar10_test)

# CIFAR-100 테스트셋
cifar100_test = torchvision.datasets.CIFAR100(
    root="./data", train=False, download=True, transform=transform
)

# 병합된 테스트셋
testset = ConcatDataset([mapped_cifar10_test_ds, cifar100_test])
testloader = DataLoader(testset, batch_size=5, shuffle=False)

# 모델 로드
checkpoint = torch.load("checkpoint_hybrid.pth", map_location=device)
model = CustomResNetWithTransformer(Block, [2, 2, 2, 2], num_classes=num_classes).to(
    device
)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# 무작위로 이미지 5장 선택 및 예측
num_samples = 5
indices = torch.randperm(len(testset))[:num_samples]
images = torch.stack([testset[i][0] for i in indices]).to(device)
labels = torch.tensor([testset[i][1] for i in indices]).to(device)

with torch.no_grad():
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)

# 시각화
fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
for i in range(num_samples):
    img = images[i].cpu().numpy().transpose(1, 2, 0)
    img = unnormalize(img)
    axes[i].imshow(img)
    axes[i].set_title(f"GT: {classes[labels[i]]}\nPred: {classes[predicted[i]]}")
    axes[i].axis("off")

plt.tight_layout()
plt.savefig("predicts/predictions_hybrid.png")
plt.show()

# 전체 정확도 측정
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
print(f"\nTest Accuracy on merged dataset: {accuracy:.2f}%")
