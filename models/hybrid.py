import torch
import torch.nn as nn
import torch.nn.functional as F


# Residual Block 정의
class Block(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        """
        Residual Block (잔차 블록)
        - 두 개의 3x3 컨볼루션 레이어(conv1, conv2)와 Batch Normalization을 포함
        - 입력이 shortcut을 통해 출력에 더해지는 구조

        Args:
        - in_ch (int): 입력 채널 수
        - out_ch (int): 출력 채널 수
        - stride (int, optional): 첫 번째 컨볼루션의 stride (기본값: 1)
        """
        super(Block, self).__init__()
        # 첫 번째 3x3 컨볼루션 레이어 (stride 적용 가능)
        self.conv1 = nn.Conv2d(
            in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_ch)  # 배치 정규화
        # 두 번째 3x3 컨볼루션 레이어 (채널 수 유지, stride=1)
        self.conv2 = nn.Conv2d(
            out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_ch)  # 배치 정규화

        # 입력과 출력 채녈이 다를 경우 1x1 Conv를 사용하여 차원 맞추기
        self.skip_connection = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.skip_connection = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),  # 배치 정규화
            )

    def forward(self, x):
        """
        Forward 함수 (잔차 학습 적용)
        - 입력 데이터 (x)를 컨볼루션과 배치 정규화를 거쳐 변환
        - Shortcut Connection을 통해 원래 입력(x)을 출력에 더함
        """
        # 첫 번째 컨볼루션 + ReLU 활성화 함수
        output = F.relu(self.bn1(self.conv1(x)))
        # 두 번째 컨볼루션 후 배치 정규화
        output = self.bn2(self.conv2(output))
        # shortcut 경로 출력과 현재 블록의 출력 더하기
        output += self.skip_connection(x)  # Residual 연결
        # 최종 ReLU 활성화 함수 적용
        output = F.relu(output)
        return output


# ResNet 모델 정의
class CustomResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        """
        Custom ResNet 모델
        - 첫 번째 컨볼루션 레이어와 배치 정규화 적용
        - Residual Block을 쌓아 네트워크 구성
        - 최종적으로 Fully Connected Layer를 통해 분류 수행

        Args:
        - block (nn.Module): Residual Block 클래스
        - layers (list): 각 단계에서 사용할 Residual Block의 개수
        - num_classes (int, optional): 최종 분류할 클래스 개수 (기본값: 10)
        """
        super(CustomResNet, self).__init__()

        self.initial_channels = 64  # 첫 번째 레이어의 입력 채널 수 정의

        # 첫 번째 컨볼루션 레이어 (입력: 3채널 이미지)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)  # 배치 정규화

        # ResNet의 각 레이어 생성 (Residual Block을 여러 개 쌓음)
        self.layer1 = self._create_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._create_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._create_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._create_layer(block, 512, layers[3], stride=2)

        # 평균 풀링 레이어 (AdaptiveAvgPool2d: 입력 크기에 관계없이 1x1로 변환)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # 드롭아웃 추가
        self.dropout = nn.Dropout(p=0.5)
        # 최종 완전 연결 레이어 (출력: num_classes)
        self.fc = nn.Linear(512, num_classes)

    # ResNet의 각 레이어를 생성하는 함수
    def _create_layer(self, block, out_ch, num_layers, stride):
        """
        Residual Block을 여러 개 쌓아 하나의 레이어를 구성하는 함수

        Args:
        - block (nn.Module): Residual Block 클래스
        - out_ch (int): 출력 채널 수
        - num_layers (int): 해당 레이어에서 사용할 Residual Block 개수
        - stride (int): 첫 번째 블록에서 적용할 stride 값

        Returns:
        - nn.Sequential: 구성된 Residual Layer
        """
        layers = []
        # 첫 번째 블록은 stride를 받을 수 있음
        layers.append(block(self.initial_channels, out_ch, stride))
        self.initial_channels = out_ch  # 다음 불록을 위해 채널 수 업데이트

        # 나머지 블록들은 기본 stride를 사용
        for _ in range(1, num_layers):
            layers.append(block(out_ch, out_ch))

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward 함수 (ResNet 모델의 순전파 과정)
        - 첫 번째 컨볼루션을 거친 후, 여러 개의 Residual Block을 순차적으로 통과
        - 마지막으로 Fully Connected Layer를 거쳐 최종 클래스 확률을 출력
        """
        # 첫 번째 컨볼루션 + ReLU 활성화 함수
        x = F.relu(self.bn1(self.conv1(x)))
        # 각 레이어를 순차적으로 통과
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Transformer 입력 준비
        B, C, H, W = x.shape
        x = x.view(B, C, H * W).permute(0, 2, 1)  # (B, 16, 512)

        # Transformer block 넣기
        x = self.transformer(x)  # (B, 16, 512)

        # Sequence 평균 → classification head로
        x = x.mean(dim=1)  # (B, 512)
        x = self.dropout(x)  # 드롭아웃 적용
        # 최종 완전 연결 레이어를 통해 클래스별 예측값 출력
        x = self.fc(x)
        return x


class SimpleTransformerBlock(nn.Module):
    def __init__(self, dim=512, heads=8, ff_dim=2048, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, dim),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        attn_output, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_output)
        ff_output = self.ff(x)
        x = self.norm2(x + ff_output)
        return x


class CustomResNetWithTransformer(CustomResNet):
    def __init__(self, block, layers, num_classes=10, dropout_prob=0.5):
        super().__init__(block, layers, num_classes)
        self.transformer = SimpleTransformerBlock(dim=512)
        self.cls_head = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, num_classes))
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # ResNet output: (B, 512, 4, 4)
        B, C, H, W = x.shape
        x = x.view(B, C, H * W).permute(0, 2, 1)  # (B, 16, 512)

        x = self.transformer(x)  # → (B, 16, 512)
        x = x.mean(dim=1)  # → (B, 512)
        x = self.dropout(x)
        x = self.cls_head(x)  # → (B, num_classes)

        return x


# ResNet-18 모델 생성 (각 레이어의 블록 수 : [2, 2, 2, 2])
# model = CustomResNet(Block, [2, 2, 2, 2], num_classes=10)

"""
[Conv1 + BN + ReLU]
→ layer1
→ layer2
→ layer3
→ layer4  ← 여기까지 CNN (output shape: B x 512 x 4 x 4)

→ reshape + permute → B x 16 x 512
→ Transformer block
→ mean(dim=1) → B x 512
→ Dropout
→ LayerNorm + FC
→ 출력
"""
