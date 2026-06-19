"""2D CNN 분류기 + soft-F1 손실."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DefectCNN(nn.Module):
    """결함 패치 (B, C, L, 10) → 클래스 로짓.

    Conv–BN–GELU 3블록. replicate padding으로 경계의 공간 정보를 보존.
    """

    def __init__(self, in_channels=4, n_classes=3, widths=(48, 96, 192)):
        super().__init__()

        def block(i, o):
            return nn.Sequential(
                nn.Conv2d(i, o, 3, padding=1, padding_mode="replicate"),
                nn.BatchNorm2d(o),
                nn.GELU(),
            )

        w1, w2, w3 = widths
        self.b1 = block(in_channels, w1)
        self.b2 = block(w1, w2)
        self.b3 = block(w2, w3)
        self.pool = nn.MaxPool2d(2)
        self.drop = nn.Dropout(0.25)
        self.fc = nn.Linear(w3, n_classes)

    def forward(self, x):
        x = self.pool(self.b1(x))
        x = self.pool(self.b2(x))
        x = self.b3(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.fc(self.drop(x))


def soft_f1_loss(logits, target, n_classes, eps=1e-6):
    """미분가능 macro soft-F1 손실 = 1 − mean_c(soft-F1_c).

    F1은 argmax(이산)이라 미분 불가 → 확률 p로 TP/FP/FN을 연속화해
    평가지표(macro-F1)를 직접 최적화함.
    """
    p = F.softmax(logits, dim=1)
    y = F.one_hot(target, n_classes).float()
    tp = (p * y).sum(0)
    fp = (p * (1 - y)).sum(0)
    fn = ((1 - p) * y).sum(0)
    f1 = 2 * tp / (2 * tp + fp + fn + eps)
    return 1 - f1.mean()
