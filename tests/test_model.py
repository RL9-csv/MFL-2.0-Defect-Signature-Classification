"""mfl.model — CNN forward·soft-F1 손실 테스트."""
import torch

from mfl.model import DefectCNN, soft_f1_loss


def test_forward_output_shape():
    model = DefectCNN(in_channels=4, n_classes=3)
    out = model(torch.randn(8, 4, 64, 10))
    assert out.shape == (8, 3)


def test_soft_f1_in_unit_range():
    logits = torch.randn(16, 3)
    target = torch.randint(0, 3, (16,))
    loss = soft_f1_loss(logits, target, 3)
    assert 0.0 <= loss.item() <= 1.0


def test_soft_f1_low_on_perfect_prediction():
    target = torch.tensor([0, 1, 2, 0, 1, 2])
    logits = torch.nn.functional.one_hot(target, 3).float() * 10.0
    assert soft_f1_loss(logits, target, 3).item() < 0.1
