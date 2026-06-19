"""학습·평가 — LOT group split, class weight, soft-F1 혼합 손실."""
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, f1_score

from .model import DefectCNN, soft_f1_loss


def lot_group_split(groups, test_size=0.25, seed=42):
    """LOT(group) 단위 train/test 인덱스 분할 — 같은 LOT 누수 방지."""
    splitter = GroupShuffleSplit(1, test_size=test_size, random_state=seed)
    return next(splitter.split(np.zeros(len(groups)), groups=groups))


def _class_weights(y_train, n_classes, device):
    counts = np.bincount(y_train, minlength=n_classes)
    w = counts.sum() / (counts + 1) / n_classes
    return torch.tensor(w, dtype=torch.float32, device=device)


def train_model(X, y, groups, n_classes=3, epochs=50, batch=256,
                device=None, seed=42):
    """패치 텐서 X·라벨 y·그룹 groups로 CNN 학습. (model, report) 반환.

    report는 sklearn classification_report dict + 'macro_f1' 키 포함.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tr, te = lot_group_split(groups, seed=seed)
    weight = _class_weights(y[tr], n_classes, device)

    model = DefectCNN(in_channels=X.shape[1], n_classes=n_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)

    Xt = torch.tensor(X[tr], device=device)
    yt = torch.tensor(y[tr], dtype=torch.long, device=device)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), batch):
            b = perm[i:i + batch]
            opt.zero_grad()
            out = model(Xt[b])
            loss = (
                0.5 * F.cross_entropy(out, yt[b], weight=weight, label_smoothing=0.05)
                + soft_f1_loss(out, yt[b], n_classes)
            )
            loss.backward()
            opt.step()
        sched.step()

    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(X[te], device=device)).argmax(1).cpu().numpy()
    report = classification_report(y[te], pred, output_dict=True, zero_division=0)
    report["macro_f1"] = float(f1_score(y[te], pred, average="macro"))
    return model, report
