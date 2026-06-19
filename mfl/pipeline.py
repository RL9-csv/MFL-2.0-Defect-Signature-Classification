"""원본 디렉토리 → 패치 데이터셋 조립 (end-to-end 전처리)."""
import glob
import os

import numpy as np

from .config import DATA_ROOT
from .defects import extract_components, extract_patch, shape_label, to_tensor
from .io import load_bar

SHAPE_CLASSES = {"staircase": 0, "band": 1, "line": 2}


def iter_bar_paths(data_root=DATA_ROOT, n_days=None):
    """날짜/LOT 폴더를 순회하며 (lot_id, bar_path)를 yield."""
    dates = sorted(d for d in os.listdir(data_root) if d.isdigit())
    if n_days:
        dates = dates[:n_days]
    for d in dates:
        day = os.path.join(data_root, d)
        for lot in sorted(os.listdir(day)):
            lot_dir = os.path.join(day, lot)
            if lot.startswith("L") and os.path.isdir(lot_dir):
                for bp in sorted(glob.glob(os.path.join(lot_dir, "BAR*.CSV"))):
                    yield lot, bp


def build_shape_dataset(data_root=DATA_ROOT, n_days=None, cap_staircase=8000, seed=42):
    """형태 3-class 패치 데이터셋 (X, y, groups) 생성.

    X: (N, 4, 64, 10) 패치 텐서, y: 0=staircase/1=band/2=line,
    groups: LOT id (group split용). 다수 클래스(staircase)는 cap으로 상한.
    """
    rng = np.random.RandomState(seed)
    buckets = {0: [], 1: [], 2: []}
    for lot, bar_path in iter_bar_paths(data_root, n_days):
        bar = load_bar(bar_path)
        if bar is None:
            continue
        for coords in extract_components(bar):
            label = shape_label(coords)
            if label is None:
                continue
            cls = SHAPE_CLASSES[label]
            center = int(round(coords[:, 0].mean()))
            buckets[cls].append((to_tensor(extract_patch(bar, center)), cls, lot))

    if cap_staircase:
        rng.shuffle(buckets[0])
        buckets[0] = buckets[0][:cap_staircase]

    data = buckets[0] + buckets[1] + buckets[2]
    rng.shuffle(data)
    X = np.stack([d[0] for d in data])
    y = np.array([d[1] for d in data])
    groups = np.array([d[2] for d in data])
    return X, y, groups
