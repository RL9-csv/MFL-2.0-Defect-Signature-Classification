"""저장된 데이터셋으로 CNN 학습·평가.

사용: python -m scripts.train_shape  (먼저 build_dataset 실행 필요)
"""
import numpy as np

from mfl.training import train_model

if __name__ == "__main__":
    X = np.load("X.npy")
    y = np.load("y.npy")
    groups = np.load("groups.npy")

    _, report = train_model(X, y, groups)

    print(f"macro-F1: {report['macro_f1']:.3f}")
    for cls, name in zip("012", ("staircase", "band", "line")):
        r = report[cls]
        print(f"  {name:10s} P={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1-score']:.3f}")
