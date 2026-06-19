"""형태 데이터셋 생성 후 .npy로 저장.

사용: python -m scripts.build_dataset
환경변수 MFL_DATA로 데이터 루트 지정 (기본: data/mlft_data).
"""
import numpy as np

from mfl.pipeline import build_shape_dataset

if __name__ == "__main__":
    X, y, groups = build_shape_dataset()
    np.save("X.npy", X)
    np.save("y.npy", y)
    np.save("groups.npy", groups)
    print(f"saved X{X.shape}  class counts={np.bincount(y).tolist()}  "
          f"LOTs={len(set(groups))}")
