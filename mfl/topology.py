"""센서 채널 인접관계 역추정 — 결함 전이 빈도로 물리 배치를 복원."""
import numpy as np

from .config import N_CHANNELS
from .defects import shape_label


def sensor_transition_matrix(components):
    """staircase component들의 채널 이동을 (10, 10) 전이행렬로 집계.

    T[i, j] = 결함이 채널 i 다음 행에서 채널 j로 이동한 횟수.
    이 행렬의 최빈 인접쌍이 센서의 물리 배치(원주 순서)를 가리킴.
    """
    T = np.zeros((N_CHANNELS, N_CHANNELS))
    for coords in components:
        if shape_label(coords) != "staircase":
            continue
        rows, cols = coords[:, 0], coords[:, 1]
        seq = [int(round(cols[rows == u].mean())) for u in np.unique(rows)]
        for a, b in zip(seq, seq[1:]):
            if 0 <= a < N_CHANNELS and 0 <= b < N_CHANNELS:
                T[a, b] += 1
    return T


def top_adjacent_pairs(T, k=8):
    """전이행렬에서 양방향 합 기준 최빈 인접쌍 top-k를 (count, i, j)로 반환."""
    pairs = {}
    for i in range(N_CHANNELS):
        for j in range(N_CHANNELS):
            if i < j:
                pairs[(i, j)] = T[i, j] + T[j, i]
    ranked = sorted(pairs.items(), key=lambda kv: kv[1], reverse=True)
    return [(int(c), i, j) for (i, j), c in ranked[:k] if c > 0]
