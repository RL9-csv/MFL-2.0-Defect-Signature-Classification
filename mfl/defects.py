"""결함 component 추출 · 형태 라벨링 · 패치 텐서화."""
import numpy as np
from scipy import ndimage

from .config import DEFECT_THRESHOLD, PATCH_LEN

_STRUCT = np.ones((3, 3))  # 8-connectivity (대각 인접 포함)


def extract_components(bar, threshold=DEFECT_THRESHOLD):
    """임계 초과 셀을 8-connectivity로 묶어 component 좌표 배열 리스트로 반환.

    각 원소는 (n, 2) 배열 = 한 결함 덩어리의 (row, channel) 좌표.
    """
    mask = bar > threshold
    if not mask.any():
        return []
    labels, n = ndimage.label(mask, structure=_STRUCT)
    return [np.argwhere(labels == i) for i in range(1, n + 1)]


def _orientation(coords):
    """component의 방향 기술: (row_span, n_channel, is_diagonal)."""
    rows, cols = coords[:, 0], coords[:, 1]
    row_span = int(rows.max() - rows.min() + 1)
    n_chan = int(len(np.unique(cols)))
    centers = [cols[rows == u].mean() for u in np.unique(rows)]
    mono = 0.0
    if len(centers) >= 2:
        diff = np.diff(centers)
        mono = max(np.mean(diff > 0), np.mean(diff < 0))  # 채널중심 단조 이동 비율
    diagonal = (
        row_span >= 3 and n_chan >= 3 and mono > 0.7
        and abs(centers[-1] - centers[0]) >= 2
    )
    return row_span, n_chan, diagonal


def _is_isolated(coords):
    return len(coords) <= 2 and len(np.unique(coords[:, 1])) <= 1


def structured_label(coords):
    """1차 라벨: 'structured' / 'isolated' / 'ambiguous' (weak label)."""
    if _is_isolated(coords):
        return "isolated"
    row_span, n_chan, diagonal = _orientation(coords)
    if diagonal or (n_chan >= 3 and row_span <= 2) or (row_span >= 4 and n_chan <= 2):
        return "structured"
    return "ambiguous"


def shape_label(coords):
    """2차 라벨: 'staircase' / 'band' / 'line', 해당 없으면 None (weak label)."""
    if _is_isolated(coords):
        return None
    row_span, n_chan, diagonal = _orientation(coords)
    if diagonal:
        return "staircase"          # 대각 진행 (회전+병진)
    if n_chan >= 3 and row_span <= 2:
        return "band"               # 원주방향 (같은 위치 다채널)
    if row_span >= 4 and n_chan <= 2:
        return "line"               # 길이방향 (한 채널 연속)
    return None


def extract_patch(bar, center, length=PATCH_LEN):
    """component 중심(center, 길이방향 index) 기준 (length, 10) 패치.

    경계는 replicate(edge) padding — 신호 단절을 막기 위함.
    """
    half = length // 2
    lo, hi = center - half, center + half
    pad_lo, pad_hi = max(0, -lo), max(0, hi - bar.shape[0])
    lo, hi = max(0, lo), min(bar.shape[0], hi)
    patch = bar[lo:hi]
    if pad_lo or pad_hi:
        patch = np.pad(patch, ((pad_lo, pad_hi), (0, 0)), mode="edge")
    if patch.shape[0] > length:
        patch = patch[:length]
    elif patch.shape[0] < length:
        patch = np.pad(patch, ((0, length - patch.shape[0]), (0, 0)), mode="edge")
    return patch


def to_tensor(patch, threshold=DEFECT_THRESHOLD):
    """패치 → (4, L, 10) 입력 텐서.

    채널: [log1p 진폭, 임계 mask, 길이방향 gradient, 채널방향 gradient].
    log1p는 결함 스파이크와 채널 간 상대크기를 보존하기 위한 선택.
    """
    amp = np.log1p(np.clip(patch, 0, None)).astype(np.float32)
    mask = (patch >= threshold).astype(np.float32)
    d_len = np.gradient(amp, axis=0).astype(np.float32)
    d_chan = np.gradient(amp, axis=1).astype(np.float32)
    return np.stack([amp, mask, d_len, d_chan], axis=0)
