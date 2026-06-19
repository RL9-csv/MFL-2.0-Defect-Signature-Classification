"""BAR / LOT CSV 로딩 (NUL-byte safe)."""
import csv
import numpy as np
from .config import CHANNELS


def load_bar(path):
    """BAR CSV를 (T, 10) 진폭 배열로 로드. 파싱 실패 시 None.

    열 순서는 config.CHANNELS(원주순서)를 따름. 일부 원본에 섞인
    NUL 바이트는 제거 후 파싱함.
    """
    try:
        with open(path, errors="ignore") as f:
            text = f.read().replace("\x00", "")
        rows = list(csv.reader(text.splitlines()))
    except OSError:
        return None

    header_idx = next(
        (i for i, r in enumerate(rows) if r and r[0] == "No" and "CH-A1" in r),
        None,
    )
    if header_idx is None:
        return None

    cols = [rows[header_idx].index(c) for c in CHANNELS]
    data = []
    for r in rows[header_idx + 1:]:
        if len(r) <= max(cols):
            continue
        try:
            data.append([float(r[j]) for j in cols])
        except ValueError:
            continue
    return np.asarray(data) if data else None


def load_lot_meta(path):
    """LOT.CSV를 {필드명: 값} dict로 로드 (STEEL, SIZE, LINE SPEED 등)."""
    meta = {}
    try:
        with open(path, errors="ignore") as f:
            for line in f:
                parts = [x.strip() for x in line.split(",")]
                if len(parts) >= 2 and parts[0]:
                    meta.setdefault(parts[0], parts[1])
    except OSError:
        return {}
    return meta
