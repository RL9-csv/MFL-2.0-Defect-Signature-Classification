"""mfl.defects — component 추출·형태 라벨·패치 텐서화 테스트."""
import numpy as np

from mfl.defects import (
    extract_components,
    extract_patch,
    shape_label,
    structured_label,
    to_tensor,
)


def test_to_tensor_shape_and_mask():
    patch = np.zeros((64, 10))
    patch[10, 3] = 2.5                       # 임계 초과 셀 하나
    t = to_tensor(patch)
    assert t.shape == (4, 64, 10)
    assert t.dtype == np.float32
    assert t[1, 10, 3] == 1.0 and t[1].sum() == 1.0   # mask 채널


def test_extract_patch_fixed_length():
    bar = np.random.rand(507, 10)
    for center in (0, 250, 506):
        assert extract_patch(bar, center).shape == (64, 10)


def test_shape_label_staircase():
    coords = np.array([[i, i] for i in range(5)])     # 대각 진행
    assert shape_label(coords) == "staircase"


def test_shape_label_line():
    coords = np.array([[i, 2] for i in range(6)])      # 한 채널 길이방향
    assert shape_label(coords) == "line"


def test_shape_label_band():
    coords = np.array([[0, c] for c in range(4)])      # 같은 위치 다채널
    assert shape_label(coords) == "band"


def test_structured_isolated_point():
    assert structured_label(np.array([[5, 5]])) == "isolated"


def test_extract_components_merges_adjacent():
    bar = np.zeros((20, 10))
    bar[5, 5] = 3.0
    bar[6, 5] = 3.0                                    # 인접 → 한 component
    comps = extract_components(bar)
    assert len(comps) == 1 and len(comps[0]) == 2
