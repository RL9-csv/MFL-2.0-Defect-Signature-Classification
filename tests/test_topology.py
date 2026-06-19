"""mfl.topology — 센서 전이행렬 역추정 테스트."""
import numpy as np

from mfl.topology import sensor_transition_matrix, top_adjacent_pairs


def test_transition_counts_diagonal_moves():
    coords = np.array([[i, i] for i in range(5)])      # staircase 0→1→2→3→4
    T = sensor_transition_matrix([coords])
    assert T[0, 1] == 1
    assert T[3, 4] == 1


def test_top_adjacent_pairs_nonempty():
    coords = np.array([[i, i] for i in range(5)])
    pairs = top_adjacent_pairs(sensor_transition_matrix([coords]), k=4)
    assert pairs and all(len(p) == 3 for p in pairs)
