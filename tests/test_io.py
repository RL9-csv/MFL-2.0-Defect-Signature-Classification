"""mfl.io — CSV 파싱·NUL 안전성 테스트 (실데이터 불필요)."""
from mfl.io import load_bar

BAR_CSV = (
    "DATE,2018.02.21 10:56\n"
    "Lot No,L820169\n"
    "Bar No,1\n"
    "Result,NO GOOD\n"
    "Defect H,9\n"
    "Defect L,8\n"
    "No,Point,A-OR,B-OR,CH-A1,CH-A2,CH-A3,CH-A4,CH-A5,CH-B1,CH-B2,CH-B3,CH-B4,CH-B5,Event H,Event L\n"
    "1,0,0.10,0.62,0.11,0.10,0.06,0.07,0.08,0.22,0.13,0.23,0.16,0.62,0,0\n"
    "2,13,0.21,0.79,0.06,0.21,0.12,0.10,0.14,0.11,0.15,0.79,0.17,0.11,0,0\n"
)


def test_load_bar_shape_and_order(tmp_path):
    p = tmp_path / "BAR00001.CSV"
    p.write_text(BAR_CSV)
    bar = load_bar(str(p))
    assert bar.shape == (2, 10)
    assert bar[0, 0] == 0.11          # CH-A1 첫 행
    assert bar[1, 9] == 0.11          # CH-B5 둘째 행


def test_load_bar_nul_safe(tmp_path):
    p = tmp_path / "b.CSV"
    p.write_bytes(BAR_CSV.encode() + b"\x00\x00\x00")
    assert load_bar(str(p)).shape == (2, 10)


def test_load_bar_invalid_returns_none(tmp_path):
    p = tmp_path / "x.CSV"
    p.write_text("garbage\nno header here\n")
    assert load_bar(str(p)) is None
