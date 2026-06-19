"""중앙 설정 — 경로와 도메인 상수를 한 곳에 모음."""
import os

# 데이터 루트: 환경변수 MFL_DATA로 오버라이드, 없으면 상대경로.
# (하드코딩 절대경로를 쓰지 않아 재현·공유가 가능함)
DATA_ROOT = os.environ.get("MFL_DATA", os.path.join("data", "mlft_data"))

# 센서 10채널 — 결함 전이 분석으로 역추정한 '원주 순서' (A5–B1 인접 확인).
# 이 순서로 배열해야 staircase 결함이 끊기지 않고 대각선으로 보존됨.
CHANNELS = ["CH-A1", "CH-A2", "CH-A3", "CH-A4", "CH-A5",
            "CH-B1", "CH-B2", "CH-B3", "CH-B4", "CH-B5"]
N_CHANNELS = len(CHANNELS)

DEFECT_THRESHOLD = 1.5   # 장비 REJECT H 기준 전압 (V)
PATCH_LEN = 64           # 길이방향 패치 고정 길이
