# MFL Defect Signature Classification
**누설자속탐상(MFL) 결함 신호의 공간 패턴 분류 — 장비 임계룰 너머의 failure signature 식별**

장비가 진폭 임계로 "결함 유무"를 판정한 *그 다음 단계*를 다룬다. 결함 스파이크의 **공간 signature(형태·방향)** 를 분류해, 단순 양·불 판정이 아닌 failure mode 후보를 식별함. 반도체 Product Engineering의 wafer/bin map failure-pattern 분석과 구조적으로 유사한 문제로 정의함.

---

## 1. 문제 정의

MFL 검사 장비는 진폭 임계(REJECT H: 1.5V / 26mm)로 결함 이벤트를 자가판정함. 따라서 **"결함 유무" 분류는 임계 한 줄(trivial)** 이라 ML로 풀 가치가 낮음. 문제를 두 단계로 재정의함:

- **1차** — 임계를 넘은 결함 후보(component) 중 **구조적 결함 후보(structured) vs 고립 노이즈성 artifact(isolated)** 분리
- **2차** — 구조적 결함의 **형태 분류**: `staircase`(대각 진행) / `band`(원주방향) / `line`(길이방향). 형태가 결함 원인(root cause)의 단서가 됨

핵심: structured와 isolated, 그리고 세 형태는 **진폭이 모두 임계를 넘음** → 진폭(임계)으로는 못 가름. 오직 **공간 패턴**으로만 갈림 → 2D CNN의 필요성이 데이터로 성립함.

## 2. 데이터와 신호 형상

- 강철바 MFL 비파괴검사. 86일 / 878 LOT / 약 85,000 bar
- 각 bar = 길이방향 스캔(약 507 point) × 10채널 (A1–A5 **상단**, B1–B5 **하단**)
- 라벨: 장비 자가판정(임계) 출발 → 형태 규칙 기반 **weak label** 설계 (정답 라벨 아님)

**신호 형상 — staircase의 정체:** 강철바가 회전(462rpm)하며 통과하므로, 하나의 물리적 결함이 센서들을 *시간차를 두고 순차적으로* 자극함. 그 결과 결함 스파이크가 2D 맵 위에서 대각선 궤적을 그림:

```
        A1   A2   A3   A4   A5   B1   B2   B3   B4   B5
t0:    0.1  0.1  0.1  0.1  0.1  2.8  0.1  0.1  0.1  0.1
t1:    0.1  0.1  0.1  0.1  2.3  0.1  0.1  0.1  0.1  0.1     ← 결함이 회전하며
t2:    0.1  0.1  0.1  2.1  0.1  0.1  0.1  0.1  0.1  0.1        센서를 순차 자극
   (베이스 0.1V vs 결함 스파이크 2~3V / 나머지 칸도 빈 게 아니라 정상 베이스 신호)
```

세 형태: `staircase`(위처럼 대각) / `band`(같은 t에서 여러 채널 동시 = 원주방향) / `line`(한 채널에서 길이방향 연속).

> **시각 자료 (추후 보완 예정)**: bar 신호맵 heatmap · staircase/band/line 형태 예시 · confusion matrix · 채널셔플 전후 비교

## 3. 주요 엔지니어링 결정 (테크닉 + 근거)

이 프로젝트의 핵심은 모델이 아니라 **각 결정의 근거**다. "왜 이렇게 했나"를 데이터로 검증함.

### 3.1 입력 표현
- **log1p 진폭** — 처음엔 채널별 robust z-score(median/MAD)를 썼으나, 그것이 결함 스파이크와 *채널 간 상대크기*를 죽여 macro-F1이 무너짐(0.337). `log1p`로 교체하자 스파이크·상대크기가 보존되며 **0.337 → 0.813 도약**. *정규화 선택 하나가 성능을 가른 사례.*
- **4채널 `[amp, mask, d_len, d_ch]`** — 진폭 외에 임계 mask와 **gradient 2채널**(길이/채널 방향 변화율)을 추가해 결함 경계·방향성을 강조. gradient + 데이터 확대가 2차 macro-F1을 0.728 → 0.894로 견인.

### 3.2 component 추출 & 패치
- `scipy.ndimage` **8-connectivity**로 임계초과 셀을 결함 덩어리(component)로 분리
- component 중심 ±32 → **`(64, 10)` 고정 패치**, 경계는 `replicate` padding

### 3.3 센서 topology 역추정 (도메인 지식을 데이터로 복원)
- 장비 표기(A/B 2×5)를 그대로 안 믿고, **약 30만 component의 채널 전이 빈도**를 집계
- 결과: `A1–A2–A3–A4–A5–B1–B2–B3–B4–B5` 원주순서가 하나의 사슬, 특히 **A5–B1 인접(5,873회)** 확인
- **B5–A1 전이는 0** → 원주 폐곡선이 아닌 *열린 사슬* → `circular` padding이 아니라 `replicate`가 맞음을 데이터로 검증
- 입력 채널을 이 원주순서로 배열 → staircase가 끊기지 않고 대각선으로 보존

### 3.4 weak label 설계
- 형태 규칙: 채널중심이 단조 이동(diagonal) → `staircase`, 다채널·짧은 row → `band`, 긴 row·소채널 → `line`, 작은 고립점 → `isolated`
- 회색지대(`ambiguous`)는 학습에서 **제외**해 label noise 감소
- *정답이 아닌 weak label임을 명시* — failure mode "후보"로 표현

### 3.5 모델 & 학습
- **2D CNN**: 3× (Conv 3×3 → BatchNorm → GELU) + Dropout, 채널 48/96/192, `replicate` padding, adaptive average pool → FC
- **class weight** (불균형: staircase 108k / band 5.9k / line 3.4k)
- **cosine LR schedule** + **label smoothing 0.05**

### 3.6 검증 (누수 방지가 핵심)
- **LOT 단위 group split** — 같은 LOT의 bar·component가 train/test에 섞이면 누수 → LOT 단위로 분리
- **accuracy 금지** (클래스 불균형) → **PR-AUC / macro-F1** 사용
- baseline: **Gradient Boosting (shape feature)** — 해석가능 모델이 어디까지 가나 비교

## 4. 결과

| 실험 | baseline (GB) | 2D CNN |
|---|---|---|
| 1차 (구조 vs 노이즈) | PR-AUC 0.859 | **PR-AUC 0.961** |
| 2차 (형태 3-class) | macro-F1 0.748 | **macro-F1 0.903** |

## 5. Ablation — 가설을 데이터로 검증

| ablation | 결과 | 해석 |
|---|---|---|
| **채널 셔플** | 2차 0.894 → **0.650** (−0.24) | 채널 순서를 깨면 GB(0.748)보다도 낮아짐 → **센서 공간 배치·방향성이 결정적** |
| **late-fusion** (명시적 방향벡터 concat) | 0.894 → 0.651 하락 | CNN이 raw에서 공간 방향을 이미 충분히 학습 → 불필요 피처 제거 (Occam) |
| **데이터·gradient** | 0.728 → 0.903 | 데이터 5배 + gradient 채널이 최대 레버 |

1차는 weak label이 "퍼짐 크기"로 정의돼 비교적 쉬움(GB도 0.859). **2차 형태분류가 본게임** — 크기로 안 갈리고 *방향*이 핵심이라 GB가 0.748로 약하고, CNN이 +0.16 압도하며, 채널 셔플에 −0.24 폭락함.

## 6. 한계 (정직)

- weak label(형태 규칙 기반) → 성능 천장이 규칙 정확도에 묶임
- 진짜 결함의 물리적 ground truth(절단검사)는 없음 → "진짜 결함" 단정이 아닌 "후보"로 표현
- 향후: 약 85,000 bar **무라벨** 신호로 self-supervised 사전학습 → 적은 라벨로 fine-tune

## 7. 반도체 Product Engineering 연결

- 결함 **형태(signature) → failure mode → 의심 원인 좁히기** = yield-learning 루프
- 장비 임계가 만든 **허위경보(overkill) 분리** = PE 핵심 KPI(수율 손실 방지)
- wafer/bin map의 공간 failure-pattern classification과 **구조적으로 유사** (물리현상 동일시 아님)

## Stack
Python · PyTorch · scikit-learn · SciPy · NumPy
