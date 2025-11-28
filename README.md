# AIMO3 Solution - AI Mathematical Olympiad Progress Prize 3

## Quick Start

### Kaggle 제출
1. `kaggle_submission_notebook.ipynb`를 Kaggle에 Import
2. Model 추가: `qwen2.5-math` (72b-instruct)
3. GPU: H100 선택
4. Run All → Submit

### 현재 상태
**Phase 1 진행 중** - 베이스라인 점수 확인 대기

자세한 내용은 `PROJECT_STATUS.md` 참조

---

## 프로젝트 구조

```
aimo3/
├── kaggle_submission_notebook.ipynb  # 메인 제출 노트북 ⭐
├── PROJECT_STATUS.md                 # 프로젝트 현황 (필독) ⭐
├── 우승전략.md                        # 6단계 우승 전략
├── 대회설명.txt                       # 대회 상세 설명
├── reference.csv                     # 연습 문제 (10개)
└── ...
```

---

## 핵심 설정

```python
Model: Qwen2.5-Math-72B-Instruct
GPU: H100
Quantization: 4-bit (BitsAndBytes)
Samples: 8 (Self-Consistency)
Features: Feedback Loop + Self-Verification
```

---

## 문서

| 문서 | 설명 |
|------|------|
| `PROJECT_STATUS.md` | 현재 상태, 다음 할 일, 코드 설명 |
| `우승전략.md` | 6단계 우승 전략 상세 |
| `대회설명.txt` | AIMO3 대회 규칙 및 정보 |

---

## 6단계 전략

1. **베이스라인** ← 현재
2. 약점 분석
3. 데이터 수집
4. 파인튜닝
5. 추론 고도화
6. 최종 제출

---

## Links

- [Kaggle Competition](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3)
- [AIMO Official](https://aimoprize.com)
