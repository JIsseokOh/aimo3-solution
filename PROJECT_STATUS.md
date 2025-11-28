# AIMO3 프로젝트 현황 및 진행 가이드

**최종 업데이트**: 2024-11-29
**GitHub**: https://github.com/JIsseokOh/aimo3-solution

---

## 프로젝트 개요

- **대회**: AI Mathematical Olympiad - Progress Prize 3
- **목표**: 우승 ($2.2M 상금)
- **접근**: Qwen2.5-Math-72B + 파인튜닝 + 고급 추론

---

## 현재 상태

### ✅ 완료된 작업

1. **대회 분석 완료**
   - 대회설명.txt 작성
   - 평가 방식 파악 (Penalized Accuracy)
   - 문제 형식 파악 (5자리 정수 답)

2. **베이스라인 코드 완성**
   - `kaggle_submission_notebook.ipynb` 작성
   - SC-TIR (Self-Consistency with Tool-Integrated Reasoning) 구현
   - 코드 실행 피드백 루프 구현
   - Self-verification 로직 구현

3. **모델 설정 완료**
   - 모델: Qwen2.5-Math-72B-Instruct
   - 경로: `/kaggle/input/qwen2.5-math/transformers/72b-instruct/1`
   - 양자화: 4-bit (BitsAndBytes)
   - GPU: H100

4. **이전 테스트 결과** (NuminaMath-7B 사용 시)
   - 샘플 3문제 모두 정답 (0, 0, 0)
   - 정상 작동 확인

### ⏳ 현재 진행 중

- **Phase 1**: Qwen2.5-Math-72B로 Submit 대기
- 점수 확인 후 Phase 2 진행 예정

---

## 파일 구조

```
aimo3/
├── kaggle_submission_notebook.ipynb  # 메인 제출 노트북
├── aimo3_solution.py                 # Python 스크립트 버전
├── local_test.py                     # 로컬 API 테스트용
├── 대회설명.txt                       # 대회 상세 설명
├── 우승전략.md                        # 6단계 우승 전략
├── PROJECT_STATUS.md                 # 이 문서
├── README.md                         # 프로젝트 안내
├── reference.csv                     # 연습 문제 (정답 포함, 10개)
├── test.csv                          # 테스트 문제 (샘플)
├── sample_submission.csv             # 제출 형식 예시
└── kaggle_evaluation/                # Kaggle 평가 API
```

---

## 현재 설정 (kaggle_submission_notebook.ipynb)

```python
class Config:
    model_id = "/kaggle/input/qwen2.5-math/transformers/72b-instruct/1"
    num_samples = 8           # 72B는 더 똑똑해서 적게 필요
    temperature = 0.7
    max_new_tokens = 2048
    top_p = 0.95
    code_timeout = 10
    max_code_executions = 3
    enable_feedback_loop = True
    max_feedback_turns = 1
    enable_verification = True
    use_4bit = True           # 4-bit 양자화
```

---

## 6단계 우승 전략

### Phase 1: 베이스라인 확립 ⏳ (현재)
- [x] Qwen2.5-Math-72B 설정
- [x] H100 GPU 환경
- [ ] **Submit 및 점수 확인** ← 다음 작업

### Phase 2: 약점 분석
- [ ] reference.csv 10문제 테스트
- [ ] 문제 유형별 정확도 측정
- [ ] 오류 패턴 분석

### Phase 3: 데이터 수집
- [ ] IMO/AIME/국가올림피아드 문제 수집
- [ ] 합성 데이터 생성
- [ ] 목표: 10,000+ 문제-풀이 쌍

### Phase 4: 파인튜닝
- [ ] QLoRA 학습 코드 작성
- [ ] Kaggle H100 128대 또는 자체 GPU
- [ ] 파인튜닝 실행

### Phase 5: 추론 고도화
- [ ] Multi-model 앙상블
- [ ] MCTS 적용
- [ ] Self-verification 강화

### Phase 6: 최종 제출
- [ ] 최적화된 솔루션 Submit
- [ ] 추가 Prize 도전

---

## 다음 세션에서 할 일

### 시나리오 A: Submit 점수가 나온 경우
```
1. 점수 확인 및 기록
2. Phase 2 시작: reference.csv로 상세 분석
3. 약점 파악 후 개선 방향 결정
```

### 시나리오 B: Submit이 아직 안 된 경우
```
1. Kaggle에서 Submit 진행
   - 노트북: kaggle_submission_notebook.ipynb
   - 모델: /kaggle/input/qwen2.5-math/transformers/72b-instruct/1
   - GPU: H100
2. 결과 대기 (최대 9시간)
```

### 시나리오 C: 에러 발생한 경우
```
1. 에러 로그를 출력.txt에 저장
2. Claude에게 파일 읽고 수정 요청
```

---

## Kaggle 제출 방법

1. https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3 접속
2. Code 탭 → New Notebook
3. File → Import Notebook → `kaggle_submission_notebook.ipynb` 업로드
4. **+ Add Input**:
   - Models: `qwen2.5-math` (72b-instruct)
   - (대회 데이터는 자동 연결됨)
5. Settings → Accelerator → **H100**
6. **Run All** → 테스트 실행
7. **Submit** 버튼 클릭

---

## 주요 코드 설명

### 1. 모델 로딩 (4-bit 양자화)
```python
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)
model = AutoModelForCausalLM.from_pretrained(
    Config.model_id,
    quantization_config=quantization_config,
    ...
)
```

### 2. 문제 풀이 흐름
```
문제 입력
    ↓
프롬프트 생성 (Chain-of-Thought)
    ↓
모델 추론 (8회 샘플링)
    ↓
코드 블록 추출 및 실행
    ↓
피드백 루프 (코드 결과 → 재추론)
    ↓
Self-verification
    ↓
다수결 투표 → 최종 답
```

### 3. 핵심 함수
- `solve_problem(problem)`: 메인 문제 풀이
- `solve_with_feedback(problem)`: 피드백 루프 포함 풀이
- `extract_code_blocks(text)`: 코드 블록 추출
- `execute_code_safely(code)`: 안전한 코드 실행
- `extract_answer(text)`: 답 추출 (\boxed{} 등)

---

## 참고 자료

- **대회 페이지**: https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3
- **AIMO 공식**: https://aimoprize.com
- **AIMO1 우승 솔루션**: https://github.com/project-numina/aimo-progress-prize
- **Qwen2.5-Math 논문**: https://arxiv.org/abs/2409.12122

---

## 연락처 / 메모

- GitHub: JIsseokOh
- 프로젝트 저장소: https://github.com/JIsseokOh/aimo3-solution

---

## 변경 이력

| 날짜 | 변경 내용 |
|------|----------|
| 2024-11-29 | 프로젝트 초기화, 베이스라인 코드 작성 |
| 2024-11-29 | NuminaMath-7B → Qwen2.5-Math-72B 업그레이드 |
| 2024-11-29 | H100 + 4-bit 양자화 설정 |
| 2024-11-29 | 6단계 우승 전략 수립 |
