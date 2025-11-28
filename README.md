# AI Mathematical Olympiad - Progress Prize 3 Solution

## 개요
이 프로젝트는 Kaggle AIMO3 대회를 위한 수학 문제 해결 솔루션입니다.
SC-TIR (Self-Consistency with Tool-Integrated Reasoning) 기법을 사용합니다.

## 파일 구조
```
aimo3/
├── 대회설명.txt              # 대회 상세 설명
├── README.md                  # 이 파일
├── aimo3_solution.py          # 메인 솔루션 코드
├── kaggle_submission_notebook.ipynb  # Kaggle 제출용 노트북
├── local_test.py              # 로컬 테스트 스크립트
├── reference.csv              # 연습 문제 (정답 포함)
├── test.csv                   # 테스트 문제 (샘플)
├── sample_submission.csv      # 제출 형식 예시
└── kaggle_evaluation/         # Kaggle 평가 API
```

## 핵심 전략

### 1. SC-TIR (Self-Consistency with Tool-Integrated Reasoning)
- 동일 문제에 대해 여러 번 답변 생성 (32회)
- 각 답변에서 Python 코드를 추출하여 실행
- 코드 실행 결과를 활용하여 계산 검증
- 다수결(Majority Voting)로 최종 답 결정

### 2. 사용 모델 (Kaggle 환경)
- DeepSeek-Math-7B-Instruct
- Qwen2-Math
- NuminaMath-7B-CoT
- GPT-OSS-120B (H100 환경)

### 3. 추론 최적화
- vLLM을 사용한 빠른 배치 추론
- bfloat16 정밀도로 메모리 최적화
- GPU 메모리 90% 활용

## 사용 방법

### Kaggle 제출
1. `kaggle_submission_notebook.ipynb`를 Kaggle에 업로드
2. 수학 모델 데이터셋 추가 (DeepSeek-Math 등)
3. GPU Accelerator 활성화 (P100 또는 T4)
4. 노트북 실행 및 제출

### 로컬 테스트
```bash
# Claude API 사용
export ANTHROPIC_API_KEY=your_key
python local_test.py

# OpenAI API 사용
export OPENAI_API_KEY=your_key
python local_test.py
```

## 참고 자료
- [AIMO 공식 사이트](https://aimoprize.com)
- [Project Numina 솔루션](https://github.com/project-numina/aimo-progress-prize)
- [Kaggle 대회 페이지](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3)
