"""
AI Mathematical Olympiad - Progress Prize 3 Solution
SC-TIR (Self-Consistency with Tool-Integrated Reasoning) 기법 적용
"""

import os
import re
import sys
import gc
import time
import traceback
from collections import Counter
from typing import Optional

import polars as pl

# Kaggle 환경 확인
IS_KAGGLE = os.path.exists('/kaggle')
IS_SUBMISSION = bool(os.getenv("KAGGLE_IS_COMPETITION_RERUN"))

# 모델 설정
MODEL_CONFIG = {
    "model_id": "deepseek-ai/DeepSeek-Math-7B-Instruct",  # 또는 다른 수학 특화 모델
    "num_samples": 32,  # 답변 생성 횟수 (Self-Consistency)
    "temperature": 0.7,
    "max_new_tokens": 2048,
    "top_p": 0.95,
}

# 시스템 프롬프트
SYSTEM_PROMPT = """You are a world-class mathematician solving olympiad-level math problems.

IMPORTANT RULES:
1. Think step by step carefully
2. Write Python code to verify calculations when needed
3. Put your code inside ```python and ``` tags
4. After running the code, analyze the results
5. Your final answer must be a single non-negative integer
6. Put your FINAL answer inside \\boxed{} at the very end

Example format:
Let me solve this step by step...
[reasoning]

```python
# calculation code
result = ...
print(result)
```

After running the code, I get...
[more reasoning]

Therefore, the answer is \\boxed{42}
"""

def extract_code_blocks(text: str) -> list[str]:
    """텍스트에서 Python 코드 블록 추출"""
    pattern = r'```python\s*(.*?)\s*```'
    matches = re.findall(pattern, text, re.DOTALL)
    return matches

def execute_code_safely(code: str, timeout: int = 10) -> tuple[bool, str]:
    """코드를 안전하게 실행하고 결과 반환"""
    import signal
    from io import StringIO
    from contextlib import redirect_stdout, redirect_stderr

    # 안전한 실행 환경
    safe_globals = {
        "__builtins__": {
            "abs": abs, "all": all, "any": any, "bin": bin,
            "bool": bool, "dict": dict, "divmod": divmod,
            "enumerate": enumerate, "filter": filter, "float": float,
            "format": format, "frozenset": frozenset, "hash": hash,
            "hex": hex, "int": int, "isinstance": isinstance,
            "iter": iter, "len": len, "list": list, "map": map,
            "max": max, "min": min, "next": next, "oct": oct,
            "ord": ord, "pow": pow, "print": print, "range": range,
            "repr": repr, "reversed": reversed, "round": round,
            "set": set, "slice": slice, "sorted": sorted,
            "str": str, "sum": sum, "tuple": tuple, "type": type,
            "zip": zip, "True": True, "False": False, "None": None,
        }
    }

    # 수학 라이브러리 추가
    try:
        import math
        import cmath
        import fractions
        import itertools
        import functools
        from decimal import Decimal
        import sympy
        import numpy as np
        from scipy import special

        safe_globals["math"] = math
        safe_globals["cmath"] = cmath
        safe_globals["fractions"] = fractions
        safe_globals["Fraction"] = fractions.Fraction
        safe_globals["itertools"] = itertools
        safe_globals["functools"] = functools
        safe_globals["Decimal"] = Decimal
        safe_globals["sympy"] = sympy
        safe_globals["np"] = np
        safe_globals["numpy"] = np
        safe_globals["scipy"] = {"special": special}
    except ImportError:
        pass

    stdout_capture = StringIO()
    stderr_capture = StringIO()

    try:
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            exec(code, safe_globals)
        output = stdout_capture.getvalue()
        return True, output
    except Exception as e:
        return False, f"Error: {str(e)}"

def extract_answer(text: str) -> Optional[int]:
    """텍스트에서 최종 답 추출"""
    # \boxed{...} 패턴 먼저 찾기
    boxed_pattern = r'\\boxed\{([^}]+)\}'
    matches = re.findall(boxed_pattern, text)

    if matches:
        # 마지막 boxed 값 사용
        answer_str = matches[-1].strip()
        # 숫자만 추출
        numbers = re.findall(r'-?\d+', answer_str)
        if numbers:
            try:
                return int(numbers[-1]) % 100000  # 5자리로 제한
            except:
                pass

    # boxed가 없으면 "answer is X" 패턴 찾기
    answer_patterns = [
        r'answer\s*(?:is|=|:)\s*(\d+)',
        r'final\s*answer\s*(?:is|=|:)\s*(\d+)',
        r'therefore[,\s]+(?:the\s+)?answer\s*(?:is)?\s*(\d+)',
        r'(?:thus|hence)[,\s]+(\d+)',
    ]

    for pattern in answer_patterns:
        matches = re.findall(pattern, text.lower())
        if matches:
            try:
                return int(matches[-1]) % 100000
            except:
                pass

    # 마지막 숫자 찾기
    numbers = re.findall(r'\b(\d+)\b', text[-500:])
    if numbers:
        try:
            return int(numbers[-1]) % 100000
        except:
            pass

    return None

def solve_with_tool_integrated_reasoning(problem: str, model, tokenizer) -> int:
    """Tool-Integrated Reasoning으로 문제 해결"""
    answers = []

    for i in range(MODEL_CONFIG["num_samples"]):
        try:
            # 프롬프트 구성
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Solve this math problem:\n\n{problem}"}
            ]

            # 모델 응답 생성
            response = generate_response(model, tokenizer, messages)

            # 코드 블록 실행
            code_blocks = extract_code_blocks(response)
            if code_blocks:
                code_outputs = []
                for code in code_blocks:
                    success, output = execute_code_safely(code)
                    code_outputs.append(output if success else f"[Code Error: {output}]")

                # 코드 결과를 포함하여 다시 추론
                follow_up = response + "\n\n**Code Output:**\n" + "\n".join(code_outputs)
                follow_up += "\n\nBased on the code output above, what is the final answer? Put it in \\boxed{}."

                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "user", "content": "What is the numerical answer? Put it in \\boxed{}."})
                response = generate_response(model, tokenizer, messages)

            # 답 추출
            answer = extract_answer(response)
            if answer is not None:
                answers.append(answer)

        except Exception as e:
            print(f"Sample {i} error: {e}")
            continue

    # 다수결로 최종 답 결정
    if answers:
        counter = Counter(answers)
        most_common = counter.most_common(1)[0][0]
        return most_common

    return 0  # 기본값

def generate_response(model, tokenizer, messages: list) -> str:
    """모델로부터 응답 생성"""
    import torch

    # 메시지를 텍스트로 변환
    if hasattr(tokenizer, 'apply_chat_template'):
        input_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    else:
        input_text = "\n".join([
            f"{m['role'].upper()}: {m['content']}" for m in messages
        ])

    # 토큰화
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    # 생성
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MODEL_CONFIG["max_new_tokens"],
            temperature=MODEL_CONFIG["temperature"],
            top_p=MODEL_CONFIG["top_p"],
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    # 디코딩
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return response

def load_model():
    """모델 로드"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = MODEL_CONFIG["model_id"]

    print(f"Loading model: {model_id}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer

def main():
    """메인 실행 함수"""
    # 데이터 경로
    if IS_KAGGLE:
        test_path = '/kaggle/input/ai-mathematical-olympiad-progress-prize-3/test.csv'
        output_path = '/kaggle/working/submission.csv'
    else:
        test_path = 'test.csv'
        output_path = 'submission.csv'

    # 테스트 데이터 로드
    test_df = pl.read_csv(test_path)
    print(f"Loaded {len(test_df)} problems")

    # 모델 로드
    model, tokenizer = load_model()

    # 결과 저장
    results = []

    for row in test_df.iter_rows(named=True):
        problem_id = row['id']
        problem = row['problem']

        print(f"\nSolving problem: {problem_id}")
        print(f"Problem: {problem[:200]}...")

        # 문제 해결
        start_time = time.time()
        answer = solve_with_tool_integrated_reasoning(problem, model, tokenizer)
        elapsed = time.time() - start_time

        print(f"Answer: {answer} (took {elapsed:.2f}s)")

        results.append({
            'id': problem_id,
            'answer': answer
        })

        # 메모리 정리
        gc.collect()

    # 결과 저장
    submission_df = pl.DataFrame(results)
    submission_df.write_csv(output_path)
    print(f"\nSubmission saved to {output_path}")

if __name__ == "__main__":
    main()
