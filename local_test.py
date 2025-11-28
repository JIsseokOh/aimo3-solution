"""
AIMO3 Local Test Script
API 기반으로 reference 문제를 테스트합니다.

사용법:
1. 환경변수 설정:
   - ANTHROPIC_API_KEY (Claude API 사용 시)
   - OPENAI_API_KEY (OpenAI API 사용 시)

2. 실행:
   python local_test.py
"""

import os
import re
import csv
import time
from collections import Counter
from typing import Optional, List, Tuple
from io import StringIO
from contextlib import redirect_stdout

# API 선택
USE_CLAUDE = os.getenv("ANTHROPIC_API_KEY") is not None
USE_OPENAI = os.getenv("OPENAI_API_KEY") is not None

SYSTEM_PROMPT = """You are a world-class mathematician solving olympiad-level math problems.

IMPORTANT RULES:
1. Think step by step carefully
2. Write Python code to verify calculations when needed
3. Put your code inside ```python and ``` tags
4. Execute the code mentally and use the results
5. Your final answer must be a single non-negative integer
6. Put your FINAL answer inside \\boxed{} at the very end

Remember: The answer should be the remainder when divided by 10^5 if the problem asks for it.
"""

def extract_code_blocks(text: str) -> List[str]:
    """텍스트에서 Python 코드 블록 추출"""
    pattern = r'```python\s*(.*?)\s*```'
    matches = re.findall(pattern, text, re.DOTALL)
    return matches

def execute_code_safely(code: str) -> Tuple[bool, str]:
    """코드를 안전하게 실행"""
    import math
    import cmath
    import fractions
    import itertools
    import functools
    from decimal import Decimal

    try:
        import sympy
    except ImportError:
        sympy = None

    try:
        import numpy as np
    except ImportError:
        np = None

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
        },
        "math": math,
        "cmath": cmath,
        "fractions": fractions,
        "Fraction": fractions.Fraction,
        "itertools": itertools,
        "functools": functools,
        "Decimal": Decimal,
    }

    if sympy:
        safe_globals["sympy"] = sympy
    if np is not None:
        safe_globals["np"] = np
        safe_globals["numpy"] = np

    stdout_capture = StringIO()

    try:
        with redirect_stdout(stdout_capture):
            exec(code, safe_globals)
        return True, stdout_capture.getvalue()
    except Exception as e:
        return False, f"Error: {str(e)}"

def extract_answer(text: str) -> Optional[int]:
    """텍스트에서 최종 답 추출"""
    # \boxed{...} 패턴 먼저 찾기
    boxed_pattern = r'\\boxed\{([^}]+)\}'
    matches = re.findall(boxed_pattern, text)

    if matches:
        answer_str = matches[-1].strip()
        # 숫자만 추출
        numbers = re.findall(r'-?\d+', answer_str)
        if numbers:
            try:
                return int(numbers[-1]) % 100000
            except:
                pass

    # "answer is X" 패턴 찾기
    patterns = [
        r'answer\s*(?:is|=|:)\s*(\d+)',
        r'remainder\s*(?:is|=|:)\s*(\d+)',
        r'final\s*answer[:\s]+(\d+)',
        r'therefore[,\s]+(\d+)',
        r'= (\d+)$',
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text.lower())
        if matches:
            try:
                return int(matches[-1]) % 100000
            except:
                pass

    return None

def solve_with_claude(problem: str, num_samples: int = 8) -> int:
    """Claude API를 사용하여 문제 해결"""
    import anthropic

    client = anthropic.Anthropic()
    answers = []

    for i in range(num_samples):
        try:
            message = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4096,
                temperature=0.7 if i > 0 else 0,  # 첫 번째는 deterministic
                system=SYSTEM_PROMPT,
                messages=[
                    {"role": "user", "content": f"Solve this math problem:\n\n{problem}"}
                ]
            )

            response = message.content[0].text

            # 코드 블록 실행
            code_blocks = extract_code_blocks(response)
            code_outputs = []
            for code in code_blocks:
                success, output = execute_code_safely(code)
                if success and output.strip():
                    code_outputs.append(output.strip())

            # 코드 결과가 있으면 추가 추론 요청
            if code_outputs:
                follow_up = f"Code execution results:\n" + "\n".join(code_outputs)
                follow_up += "\n\nBased on these results, what is the final numerical answer? Put it in \\boxed{}"

                message2 = client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=1024,
                    temperature=0,
                    messages=[
                        {"role": "user", "content": f"Solve this math problem:\n\n{problem}"},
                        {"role": "assistant", "content": response},
                        {"role": "user", "content": follow_up}
                    ]
                )
                response = response + "\n" + message2.content[0].text

            answer = extract_answer(response)
            if answer is not None:
                answers.append(answer)
                print(f"  Sample {i+1}: {answer}")

        except Exception as e:
            print(f"  Sample {i+1} error: {e}")

    if answers:
        counter = Counter(answers)
        return counter.most_common(1)[0][0]
    return 0

def solve_with_openai(problem: str, num_samples: int = 8) -> int:
    """OpenAI API를 사용하여 문제 해결"""
    from openai import OpenAI

    client = OpenAI()
    answers = []

    for i in range(num_samples):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                max_tokens=4096,
                temperature=0.7 if i > 0 else 0,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Solve this math problem:\n\n{problem}"}
                ]
            )

            text = response.choices[0].message.content

            # 코드 블록 실행
            code_blocks = extract_code_blocks(text)
            for code in code_blocks:
                success, output = execute_code_safely(code)

            answer = extract_answer(text)
            if answer is not None:
                answers.append(answer)
                print(f"  Sample {i+1}: {answer}")

        except Exception as e:
            print(f"  Sample {i+1} error: {e}")

    if answers:
        counter = Counter(answers)
        return counter.most_common(1)[0][0]
    return 0

def main():
    # Reference 문제 로드
    reference_path = os.path.join(os.path.dirname(__file__), "reference.csv")

    problems = []
    with open(reference_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            problems.append({
                'id': row['id'],
                'problem': row['problem'],
                'answer': int(row['answer'])
            })

    print(f"Loaded {len(problems)} reference problems")
    print(f"Using Claude API: {USE_CLAUDE}")
    print(f"Using OpenAI API: {USE_OPENAI}")
    print("=" * 60)

    correct = 0
    total = 0

    for prob in problems:
        print(f"\nProblem {prob['id']}")
        print(f"Expected answer: {prob['answer']}")
        print("-" * 40)

        start_time = time.time()

        if USE_CLAUDE:
            predicted = solve_with_claude(prob['problem'], num_samples=4)
        elif USE_OPENAI:
            predicted = solve_with_openai(prob['problem'], num_samples=4)
        else:
            print("No API key found. Please set ANTHROPIC_API_KEY or OPENAI_API_KEY")
            return

        elapsed = time.time() - start_time

        is_correct = predicted == prob['answer']
        if is_correct:
            correct += 1
        total += 1

        print(f"\nPredicted: {predicted}")
        print(f"Expected: {prob['answer']}")
        print(f"Correct: {'✓' if is_correct else '✗'}")
        print(f"Time: {elapsed:.1f}s")
        print("=" * 60)

    print(f"\n\nFinal Score: {correct}/{total} ({100*correct/total:.1f}%)")

if __name__ == "__main__":
    main()
