import pandas as pd
from .client import client

CANNONICALIZATION_MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = "\n".join([
    "당신은 수학 문제 텍스트를 표준화하는 전처리 시스템입니다. 주어진 수학 문제에서 의미는 유지하면서 표현상의 차이를 제거하여 하나의 일관된 형태로 변환하는 것이 목표입니다.",
    "",
    "다음 규칙을 따르세요.",
    "",
    "1. 문제의 수학적 의미는 절대 변경하지 않습니다.",
    "2. 문장 종결 표현은 \"구하시오.\" 형태로 통일합니다.",
    "3. 미지수는 가능한 경우 x, y, z를 사용합니다.",
    "4. 상수나 계수는 가능한 경우 a, b, c를 사용합니다.",
    "5. 불필요한 설명이나 장황한 표현은 제거합니다.",
    "6. 수학적 식과 조건은 그대로 유지합니다.",
    "7. 추가 설명이나 해설은 출력하지 않습니다.",
])

USER_PROMPT = "다음 수학 문제를 표준화된 형태로 변환하세요:\n\n{}"

def clean_question(question):
    # Strip whitespace, normalize newlines, etc.
    question = str(question).strip()
    # Replace multiple newlines with single space
    question = " ".join(question.split())

    return question

def canonicalize_question(question):
    # Use LLM to canonicalize question (e.g., standardizing LaTeX)
    try:
        response = client.chat.completions.create(
            model=CANNONICALIZATION_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT.format(question)}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error canonicalizing question: {e}")
        return question

def preprocess(df: pd.DataFrame):
    print("Cleaning questions...")
    df['question'] = df['question'].progress_apply(clean_question)

    is_duplicate = df.duplicated(subset=['question'], keep='first')

    dup_question_series = df[is_duplicate]['question']
    dup_df = pd.DataFrame({
        'question': dup_question_series,
        'duplicated_from': dup_question_series,
        'reason': 'Identical question text'
    })

    df = df[~is_duplicate]  # Keep only unique questions for canonicalization

    print("Canonicalizing questions with LLM...")
    df['canonical_question'] = df['question'].progress_apply(canonicalize_question)

    is_duplicate = df.duplicated(subset=['canonical_question'], keep='first')

    dup_canonical_series = df[is_duplicate]['canonical_question']
    dup_canonical_df = pd.DataFrame({
        'question': dup_canonical_series,
        'duplicated_from': dup_canonical_series,
        'reason': 'Identical canonical question'
    })

    # Combine duplicate dataframes
    dup_df = pd.concat([dup_df, dup_canonical_df], ignore_index=True)

    return df, dup_df
