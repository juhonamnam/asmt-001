from .client import client

VERIFICATION_MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = "\n".join([
    "당신은 수학 문제 풀이를 검증하는 시스템입니다. 주어진 수학 문제, 정답, 그리고 reasoning을 검토하여 reasoning이 논리적으로 올바른지 판단하세요.",
    "",
    "다음 기준을 따르세요.",
    "1. reasoning이 문제 해결 과정을 올바르게 설명하는지 확인합니다.",
    "2. reasoning의 계산 과정에 오류가 없는지 확인합니다.",
    "3. reasoning에서 도출된 최종 답이 정답과 일치하는지 확인합니다.",
    "4. reasoning이 잘못된 논리를 포함하거나 계산 오류가 있으면 INVALID로 판단합니다.",
    "",
    "출력 형식:",
    "",
    "VALID",
    "INVALID",
    "",
    "설명 없이 VALID 또는 INVALID만 출력하세요.",
])

USER_PROMPT = "다음 수학 문제 풀이를 검증하세요:\n\n문제: {}\n\nReasoning: {}\n\nAnswer: {}"

def verify_answer(question, answer, reasoning):
    # Verify the generated answer against the ground truth
    try:
        response = client.chat.completions.create(
            model=VERIFICATION_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT.format(question, reasoning, answer)}
            ]
        )
        return response.choices[0].message.content.strip().lower() == "valid"
    except Exception as e:
        print(f"Error verifying answer: {e}")
        return False

def verify(df):
    is_correct = df.progress_apply(lambda row: verify_answer(row['canonical_question'], row['answer'], row['reasoning']), axis=1)

    correct_df = df[is_correct]
    incorrect_df = df[~is_correct]

    return correct_df, incorrect_df

