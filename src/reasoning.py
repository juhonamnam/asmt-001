from .client import client

REASONING_MODEL = "gpt-4o"

SYSTEM_PROMPT = "\n".join([
    "당신은 수학 문제를 해결하는 튜터입니다. 주어진 수학 문제를 단계적으로 풀고 reasoning을 작성하세요.",
    "",
    "다음 규칙을 따르세요.",
    "",
    "1. 문제를 해결하는 과정을 단계적으로 설명합니다.",
    "2. 각 단계는 논리적으로 이어져야 합니다.",
    "3. 계산 과정이 있다면 명확히 설명합니다.",
    "4. 마지막 줄에는 반드시 최종 답을 작성합니다.",
    "",
    "출력 형식:",
    "",
    "Step 1: ...",
    "Step 2: ...",
    "Step 3: ...",
    "Final Answer: ...",
])

USER_PROMPT = "다음 수학 문제에 대한 자세한 reasoning을 작성하세요:\n\n{}"

def generate_reasoning(question):
    # Generate step-by-step reasoning
    try:
        response = client.chat.completions.create(
            model=REASONING_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT.format(question)}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating reasoning: {e}")
        return "Reasoning generation failed."

def add_reasoning(df):
    df['reasoning'] = df['question'].progress_apply(generate_reasoning)
    return df
