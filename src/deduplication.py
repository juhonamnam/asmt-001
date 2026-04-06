import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from .client import client

DEDUPLICATION_SIMILARITY_THRESHOLD = 0.9
DEDUPLICATION_LLM_MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = "\n".join([
    "당신은 수학 문제의 의미적 동일성을 판단하는 시스템입니다. 두 개의 수학 문제가 주어지면, 두 문제가 실제로 동일한 문제인지 판단하세요.",
    "",
    "판단 기준은 다음과 같습니다.",
    "",
    "1. 두 문제가 동일한 수학적 식이나 조건을 가지고 있고 같은 값을 구하도록 요구한다면 동일한 문제로 판단합니다.",
    "2. 문장 표현이 다르더라도 수학적으로 완전히 같은 문제라면 동일한 문제입니다.",
    "3. 단순히 같은 주제를 다루거나 비슷한 단어가 사용되었다고 해서 동일한 문제로 판단하지 않습니다.",
    "4. 수식의 계수나 조건이 조금이라도 다르면 다른 문제로 판단합니다.",
    "",
    "출력 형식:",
    "",
    "YES → 두 문제가 동일한 문제",
    "NO → 두 문제가 다른 문제",
    "",
    "설명 없이 YES 또는 NO만 출력하세요.",
])

USER_PROMPT = "다음 두 수학 문제의 의미적 동일성을 판단하세요:\n\n문제 1: {}\n\n문제 2: {}"

def deduplicate_with_llm(q1, q2):
    # Use LLM to check if two questions are semantically the same
    try:
        response = client.chat.completions.create(
            model=DEDUPLICATION_LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT.format(q1, q2)}
            ]
        )
        return response.choices[0].message.content.strip().lower() == "yes"
    except Exception as e:
        print(f"Error deduplicating with LLM: {e}")
        return False

def deduplicate_with_embedding(emb1, emb2):
    similarity = cosine_similarity([emb1], [emb2])[0][0]
    return similarity > DEDUPLICATION_SIMILARITY_THRESHOLD

def remove_duplicates(questions, embeddings_map):
    unique_questions = []
    for q in questions:
        is_duplicate = False
        current_embedding = np.array(embeddings_map[q]).reshape(1, -1)
        for uq in unique_questions:
            uq_embedding = np.array(embeddings_map[uq]).reshape(1, -1)
            similarity = cosine_similarity(current_embedding, uq_embedding)[0][0]
            if similarity > DEDUPLICATION_SIMILARITY_THRESHOLD:
                if deduplicate_with_llm(q, uq):
                    is_duplicate = True
                    break
        if not is_duplicate:
            unique_questions.append(q)
    return unique_questions

def deduplicate(df, embeddings_map):
    unique_questions = []
    duplicated_questions = []
    duplicated_from = []
    duplicated_reason = []

    def check_duplicates(q):
        is_duplicate = False
        for uq in unique_questions:
            if deduplicate_with_embedding(embeddings_map[q], embeddings_map[uq]):
                if deduplicate_with_llm(q, uq):
                    is_duplicate = True
                    duplicated_questions.append(q)
                    duplicated_from.append(uq)
                    duplicated_reason.append("High embedding similarity + LLM confirmation")
                    break
                else:
                    duplicated_questions.append(q)
                    duplicated_from.append(uq)
                    duplicated_reason.append("High embedding similarity but LLM says different")
        if not is_duplicate:
            unique_questions.append(q)
        return is_duplicate

    is_duplicate = df['canonical_question'].progress_apply(check_duplicates)

    dup_df = pd.DataFrame({
        'question': duplicated_questions,
        'duplicated_from': duplicated_from,
        'reason': duplicated_reason
    })

    df = df[~is_duplicate]

    return df, dup_df
