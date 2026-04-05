import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from .client import client

DEDUPLICATION_SIMILARITY_THRESHOLD = 0.9
DEDUPLICATION_LLM_MODEL = "gpt-4o-mini"

def deduplicate_with_llm(q1, q2):
    # Use LLM to check if two questions are semantically the same
    try:
        response = client.chat.completions.create(
            model=DEDUPLICATION_LLM_MODEL,
            messages=[
                {"role": "system", "content": "You are a math expert. Compare two math questions and determine if they are the same or not. They might be phrased differently but ask for the same thing. Answer only 'True' if they are the same, or 'False' if they are different."},
                {"role": "user", "content": f"Question 1: {q1}\n\nQuestion 2: {q2}"}
            ]
        )
        return response.choices[0].message.content.strip().lower() == "true"
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
