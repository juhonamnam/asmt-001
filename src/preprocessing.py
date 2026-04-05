import pandas as pd
from .client import client

CANNONICALIZATION_MODEL = "gpt-4o-mini"

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
                {"role": "system", "content": "You are a math expert. Rewrite the following math question in a standard, canonical form. Keep all mathematical notations (LaTeX) intact but ensure the phrasing is clean and standardized in Korean. Only return the canonicalized question without any other text."},
                {"role": "user", "content": question}
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
