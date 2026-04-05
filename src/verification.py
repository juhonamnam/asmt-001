from .client import client

VERIFICATION_MODEL = "gpt-4o-mini"

def verify_answer(question, answer, reasoning):
    # Verify the generated answer against the ground truth
    try:
        response = client.chat.completions.create(
            model=VERIFICATION_MODEL,
            messages=[
                {"role": "system", "content": "You are a math expert. Verify if the provided reasoning correctly leads to the given answer for the specific question. Answer only 'True' if it's correct, or 'False' if it's not."},
                {"role": "user", "content": f"Question: {question}\nReasoning: {reasoning}\nAnswer: {answer}"}
            ]
        )
        return response.choices[0].message.content.strip().lower() == "true"
    except Exception as e:
        print(f"Error verifying answer: {e}")
        return False

def verify(df):
    is_correct = df.progress_apply(lambda row: verify_answer(row['canonical_question'], row['answer'], row['reasoning']), axis=1)

    correct_df = df[is_correct]
    incorrect_df = df[~is_correct]

    return correct_df, incorrect_df

