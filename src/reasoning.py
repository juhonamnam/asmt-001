from .client import client

REASONING_MODEL = "gpt-4o"

def generate_reasoning(question):
    # Generate step-by-step reasoning
    try:
        response = client.chat.completions.create(
            model=REASONING_MODEL,
            messages=[
                {"role": "system", "content": "You are a math expert. Given a math question, provide a solution and a detailed step-by-step reasoning in Korean. Format the reasoning as a numbered list."},
                {"role": "user", "content": f"Question: {question}"}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating reasoning: {e}")
        return "Reasoning generation failed."

def add_reasoning(df):
    df['reasoning'] = df['canonical_question'].progress_apply(generate_reasoning)
    return df
