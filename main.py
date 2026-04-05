import pandas as pd
from tqdm import tqdm
from src.data_manager import load_dataset, save_dataset
from src.preprocessing import preprocess
from src.embedding import embedding
from src.deduplication import deduplicate
from src.reasoning import add_reasoning
from src.verification import verify

tqdm.pandas()

def main():
    # 1. Load Dataset
    print("Loading dataset...")
    try:
        df = load_dataset("dataset.csv")
    except FileNotFoundError:
        print("Error: dataset.csv not found.")
        return
    
    # Optional: limit for quick testing (commented out for full run)
    # df = df.head(5)

    # 2. Preprocess Dataset
    print("Preprocessing questions (LLM-based canonicalization)...")
    df, dup_df = preprocess(df)

    # 3. Embedding Generation
    print("Generating embeddings for deduplication...")
    embeddings_map = embedding(df)

    # 4. Deduplication
    print("Deduplicating...")
    df, dup_dedup_df = deduplicate(df, embeddings_map)

    dup_df = pd.concat([dup_df, dup_dedup_df], ignore_index=True)

    # 5. Reasoning Generation
    print("Generating reasoning...")
    df = add_reasoning(df)

    # 6. Answer Verification
    print("Verifying answers...")
    correct_df, incorrect_df = verify(df)

    # 7. Output Dataset
    print("Saving output dataset...")
    save_dataset(dup_df, "duplicates.csv")

    correct_df = correct_df[['question', 'answer', 'reasoning']]
    correct_df.columns = ['question', 'answer', 'reasoning']
    save_dataset(correct_df, "correct_dataset.csv")

    incorrect_df = incorrect_df[['question', 'answer', 'reasoning']]
    incorrect_df.columns = ['question', 'answer', 'reasoning']
    save_dataset(incorrect_df, "incorrect_dataset.csv")
    print("Pipeline completed.")

if __name__ == "__main__":
    main()
