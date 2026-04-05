from .client import client
from tqdm import tqdm

EMBEDDING_MODEL = "text-embedding-3-small"

def get_embedding(text):
    text = str(text).replace("\n", " ")
    try:
        return client.embeddings.create(input=[text], model=EMBEDDING_MODEL).data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return [0.0] * 1536 # Default size for text-embedding-3-small

def embedding(df):
    embeddings_map = {}
    for q in tqdm(df['canonical_question'], desc="Embeddings"):
        embeddings_map[q] = get_embedding(q)
    return embeddings_map
