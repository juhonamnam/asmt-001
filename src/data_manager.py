import pandas as pd

def load_dataset(file_path):
    return pd.read_csv(file_path)

def save_dataset(df, file_path):
    df.to_csv(file_path, index=False)
