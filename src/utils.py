import pandas as pd

def load_data(path):
    return pd.read_csv(path)

def save_to_csv(df, path):
    df.to_csv(path, index=False)

def format_columns(df):
    df.columns = df.columns.str.strip()
    return df