import pandas as pd

def solution(df):
    df_bin = df.isna().astype(int)
    keep_cols = df_bin.mean() <= 0.5
    keep_rows = df_bin.mean(axis=1) <= 0.5

    df = df.loc[:, keep_cols]
    df = df[keep_rows]

    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].fillna(df[c].mean())
        else:
            df[c] = df[c].fillna(df[c].mode().iloc[0])
    
    return df