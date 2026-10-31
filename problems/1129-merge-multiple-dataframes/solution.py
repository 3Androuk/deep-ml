import pandas as pd

def solution(df1, df2, df3):
    return df1.merge(df2, on='emp_id', how='inner').merge(df3, on='emp_id', how='left')
    

