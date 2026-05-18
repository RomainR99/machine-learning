import pandas as pd

def load_data(csv_path, target_col="target"):
    df = pd.read_csv(csv_path)
    X = df.drop(columns=[target_col]) # X est un dataframe 
    Y = df[target_col] # Y est une série  , c'est les valeur da