import pandas as pd

def load_data(csv_path, target_col="target"):
    df = pd.read_csv(csv_path)
    X = df.drop(columns=[target_col]) # X est un dataframe 
    Y = df[target_col] # Y est une série  , c'est les valeur dans la colonne.
    
    return X, Y

if __name__ == "__main__":
    X, Y = load_data(csv_path="projet de fin de module 2025-2026/bienetre.csv", target_col="target")
    print(X.head())
    print(Y.head())