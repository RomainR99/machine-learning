import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report


def load_data(csv_path, target_col="target"):
    # Chargement du CSV
    df = pd.read_csv(csv_path)

    # Features
    X = df.drop(columns=[target_col])

    # Variable cible
    y = df[target_col]

    return df, X, y


if __name__ == "__main__":

    # Charger les données
    df, X, y = load_data(
        csv_path="projet de fin de module 2024-2025/bienetre.csv",
        target_col="target"
    )

    # Séparation train / test
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Pipeline Machine Learning
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(
            n_neighbors=5,
            p=2,
            metric="minkowski"
        ))
    ])

    # Entraînement du modèle
    model.fit(X_train, y_train)

    # Prédictions
    y_pred = model.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)

    print("Accuracy :", accuracy)

    # Rapport détaillé
    print("\nClassification Report :")
    print(classification_report(y_test, y_pred))

    # Corrélation avec la target
    print("\nCorrélation avec target :")
    print(df.corr()["target"].sort_values(ascending=False))