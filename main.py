import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Chemin par défaut du jeu de données bien-être
DEFAULT_CSV_PATH = "projet de fin de module 2024-2025/bienetre.csv"


def load_data(csv_path, target_col="target"):
    """
    Charge le CSV et sépare features (X) et cible (y).

    X ne contient jamais la colonne cible : indispensable pour éviter
    une fuite de données (le modèle ne doit pas voir target à l'entraînement).
    """
    df = pd.read_csv(csv_path)

    # X = toutes les colonnes sauf la cible
    X = df.drop(columns=[target_col])
    # y = variable à prédire
    y = df[target_col]

    return df, X, y


if __name__ == "__main__":

    df, X, y = load_data(
        csv_path=DEFAULT_CSV_PATH,
        target_col="target",
    )

    print("Dimensions X :", X.shape)
    print("Dimensions y :", y.shape)

    print("\nRépartition des classes :")
    print(y.value_counts())

    # Pipeline : normalisation puis KNN (l'ordre compte)
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(
            n_neighbors=5,
            p=2,
            metric="minkowski",
        )),
    ])

    # Cross-validation stratifiée : chaque fold garde les mêmes proportions de classes
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="accuracy",
    )

    print("\nScores cross-validation :")
    print(scores)

    print("\nAccuracy moyenne CV :", scores.mean())
    print("Écart-type CV :", scores.std())

    # 80 % train / 20 % test, avec les mêmes proportions de classes qu'avant le split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\nAccuracy test :", accuracy_score(y_test, y_pred))

    print("\nClassification Report :")
    print(classification_report(y_test, y_pred))

    print("\nMatrice de confusion :")
    print(confusion_matrix(y_test, y_pred))

    # Corrélation de chaque feature avec la cible (exploration des données)
    print("\nCorrélation avec target :")
    print(df.corr()["target"].sort_values(ascending=False))
