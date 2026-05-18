"""Tests unitaires du pipeline KNN (chargement, features, entraînement, métriques)."""

from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from main import load_data

CSV_PATH = "projet de fin de module 2024-2025/bienetre.csv"


def test_load_data():
    """Vérifie que le CSV charge et que X / y ont les bonnes dimensions."""
    df, X, y = load_data(CSV_PATH)

    assert df is not None
    assert X.shape[0] == y.shape[0]
    assert X.shape[1] == 20


def test_target_not_in_features():
    """La variable cible ne doit jamais apparaître dans les features."""
    _, X, _ = load_data(CSV_PATH)

    assert "target" not in X.columns


def test_target_classes():
    """Le problème est une classification à 3 classes : 0, 1 et 2."""
    _, _, y = load_data(CSV_PATH)

    classes = set(y.unique())

    assert classes == {0, 1, 2}


def test_knn_training():
    """Le pipeline StandardScaler + KNN doit s'entraîner sans erreur."""
    _, X, y = load_data(CSV_PATH)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier()),
    ])

    model.fit(X, y)

    assert model is not None


def test_prediction_shape():
    """Le nombre de prédictions doit correspondre au nombre d'échantillons."""
    _, X, y = load_data(CSV_PATH)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier()),
    ])

    model.fit(X, y)

    predictions = model.predict(X)

    assert len(predictions) == len(y)


def test_accuracy():
    """L'accuracy sur l'ensemble d'entraînement doit rester au-dessus d'un seuil minimal."""
    _, X, y = load_data(CSV_PATH)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier()),
    ])

    model.fit(X, y)

    predictions = model.predict(X)

    accuracy = accuracy_score(y, predictions)

    assert accuracy > 0.8
