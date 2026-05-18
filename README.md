# Pipeline ML propre

Projet de classification KNN sur le dataset **bien-être** (`bienetre.csv`), avec pipeline scikit-learn et **6 tests automatisés** (`pytest`).

**Démarrage rapide**

```bash
source venv/bin/activate
pip install -r requirements.txt
python main.py      # entraînement + métriques
pytest -v           # 6 tests unitaires
```

## Sommaire

### Projet

- [Structure du projet](#structure-du-projet)
- [Installation](#installation)

### Tests (`tests/test_main.py`)

- [Tests automatisés](#tests-automatisés)
  - [Récapitulatif des 6 tests](#récapitulatif-des-6-tests)
  - [Lancer les tests](#lancer-les-tests)
  - [Sortie attendue](#sortie-attendue)
  - [`test_load_data` — chargement](#test_load_data--chargement-des-données)
  - [`test_target_not_in_features` — fuite de données](#test_target_not_in_features--target-absente-de-x)
  - [`test_target_classes` — classes](#test_target_classes--classes-de-la-cible)
  - [`test_knn_training` — entraînement](#test_knn_training--entraînement-knn)
  - [`test_prediction_shape` — prédictions](#test_prediction_shape--forme-des-prédictions)
  - [`test_accuracy` — seuil de performance](#test_accuracy--accuracy-minimale)
  - [Ce qu'on teste en ML](#ce-quon-teste-en-ml)
  - [Lien avec `main.py`](#lien-avec-mainpy)

### Pipeline et résultats

- [Explication rapide](#explication-rapide)
  - [`load_data`](#load_data)
  - [train_test_split](#train_test_split)
  - [StandardScaler](#standardscaler)
  - [KNeighborsClassifier](#kneighborsclassifier)
  - [accuracy_score](#accuracy_score)
  - [classification_report](#classification_report)
  - [Corrélation](#corrélation)
- [Résultats](#résultats)
  - [Sortie de `python3 main.py`](#sortie-de-python3-mainpy)
  - [Sortie de `python3 main2.py`](#sortie-de-python3-main2py)
  - [Autre exemple `python3 main.py`](#autre-exemple-python3-mainpy)

## Structure du projet

```text
machine-learning/
├── main.py                 # Pipeline KNN + validation croisée
├── main2.py                # Exemple minimal de chargement
├── tests/
│   ├── conftest.py         # Configuration des imports pour pytest
│   └── test_main.py        # Tests unitaires (pytest)
├── pytest.ini              # pythonpath pour les imports
├── requirements.txt        # pandas, scikit-learn, pytest, …
├── README.md
├── .vscode/settings.json   # interpréteur Python du venv (optionnel)
└── projet de fin de module 2024-2025/
    └── bienetre.csv        # Jeu de données (non versionné)
```

## Installation

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt   # pandas, scikit-learn, pytest, etc.
```

Lancer le pipeline :

```bash
python main.py
```

## Tests automatisés

Le fichier [`tests/test_main.py`](tests/test_main.py) contient **6 tests** qui valident le pipeline de bout en bout : chargement → features → modèle → prédictions → métrique.

Chaque test importe `load_data` depuis [`main.py`](main.py) (même fonction que le script principal).

### Récapitulatif des 6 tests

| # | Fonction | Vérifie | Assertion clé |
|---|----------|---------|----------------|
| 1 | `test_load_data` | CSV, `X`, `y`, dimensions | `X.shape[1] == 20` et `X.shape[0] == y.shape[0]` |
| 2 | `test_target_not_in_features` | Pas de fuite de données | `"target" not in X.columns` |
| 3 | `test_target_classes` | Classification 3 classes | `set(y.unique()) == {0, 1, 2}` |
| 4 | `test_knn_training` | Pipeline s'entraîne | `model.fit(X, y)` sans erreur |
| 5 | `test_prediction_shape` | Une prédiction par ligne | `len(predictions) == len(y)` |
| 6 | `test_accuracy` | Performance minimale | `accuracy > 0.8` |

### Lancer les tests

```bash
# tous les tests, mode verbeux
pytest -v

# un seul test
pytest -v tests/test_main.py::test_load_data
```

Prérequis : le fichier `projet de fin de module 2024-2025/bienetre.csv` doit être présent localement (non versionné sur Git).

### Sortie attendue

```text
tests/test_main.py::test_load_data PASSED
tests/test_main.py::test_target_not_in_features PASSED
tests/test_main.py::test_target_classes PASSED
tests/test_main.py::test_knn_training PASSED
tests/test_main.py::test_prediction_shape PASSED
tests/test_main.py::test_accuracy PASSED

========================= 6 passed =========================
```

### `test_load_data` — chargement des données

**But :** vérifier que le CSV charge, que `X` et `y` existent, et que les dimensions sont cohérentes.

| Vérification | Détail |
|--------------|--------|
| DataFrame | `df is not None` |
| Alignement | autant de lignes dans `X` que dans `y` |
| Features | **20 colonnes** dans `X` (sans `target`) |

```python
def test_load_data():
    df, X, y = load_data("projet de fin de module 2024-2025/bienetre.csv")

    assert df is not None
    assert X.shape[0] == y.shape[0]
    assert X.shape[1] == 20
```

### `test_target_not_in_features` — target absente de X

**But :** éviter la **fuite de données** (*data leakage*).

Si `target` reste dans `X`, le modèle « triche » en voyant la réponse pendant l'entraînement. L'accuracy serait artificiellement parfaite et inutilisable en production.

```python
def test_target_not_in_features():
    _, X, _ = load_data("projet de fin de module 2024-2025/bienetre.csv")

    assert "target" not in X.columns
```

### `test_target_classes` — classes de la cible

**But :** confirmer une classification à **3 classes** : `0`, `1` et `2`.

Détecte un CSV corrompu, une colonne cible renommée ou des classes inattendues après une mise à jour des données.

```python
def test_target_classes():
    _, _, y = load_data("projet de fin de module 2024-2025/bienetre.csv")

    assert set(y.unique()) == {0, 1, 2}
```

### `test_knn_training` — entraînement KNN

**But :** le pipeline `StandardScaler` + `KNeighborsClassifier` doit s'entraîner **sans erreur**.

Même structure que dans `main.py`, avec les paramètres par défaut de scikit-learn.

```python
model = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier()),
])
model.fit(X, y)
assert model is not None
```

### `test_prediction_shape` — forme des prédictions

**But :** chaque ligne de `X` produit **exactement une** prédiction.

Contrôle la cohérence des dimensions en sortie du modèle.

```python
predictions = model.predict(X)
assert len(predictions) == len(y)
```

### `test_accuracy` — accuracy minimale

**But :** garantir un **seuil de performance** (`accuracy > 0.8`).

En production, ce type de test sert de **garde-fou** : si les données changent et dégradent le modèle, la CI échoue avant un déploiement.

```python
accuracy = accuracy_score(y, predictions)
assert accuracy > 0.8
```

> **Note :** ce test entraîne et évalue sur le **même** jeu (`X`). C'est un contrôle de non-régression rapide. Pour mesurer la généralisation, `main.py` utilise `train_test_split` et la validation croisée stratifiée.

### Ce qu'on teste en ML

| Étape | Test(s) associé(s) |
|--------|---------------------|
| Chargement des données | `test_load_data` |
| Preprocessing / pipeline | `test_knn_training` |
| Shapes (dimensions) | `test_load_data`, `test_prediction_shape` |
| Absence de fuite (`target` ∉ `X`) | `test_target_not_in_features` |
| Classes attendues | `test_target_classes` |
| Entraînement | `test_knn_training` |
| Prédictions | `test_prediction_shape` |
| Métriques minimales | `test_accuracy` |

Quand le dataset ou les features évoluent, ces tests cassent **tôt** au lieu de livrer un modèle silencieusement dégradé.

### Lien avec `main.py`

| Composant | `main.py` | Tests |
|-----------|-----------|-------|
| Chargement | `load_data()` | tests 1, 2, 3 |
| Pipeline KNN | `Pipeline([scaler, knn])` | tests 4, 5, 6 |
| Évaluation réaliste | `cross_val_score`, `train_test_split` | non couvert par les tests (volontairement) |

Les tests couvrent le **socle** ; `main.py` ajoute la validation croisée et les rapports détaillés.

## Explication rapide

### `load_data`

Fonction centrale du projet (utilisée par `main.py`, `main2.py` et les tests) :

```python
def load_data(csv_path, target_col="target"):
    df = pd.read_csv(csv_path)
    X = df.drop(columns=[target_col])  # features uniquement
    y = df[target_col]                 # cible
    return df, X, y
```

- **`X`** : 20 variables explicatives (âge, stress, activité, etc.)
- **`y`** : variable à prédire (`target`, classes 0 / 1 / 2)

### train_test_split

- `X_train, X_test, y_train, y_test`
- `train` = apprentissage,
- `test` = évaluation.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

Le paramètre :

- `test_size=0.2`

signifie :

- 20% des données → test
- 80% → entraînement

Ici :

- 80% entraînement,
- 20% test.

### StandardScaler

Très important pour KNN.

Il normalise les variables :

- âge : 20-80
- revenu : 1000-5000
- stress : 1-10

Sinon :

- les grandes valeurs dominent la distance.

### KNeighborsClassifier

- `n_neighbors=5`
- Le modèle regarde les 5 voisins les plus proches.
- `p=2`
- Distance euclidienne :

  \[ d(x, y) = \sqrt{\sum_{i=1}^n (x_i - y_i)^2} \]

### accuracy_score

Exemple :

- Accuracy : 0.91
- → 91% des prédictions sont correctes.

### classification_report

Affiche :

- précision,
- recall,
- f1-score,
- support.

Très utilisé en ML.

### Corrélation

La corrélation est calculée ici :

```python
print("\nCorrélation avec target :")
print(df.corr()["target"].sort_values(ascending=False))
```

Plus précisément :

- `df.corr()`

calcule la corrélation entre toutes les colonnes numériques.

Puis :

- `["target"]`

récupère seulement la corrélation de chaque colonne avec target.

Puis :

- `.sort_values(ascending=False)`

classe les résultats du plus corrélé au moins corrélé.

Donc par exemple :

- `risque  0.916`

veut dire : quand risque augmente, target a fortement tendance à augmenter.

Et :

- `activite  -0.885`

veut dire : quand activite augmente, target a fortement tendance à diminuer.

## Résultats

### Sortie de `python3 main.py`

```text
Dimensions X : (10000, 20)
Dimensions y : (10000,)

Répartition des classes :
target
1    4000
0    4000
2    2000
Name: count, dtype: int64

Scores cross-validation :
[1.     0.9995 1.     0.9985 0.9995]

Accuracy moyenne CV : 0.9995
Écart-type CV : 0.0005477225575051464

Accuracy test : 0.999

Classification Report :
              precision    recall  f1-score   support

           0       1.00      1.00      1.00       800
           1       1.00      1.00      1.00       800
           2       1.00      1.00      1.00       400

    accuracy                           1.00      2000
   macro avg       1.00      1.00      1.00      2000
weighted avg       1.00      1.00      1.00      2000


Matrice de confusion :
[[800   0   0]
 [  1 798   1]
 [  0   0 400]]

Corrélation avec target :
target          1.000000
risque          0.916442
stress          0.830519
cholesterol     0.754946
imc             0.752104
experience      0.598300
pression        0.597060
age             0.592656
nb_enfants      0.584322
poids           0.457818
taille         -0.143555
depenses       -0.351728
sommeil        -0.590884
exercice       -0.740145
revenu         -0.747124
education      -0.749609
satisfaction   -0.831475
bienetre       -0.831849
alimentation   -0.832196
sante          -0.873860
activite       -0.885007
Name: target, dtype: float64
```

### Sortie de `python3 main2.py`

```text
         age      taille      poids  ...     risque      sante  bienetre
0  76.378818  172.472900  93.637574  ...  46.007184  71.051570  4.829417
1  39.629339  162.527782  88.071856  ...  42.880652  64.900618  5.994971
2  36.506832  196.457696  85.160759  ...  23.152283  87.163500  8.123009
3  50.016190  174.829299  72.362368  ...  51.413594  72.278489  6.326265
4  60.743330  171.536491  54.874865  ...  47.531057  72.340850  4.493604

[5 rows x 20 columns]
0    1
1    1
2    0
3    1
4    1
Name: target, dtype: int64
```

### Autre exemple `python3 main.py`

```text
Accuracy : 1.0

Classification Report :
              precision    recall  f1-score   support

           0       1.00      1.00      1.00       808
           1       1.00      1.00      1.00       776
           2       1.00      1.00      1.00       416

    accuracy                           1.00      2000
   macro avg       1.00      1.00      1.00      2000
weighted avg       1.00      1.00      1.00      2000
```
