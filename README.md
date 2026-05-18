# Pipeline ML propre

Voici un pipeline complet propre pour :

- charger les données,
- faire un train_test_split,
- normaliser,
- entraîner le KNN,
- prédire,
- calculer l’accuracy.

## Sommaire

- [Explication rapide](#explication-rapide)
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

## Explication rapide

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
