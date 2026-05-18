# Pipeline ML propre

Voici un pipeline complet propre pour :

- charger les données,
- faire un train_test_split,
- normaliser,
- entraîner le KNN,
- prédire,
- calculer l’accuracy.

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

## Résultats

Exemple de sortie de `python3 main2.py` :

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

Exemple de sortie de `python3 main.py` :

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
