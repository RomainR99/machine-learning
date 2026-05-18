import pandas as pd

# Charger CSV
df = pd.read_csv("projet de fin de module 2024-2025/bienetre.csv")

print(df.head())
print(df.info())



from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


# Variable cible
y = df.iloc[:, 0]

# Variables explicatives
X = df.iloc[:, 1:]

# Séparation train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Modèle KNN
model = KNeighborsClassifier(
    n_neighbors=5,
    p=20,
    metric='minkowski'
)

# Entraînement
model.fit(X_train, y_train)

# Prédictions
predictions = model.predict(X_test)

print(predictions)

from sklearn.metrics import acc