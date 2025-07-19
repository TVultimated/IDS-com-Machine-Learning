import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix

# carregar e preparar dados
df = pd.read_csv('../../../Dataset/cleaned_ids2018.csv')
X = df.drop('Label', axis=1).values
y = df['Label'].values

# normalizar
scaler = StandardScaler()
X = scaler.fit_transform(X)

# dividir treino/teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.70, random_state=42, stratify=y
)

# MLP melhorado
mlp = MLPClassifier(
    hidden_layer_sizes=(100, 50),
    activation='relu',
    solver='adam',
    alpha=1e-4,
    learning_rate='adaptive',
    learning_rate_init=1e-3,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=10,
    max_iter=200,
    random_state=42
)

# treinar
mlp.fit(X_train, y_train)

# avaliar
y_pred = mlp.predict(X_test)
print(classification_report(y_test, y_pred))
print("Matriz de Confusão:")
print(confusion_matrix(y_test, y_pred))