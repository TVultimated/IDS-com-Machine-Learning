import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

# === Carregar dados ===
df = pd.read_csv('../../../Dataset/cleaned_ids2018.csv')
X = df.drop('Label', axis=1).values
y = df['Label'].values

# === Normalizar ===
scaler = StandardScaler()
X = scaler.fit_transform(X)

# === Dividir treino/teste ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.7, random_state=42, stratify=y
)

# === SVM com parâmetros ajustados ===
svm = SVC(C=5.0, kernel='rbf', gamma=0.01)
svm.fit(X_train, y_train)

# === Avaliar ===
y_pred = svm.predict(X_test)
print("=== SVM - Teste 3 ===")
print(classification_report(y_test, y_pred))
print("Matriz de Confusão:")
print(confusion_matrix(y_test, y_pred))
