import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
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

# SVM com kernel RBF e C ligeiramente ajustado
svm = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm.fit(X_train, y_train)

# avaliar
y_pred = svm.predict(X_test)
print("=== SVM - Teste 2 ===")
print(classification_report(y_test, y_pred))
print("Matriz de Confusão:")
print(confusion_matrix(y_test, y_pred))
