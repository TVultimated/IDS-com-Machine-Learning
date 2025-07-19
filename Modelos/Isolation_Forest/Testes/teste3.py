import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# === Carregar dados ===
df = pd.read_csv('../../../Dataset/cleaned_ids2018.csv')
X = df.drop('Label', axis=1).values
y = df['Label'].values

# === Converter rótulos: 0 = benigno, 1 = anómalo ===
y = (y != 1).astype(int)

# === Normalizar ===
scaler = StandardScaler()
X = scaler.fit_transform(X)

# === Remover colunas com baixa variância ===
vt = VarianceThreshold(threshold=0.01)
X = vt.fit_transform(X)

# === Redução de dimensionalidade com PCA ===
pca = PCA(n_components=0.95, random_state=42)
X = pca.fit_transform(X)

# === Dividir treino/teste ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.70, random_state=42, stratify=y
)

# === Treinar Isolation Forest ===
iso_forest = IsolationForest(n_estimators=100, contamination='auto', random_state=42)
iso_forest.fit(X_train)

# === Prever (retorna 1 para normal, -1 para anomalia) ===
y_pred_raw = iso_forest.predict(X_test)
y_pred = np.where(y_pred_raw == 1, 0, 1)

# === Avaliação ===
print("=== Isolation Forest - Teste 3 ===")
print(classification_report(y_test, y_pred, target_names=["normal", "anomaly"]))
print("Matriz de Confusão:")
print(confusion_matrix(y_test, y_pred))
