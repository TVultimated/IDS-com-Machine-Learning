import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
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

# === Remover features com baixa variância ===
vt = VarianceThreshold(threshold=0.01)
X = vt.fit_transform(X)

# === PCA com >95% variância explicada ===
pca = PCA(n_components=0.95, random_state=42)
X = pca.fit_transform(X)

# === Dividir treino/teste ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.70, random_state=42, stratify=y
)

# === K-Means com otimizações ===
kmeans = KMeans(
    n_clusters=2,
    init='k-means++',
    n_init=100,
    max_iter=500,
    random_state=42
)
kmeans.fit(X_train)

# === Prever ===
y_pred = kmeans.predict(X_test)

# === Alinhar clusters (assumir cluster 0 = normal) ===
if np.mean(y_test[y_pred == 0]) > 0.5:
    y_pred = 1 - y_pred

# === Avaliação ===
print("=== K-Means Otimizado ===")
print(classification_report(y_test, y_pred, target_names=["normal", "anomaly"]))
print("Matriz de Confusão:")
print(confusion_matrix(y_test, y_pred))
