import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix

# carregar e preparar dados
df = pd.read_csv('../../../Dataset/cleaned_ids2018.csv')
X = df.drop('Label', axis=1).values
y = df['Label'].values

# converter rótulos para binário: 0 = benigno, 1 = anómalo
y = (y != 1).astype(int)

# normalizar
scaler = StandardScaler()
X = scaler.fit_transform(X)

# redução de dimensionalidade com PCA
pca = PCA(n_components=0.95, random_state=42)
X = pca.fit_transform(X)

# dividir treino/teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.70, random_state=42, stratify=y
)

# modelo K-Means com melhorias
kmeans = KMeans(n_clusters=2, random_state=42, n_init=20)
kmeans.fit(X_train)

# prever
y_pred = kmeans.predict(X_test)

# alinhar clusters
if np.mean(y_test[y_pred == 0]) > 0.5:
    y_pred = 1 - y_pred

# avaliar
print(classification_report(y_test, y_pred, target_names=["normal", "anomaly"]))
print("Matriz de Confusão:")
print(confusion_matrix(y_test, y_pred))
