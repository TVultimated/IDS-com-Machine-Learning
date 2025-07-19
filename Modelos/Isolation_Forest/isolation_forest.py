import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, f1_score, precision_score, recall_score
from sklearn.ensemble import IsolationForest
import seaborn as sns
import joblib
import os

# === Criar diretório de avaliações ===
os.makedirs("Avaliações", exist_ok=True)

# === Carregar dados ===
df = pd.read_csv('../../Dataset/cleaned_ids2018.csv')
X = df.drop(['Label'], axis=1).values
y = df['Label'].values

# === Converter rótulos: 0 = benigno, 1 = anómalo ===
y = (y != 1).astype(int)

# === Normalizar ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === Remover baixa variância ===
vt = VarianceThreshold(threshold=0.01)
X_vt = vt.fit_transform(X_scaled)

# === Reduzir dimensionalidade com PCA (95% variância) ===
pca = PCA(n_components=0.95, random_state=42)
X_pca = pca.fit_transform(X_vt)

# === Dividir treino/teste ===
X_train, X_test, y_train, y_test = train_test_split(
    X_pca, y, train_size=0.70, random_state=42, stratify=y
)

# === Treinar Isolation Forest ===
iso_forest = IsolationForest(n_estimators=100, contamination='auto', random_state=42)
iso_forest.fit(X_train)
y_pred_raw = iso_forest.predict(X_test)
y_pred = np.where(y_pred_raw == 1, 0, 1)

# === Avaliação ===
print("=== Isolation Forest - Avaliação ===")
print(classification_report(y_test, y_pred, target_names=["normal", "anomaly"]))
print(f"F1-score: {f1_score(y_test, y_pred):.2f}")
print(f"Precision: {precision_score(y_test, y_pred):.2f}")
print(f"Recall: {recall_score(y_test, y_pred):.2f}")

# === Guardar relatório ===
with open("Avaliações/isolation_forest_evaluation.txt", "w") as f:
    f.write("=== Isolation Forest - Avaliação ===\n")
    f.write(classification_report(y_test, y_pred, target_names=["normal", "anomaly"]))
    f.write(f"\nF1-score: {f1_score(y_test, y_pred):.2f}\n")
    f.write(f"Precision: {precision_score(y_test, y_pred):.2f}\n")
    f.write(f"Recall: {recall_score(y_test, y_pred):.2f}\n")

# === Matriz de confusão ===
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["normal", "anomaly"], yticklabels=["normal", "anomaly"])
plt.title("Confusion Matrix - Isolation Forest")
plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.tight_layout()
plt.savefig("Avaliações/confusion_matrix_isolation_forest.png")
plt.close()

# === Gráfico F1-score ===
f1 = f1_score(y_test, y_pred) * 100
plt.figure(figsize=(5, 5))
plt.bar(["Isolation Forest"], [f1])
plt.title("F1-score - Isolation Forest")
plt.ylim([50, 100])
plt.ylabel("F1-score (%)")
plt.tight_layout()
plt.savefig("Avaliações/f1_score_isolation_forest.png")
plt.close()

# === Guardar modelo e transformadores ===
joblib.dump(iso_forest, 'if_model.pkl')
joblib.dump(scaler, 'scaler_if.pkl')
joblib.dump(vt, 'variance_if.pkl')
joblib.dump(pca, 'pca_if.pkl')
