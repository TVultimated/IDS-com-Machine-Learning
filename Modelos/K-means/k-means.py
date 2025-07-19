import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import seaborn as sns
import joblib

# === Carregar dados ===
df = pd.read_csv('../../Dataset/cleaned_ids2018.csv')
X = df.drop(['Label'], axis=1).values
y = df['Label'].values

# === Converter rótulos: 0 = benigno, 1 = anómalo ===
y = (y != 1).astype(int)

# === Normalização ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === Dividir treino/teste ===
x_train, x_test, y_train, y_test = train_test_split(
    X_scaled, y, train_size=0.70, random_state=42, stratify=y
)

# === K-Means ===
kmeans_model = KMeans(n_clusters=2, random_state=42, n_init='auto')
kmeans_model.fit(x_train)
kmeans_preds = kmeans_model.predict(x_test)

# === Alinhar clusters ===
if np.mean(y_test[kmeans_preds == 0]) > 0.5:
    kmeans_preds = 1 - kmeans_preds

# === Avaliação ===
print("\n=== K-Means - Avaliação ===")
print(classification_report(y_test, kmeans_preds, target_names=["normal", "anomaly"]))
print(f"F1-score: {f1_score(y_test, kmeans_preds):.2f}")

# === Guardar relatório ===
with open("Avaliações/kmeans_evaluation.txt", "w") as f:
    f.write("=== K-Means - Avaliação ===\n")
    f.write(classification_report(y_test, kmeans_preds, target_names=["normal", "anomaly"]))
    f.write(f"\nF1-score: {f1_score(y_test, kmeans_preds):.2f}\n")

# === Matriz de confusão ===
cm = confusion_matrix(y_test, kmeans_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["normal", "anomaly"], yticklabels=["normal", "anomaly"])
plt.title("Confusion Matrix - K-Means")
plt.xlabel("Classe Prevista")
plt.ylabel("Classe Real")
plt.tight_layout()
plt.savefig("Avaliações/confusion_matrix_kmeans.png")
plt.show()

# === Gráfico de F1-score ===
f1 = f1_score(y_test, kmeans_preds) * 100
plt.figure(figsize=(5, 5))
plt.bar(["K-Means"], [f1])
plt.title("F1-score")
plt.ylim([50, 100])
plt.ylabel("F1-score (%)")
plt.tight_layout()
plt.savefig("Avaliações/f1_score_kmeans.png")
plt.show()

# === Guardar modelo e transformadores ===
joblib.dump(kmeans_model, 'km_model.pkl')
joblib.dump(scaler, 'scaler_km.pkl')
