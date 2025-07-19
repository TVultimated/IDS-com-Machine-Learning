import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
from tensorflow.keras.models import load_model
import joblib
import os

# === Carregar modelo e scaler ===
autoencoder = load_model("../ae_model.keras")
scaler = joblib.load("../scaler_ae.pkl")

# === Carregar dados ===
df = pd.read_csv("../../../Dataset/cleaned_ids2018.csv")
X = df.drop('Label', axis=1).values
y = (df['Label'].values != 1).astype(int)  # 0 = normal, 1 = anomalia

# === Normalizar ===
X = scaler.transform(X)

# === Dividir treino/teste ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.7, random_state=42, stratify=y
)

# === Apenas normais para treino
X_train_norm = X_train[y_train == 0]

# === Reconstrução e erros
train_pred = autoencoder.predict(X_train_norm)
X_test_pred = autoencoder.predict(X_test)
test_errors = np.mean(np.square(X_test - X_test_pred), axis=1)

# === Testar thresholds (MSE)
thresholds = [50, 75, 100, 125, 150, 200, 300, 500, 1000]
resultados = []

melhor_f1 = 0
melhor_resultado = {}

for th in thresholds:
    y_pred = [1 if err >= th else 0 for err in test_errors]

    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    resultados.append({
        "Threshold": th,
        "F1-score": f1,
        "Precision": precision,
        "Recall": recall
    })

    if f1 > melhor_f1:
        melhor_f1 = f1
        melhor_resultado = resultados[-1]

# === Mostrar resultados
print("\n{:<10} | {:<8} | {:<9} | {:<7}".format("Threshold", "F1-score", "Precision", "Recall"))
print("-" * 40)
for r in resultados:
    print("{:<10} | {:<8.4f} | {:<9.4f} | {:<7.4f}".format(
        r["Threshold"], r["F1-score"], r["Precision"], r["Recall"]
    ))

print("\n=== Melhor Threshold ===")
for k, v in melhor_resultado.items():
    print(f"{k}: {v:.4f}")

# === Guardar resultados
os.makedirs("../Avaliações", exist_ok=True)
pd.DataFrame(resultados).to_csv("../Avaliações/ae_resultados_thresholds.csv", index=False)

with open("../ae_threshold.txt", "w") as f:
    f.write("=== Melhor threshold baseado em F1-score ===\n")
    for k, v in melhor_resultado.items():
        f.write(f"{k}: {v:.6f}\n")
