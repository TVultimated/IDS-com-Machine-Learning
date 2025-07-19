import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
import seaborn as sns
import joblib
import os

# === Carregar dados ===
df = pd.read_csv('../../Dataset/cleaned_ids2018.csv')
X = df.drop('Label', axis=1).values
y = (df['Label'].values != 1).astype(int)  # 0 = normal, 1 = anomaly

# === Dividir treino/teste ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.7, random_state=42, stratify=y
)

# === Apenas dados normais para treino
X_train_norm = X_train[y_train == 0]

# === Normalizar com base nos dados normais
scaler = StandardScaler()
scaler.fit(X_train_norm)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
X_train_norm = scaler.transform(X_train_norm)

# === Autoencoder ===
input_dim = X_train.shape[1]
input_layer = Input(shape=(input_dim,))
encoded = Dense(128, activation='relu')(input_layer)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(32, activation='relu')(encoded)
decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(input_dim, activation='linear')(decoded)

autoencoder = Model(inputs=input_layer, outputs=decoded)
autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

# === Treinar modelo ===
autoencoder.fit(X_train_norm, X_train_norm,
                epochs=40,
                batch_size=128,
                shuffle=True,
                verbose=1)

# === Reconstrução e erro ===
X_test_pred = autoencoder.predict(X_test)
reconstruction_error = np.mean(np.square(X_test - X_test_pred), axis=1)

# === Threshold automático baseado no treino
train_pred = autoencoder.predict(X_train_norm)
train_error = np.mean(np.square(X_train_norm - train_pred), axis=1)
threshold = np.percentile(train_error, 95)

# === Classificação
y_pred = (reconstruction_error > threshold).astype(int)

# === Avaliação
report = classification_report(y_test, y_pred, target_names=["normal", "anomaly"])
cm = confusion_matrix(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n=== Autoencoder - Avaliação ===")
print(report)
print("Matriz de Confusão:")
print(cm)
print(f"F1-score: {f1:.2f}")

# === Guardar avaliação
os.makedirs("Avaliações", exist_ok=True)
with open("Avaliações/autoencoder_evaluation.txt", "w") as f:
    f.write("=== Autoencoder - Avaliação ===\n")
    f.write(report)
    f.write(f"\nF1-score: {f1:.2f}\n")

# === Guardar matriz de confusão
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', xticklabels=["normal", "anomaly"], yticklabels=["normal", "anomaly"])
plt.title("Matriz de Confusão - Autoencoder")
plt.xlabel("Classe Prevista")
plt.ylabel("Classe Real")
plt.tight_layout()
plt.savefig("Avaliações/confusion_matrix_autoencoder.png")
plt.show()

# === Guardar modelo e scaler
autoencoder.save("ae_model.keras")
joblib.dump(scaler, "scaler_ae.pkl")