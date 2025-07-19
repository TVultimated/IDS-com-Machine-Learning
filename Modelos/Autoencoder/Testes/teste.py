import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam

# === Carregar dados ===
df = pd.read_csv('../../../Dataset/cleaned_ids2018.csv')
X = df.drop('Label', axis=1).values
y = df['Label'].values
y = (y != 1).astype(int)  # 0 = benigno, 1 = anómalo

# === Normalizar ===
scaler = StandardScaler()
X = scaler.fit_transform(X)

# === Dividir treino/teste ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.7, random_state=42, stratify=y
)

# === Treinar apenas com dados normais ===
X_train_norm = X_train[y_train == 0]

# === Arquitetura do Autoencoder ===
input_dim = X_train.shape[1]
input_layer = Input(shape=(input_dim,))
encoded = Dense(32, activation='relu')(input_layer)
decoded = Dense(input_dim, activation='linear')(encoded)

autoencoder = Model(inputs=input_layer, outputs=decoded)
autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

# === Treinar o modelo ===
autoencoder.fit(X_train_norm, X_train_norm,
                epochs=20,
                batch_size=128,
                shuffle=True,
                verbose=1)

# === Reconstrução e cálculo do erro ===
X_test_pred = autoencoder.predict(X_test)
reconstruction_error = np.mean(np.square(X_test - X_test_pred), axis=1)

# === Threshold automático: percentil 95 dos erros de treino normal ===
train_pred = autoencoder.predict(X_train_norm)
train_error = np.mean(np.square(X_train_norm - train_pred), axis=1)
threshold = np.percentile(train_error, 95)

# === Classificação final ===
y_pred = (reconstruction_error > threshold).astype(int)

# === Avaliação ===
print("=== Autoencoder - Teste 1 (Baseline) ===")
print(classification_report(y_test, y_pred, target_names=["normal", "anomaly"]))
print("Matriz de Confusão:")
print(confusion_matrix(y_test, y_pred))
print(f"F1-score: {f1_score(y_test, y_pred):.2f}")
