import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, f1_score, precision_score, recall_score
from sklearn.neural_network import MLPClassifier
import seaborn as sns
import joblib

# === Carregar dados ===
df = pd.read_csv('../../Dataset/cleaned_ids2018.csv')

# === Separar features and labels ===
X = df.drop(['Label'], axis=1)
y = df['Label']

# === Normalização ===
scaler = StandardScaler()
X = scaler.fit_transform(X)

# === Dividir treino/teste ===
x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.70, random_state=42, stratify=y)

# === MLP ===
mlp_model = MLPClassifier(
    hidden_layer_sizes=(100, 50),
    activation='relu',
    solver='adam',
    alpha=1e-4,
    learning_rate='adaptive',
    learning_rate_init=1e-3,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=10,
    max_iter=200,
    random_state=42
)
mlp_model.fit(x_train, y_train)

# === Predição com conjunto de teste ===
y_pred = mlp_model.predict(x_test)

print("\n=== MLP Classifier - Avaliação ===")
print(classification_report(y_test, y_pred))
print(f"F1-score: {f1_score(y_test, y_pred, average='weighted'):.2f}")
print(f"Precision: {precision_score(y_test, y_pred, average='weighted'):.2f}")
print(f"Recall: {recall_score(y_test, y_pred, average='weighted'):.2f}")

with open("Avaliações/mlp_evaluation.txt", "w") as f:
    f.write("=== MLP - Avaliação ===\n")
    f.write(classification_report(y_test, y_pred))
    f.write(f"\nF1-score: {f1_score(y_test, y_pred, average='weighted'):.2f}\n")
    f.write(f"Precision: {precision_score(y_test, y_pred, average='weighted'):.2f}\n")
    f.write(f"Recall: {recall_score(y_test, y_pred, average='weighted'):.2f}\n")

# === Matriz de Confusão ===
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.title("Confusion Matrix - MLP")
plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.tight_layout()
plt.savefig("Avaliações/confusion_matrix_mlp.png")
plt.show()

# === Cross-Validation ===
print("\n=== MLP - Cross-Validation ===")
with open("Avaliações/mlp_cross_validation.txt", "w") as f:
    f.write("=== MLP - Cross-Validation ===\n")
    for metric in ['precision_weighted', 'recall_weighted', 'f1_weighted']:
        scores = cross_val_score(mlp_model, X, y, cv=5, scoring=metric, n_jobs=-1)
        mean_score = np.mean(scores) * 100
        std_score = np.std(scores) * 100
        print(f"{metric}: {mean_score:.2f}% ± {std_score:.2f}%")
        f.write(f"{metric}: {mean_score:.2f}% ± {std_score:.2f}%\n")

# === F1-score plot per fold ===
f1_scores = cross_val_score(mlp_model, X, y, cv=10, scoring='f1_weighted')
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), f1_scores * 100, marker='o', linestyle='--')
plt.title('F1-score in Each Fold of Cross-Validation')
plt.xlabel('Fold')
plt.ylabel('F1-score (%)')
plt.ylim([80, 100])
plt.grid(True)
plt.tight_layout()
plt.savefig("Avaliações/f1_score_cross_validation_mlp.png")
plt.show()

# Guardar F1-scores per fold
np.savetxt("Avaliações/f1_scores_per_fold_mlp.csv", f1_scores, delimiter=",", header="F1-score per fold", comments="")

# === Guardar modelo e scaler ===
joblib.dump(mlp_model, 'mlp.pkl')
joblib.dump(scaler, 'scaler_mlp.pkl')