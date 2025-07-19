import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

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

rf = RandomForestClassifier(
    n_estimators=1000,
    max_depth=40,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features=0.3,
    bootstrap=True,
    criterion='entropy',
    oob_score=True,
    class_weight='balanced_subsample',
    random_state=42,
    n_jobs=-1,
    warm_start=False
)

# treinar
rf.fit(X_train, y_train)

# avaliar
y_pred = rf.predict(X_test)
print(classification_report(y_test, y_pred))
print("Matriz de Confusão:")
print(confusion_matrix(y_test, y_pred))
print(f"OOB-score: {rf.oob_score_:.4f}")