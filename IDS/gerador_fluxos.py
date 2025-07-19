import pandas as pd
import numpy as np
import time
from datetime import datetime
import os

# === Configurações ===
DATASET_PATH = '../Dataset/cleaned_ids2018.csv'
OUTPUT_FOLDER = 'Flow_outputs'
INTERVALO = 10

# === Carregar dados ===
df = pd.read_csv(DATASET_PATH)

print("[INFO] Gerador de fluxos iniciado.")
while True:
    fluxo = df.sample(1, random_state=np.random.randint(10000))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    nome_ficheiro = f"fluxo_{timestamp}.csv"
    caminho = os.path.join(OUTPUT_FOLDER, nome_ficheiro)
    fluxo.to_csv(caminho, index=False)
    print(f"[INFO] Ficheiro gerado: {nome_ficheiro}")
    time.sleep(INTERVALO)
