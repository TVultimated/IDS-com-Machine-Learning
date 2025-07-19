import numpy as np
import pandas as pd
import joblib
import time
import os
from datetime import datetime
import sys
import matplotlib.pyplot as plt
from colorama import init, Fore

sys.path.append("..")
from mapping import FEATURE_MAP, CLASS_NAMES

# === Inicializar colorama ===
init(autoreset=True)

# === Configurações ===
FOLDER_PATH = "../Flow_outputs"
CHECK_INTERVAL = 1

# === Pastas ===
RELATORIO_FOLDER = os.path.join("Relatórios", "supervised")
EVAL_FOLDER = os.path.join("Avaliações", "supervised")
os.makedirs(RELATORIO_FOLDER, exist_ok=True)
os.makedirs(EVAL_FOLDER, exist_ok=True)

# === Índice inicial ===
i = 1
while os.path.exists(os.path.join(RELATORIO_FOLDER, f"relatorio_{i}.txt")):
    i += 1

# === Carregar modelos e scalers ===
mlp = joblib.load("../../Modelos/Rede_Neuronal/mlp.pkl")
scaler_mlp = joblib.load("../../Modelos/Rede_Neuronal/scaler_mlp.pkl")

rf = joblib.load("../../Modelos/Random_Forest/rf.pkl")
scaler_rf = joblib.load("../../Modelos/Random_Forest/scaler_rf.pkl")

svm = joblib.load("../../Modelos/SVM/svm.pkl")
scaler_svm = joblib.load("../../Modelos/SVM/scaler_svm.pkl")

# === Carregar colunas esperadas ===
with open("../features.txt", "r") as f:
    ordered_cols = [line.strip() for line in f if line.strip()]

# === Histórico ===
processed_files = set()

print(Fore.CYAN + "[INFO] IDS com MLP, RF e SVM a correr. À espera de fluxos...")

def classes_detectadas(preds):
    suspeitos = preds[preds != 1]
    unicos, contagens = np.unique(suspeitos, return_counts=True)
    return {CLASS_NAMES.get(u, str(u)): int(c) for u, c in zip(unicos, contagens)}

try:
    while True:
        files = [f for f in os.listdir(FOLDER_PATH) if f.endswith(".csv")]
        for fname in sorted(files):
            fpath = os.path.join(FOLDER_PATH, fname)
            if fpath in processed_files:
                continue

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            df = pd.read_csv(fpath)
            processed_files.add(fpath)

            print(f"\n{Fore.CYAN}[INFO] Ficheiro recebido: {fname} ({len(df)} fluxos)\n")

            df.rename(columns=FEATURE_MAP, inplace=True)
            missing = [col for col in ordered_cols if col not in df.columns]
            if missing:
                print(Fore.RED + f"[ERRO - Construção] Faltam colunas: {missing}")
                continue

            df_filtrado = df[ordered_cols].copy()

            try:
                alertas_all = {}

                # === MLP
                x_mlp = scaler_mlp.transform(df_filtrado)
                preds_mlp = mlp.predict(x_mlp)
                n_mlp = np.sum(preds_mlp != 1)
                cl_mlp = classes_detectadas(preds_mlp)
                alertas_all['MLP'] = (n_mlp, cl_mlp)

                # === RF
                x_rf = scaler_rf.transform(df_filtrado)
                preds_rf = rf.predict(x_rf)
                n_rf = np.sum(preds_rf != 1)
                cl_rf = classes_detectadas(preds_rf)
                alertas_all['Random Forest'] = (n_rf, cl_rf)

                # === SVM
                x_svm = scaler_svm.transform(df_filtrado)
                preds_svm = svm.predict(x_svm)
                n_svm = np.sum(preds_svm != 1)
                cl_svm = classes_detectadas(preds_svm)
                alertas_all['SVM'] = (n_svm, cl_svm)

                # === Relatório (opcional)
                report_file = os.path.join(RELATORIO_FOLDER, f"relatorio_{i}.txt")
                with open(report_file, "w", encoding="utf-8") as f:
                    f.write(f"Relatório gerado em: {timestamp}\n")
                    f.write(f"Ficheiro analisado: {fname}\n")
                    for name, (n_alertas, classes) in alertas_all.items():
                        f.write(f"{name}: {n_alertas} fluxos suspeitos em {len(df_filtrado)}\n")
                        f.write(f"Classes detetadas: {classes}\n")

                # === Gráfico completo por ficheiro
                fig, axs = plt.subplots(2, 1, figsize=(8, 8))

                # Gráfico de barras (nº de fluxos suspeitos)
                modelos = list(alertas_all.keys())
                valores = [alertas_all[m][0] for m in modelos]
                axs[0].bar(modelos, valores, color=['blue', 'green', 'red'])
                axs[0].set_ylabel('N.º de fluxos suspeitos')
                axs[0].set_title(f'Comparação dos Modelos - {fname}')
                axs[0].set_ylim(0)

                # Gráfico de classes detetadas
                for name, (_, classes) in alertas_all.items():
                    if classes:
                        labels = list(classes.keys())
                        counts = list(classes.values())
                        axs[1].bar([f"{name}-{l}" for l in labels], counts, alpha=0.7)

                axs[1].set_ylabel('N.º de fluxos por classe')
                axs[1].set_title('Classes detetadas por modelo')
                axs[1].tick_params(axis='x', rotation=45)

                plt.tight_layout()
                grafico_path = os.path.join(EVAL_FOLDER, f"grafico_completo_{i}.png")
                plt.savefig(grafico_path)
                plt.close()

                print(Fore.CYAN + f"[INFO] Gráfico salvo como {grafico_path}")

                i += 1

            except Exception as e:
                print(Fore.RED + f"[ERRO - Previsão] {e}")
                with open(os.path.join(RELATORIO_FOLDER, f"relatorio_{i}.txt"), "w", encoding="utf-8") as f:
                    f.write(f"{timestamp} | Ficheiro: {fname} | ERRO: {e}\n")
                i += 1

        time.sleep(CHECK_INTERVAL)

except KeyboardInterrupt:
    print(Fore.CYAN + "\n[INFO] Interrupção detetada. Execução terminada.")
