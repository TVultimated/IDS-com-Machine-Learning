import numpy as np
import pandas as pd
import joblib
import time
import os
from datetime import datetime
import sys
import matplotlib.pyplot as plt
from colorama import init, Fore
from tensorflow.keras.models import load_model

sys.path.append("..")
from mapping import FEATURE_MAP

# === Inicializar colorama ===
init(autoreset=True)

# === Configurações ===
FOLDER_PATH = "../Flow_outputs"
CHECK_INTERVAL = 1

# === Pastas ===
RELATORIO_FOLDER = os.path.join("Relatórios", "unsupervised")
EVAL_FOLDER = os.path.join("Avaliações", "unsupervised")
os.makedirs(RELATORIO_FOLDER, exist_ok=True)
os.makedirs(EVAL_FOLDER, exist_ok=True)

# === Índice inicial ===
i = 1
while os.path.exists(os.path.join(RELATORIO_FOLDER, f"relatorio_{i}.txt")):
    i += 1

# === Carregar modelos e scalers ===
ae = load_model("../../Modelos/Autoencoder/ae_model.keras")
scaler_ae = joblib.load("../../Modelos/Autoencoder/scaler_ae.pkl")
with open("../../Modelos/Autoencoder/ae_threshold.txt", "r") as f:
    VALIDACAO_MSE_LIMIAR = float(f.readlines()[1].split(":")[1])

if_model = joblib.load("../../Modelos/Isolation_Forest/if_model.pkl")
scaler_if = joblib.load("../../Modelos/Isolation_Forest/scaler_if.pkl")
vt_if = joblib.load("../../Modelos/Isolation_Forest/variance_if.pkl")
pca_if = joblib.load("../../Modelos/Isolation_Forest/pca_if.pkl")

km_model = joblib.load("../../Modelos/K-means/km_model.pkl")
scaler_km = joblib.load("../../Modelos/K-means/scaler_km.pkl")

# === Carregar colunas esperadas ===
with open("../features.txt", "r") as f:
    ordered_cols = [line.strip() for line in f if line.strip()]

# === Histórico ===
processed_files = set()

print(Fore.CYAN + "[INFO] IDS com Autoencoder, Isolation Forest e K-Means a correr. À espera de fluxos...")

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

                # === Autoencoder
                x_ae = scaler_ae.transform(df_filtrado)
                reconstructions = ae.predict(x_ae, verbose=0)
                mses = np.mean(np.square(x_ae - reconstructions), axis=1)
                ae_alertas = np.sum(mses >= VALIDACAO_MSE_LIMIAR)
                alertas_all["Autoencoder"] = (ae_alertas, {"Suspeito": ae_alertas, "Benigno": len(df) - ae_alertas})

                # === Isolation Forest
                x_if = scaler_if.transform(df_filtrado)
                x_if = vt_if.transform(x_if)
                x_if = pca_if.transform(x_if)
                preds_if = if_model.predict(x_if)
                if_alertas = np.sum(preds_if == -1)
                alertas_all["Isolation Forest"] = (if_alertas, {"Suspeito": if_alertas, "Benigno": len(df) - if_alertas})

                # === K-Means
                x_km = scaler_km.transform(df_filtrado)
                preds_km = km_model.predict(x_km)
                km_alertas = np.sum(preds_km != 0)
                alertas_all["K-Means"] = (km_alertas, {"Suspeito": km_alertas, "Benigno": len(df) - km_alertas})

                # === Relatório
                report_file = os.path.join(RELATORIO_FOLDER, f"relatorio_{i}.txt")
                with open(report_file, "w", encoding="utf-8") as f:
                    f.write(f"Relatório gerado em: {timestamp}\n")
                    f.write(f"Ficheiro analisado: {fname}\n")
                    for name, (n_alertas, classes) in alertas_all.items():
                        f.write(f"{name}: {n_alertas} fluxos suspeitos em {len(df)}\n")
                        f.write(f"Classes detetadas: {classes}\n")

                # === Gráfico completo por ficheiro
                fig, axs = plt.subplots(2, 1, figsize=(10, 8))

                modelos = list(alertas_all.keys())
                valores = [alertas_all[m][0] for m in modelos]
                axs[0].bar(modelos, valores, color=['blue', 'green', 'red'])
                axs[0].set_ylabel('N.º de fluxos suspeitos')
                axs[0].set_title(f'Comparação dos Modelos - {fname}')
                axs[0].set_ylim(0)

                labels = []
                counts = []
                for name, (_, classes) in alertas_all.items():
                    for cl, ct in classes.items():
                        labels.append(f"{name}-{cl}")
                        counts.append(ct)

                axs[1].bar(labels, counts, alpha=0.7, color='grey')
                axs[1].set_ylabel('N.º de fluxos')
                axs[1].set_title('Categorias por modelo')
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
