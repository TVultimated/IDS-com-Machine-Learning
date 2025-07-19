import numpy as np
import pandas as pd
import joblib
import time
import os
from datetime import datetime
from tensorflow.keras.models import load_model
from mapping import FEATURE_MAP, CLASS_NAMES
from colorama import init, Fore, Style

# === Inicializar colorama ===
init(autoreset=True)

# === Configurações ===
FOLDER_PATH = "Flow_outputs"
CHECK_INTERVAL = 1

# === Relatórios ===
REPORT_FOLDER = "Relatórios"
os.makedirs(REPORT_FOLDER, exist_ok=True)
i = 1
while os.path.exists(os.path.join(REPORT_FOLDER, f"relatorio_{i}.txt")):
    i += 1
report_file = os.path.join(REPORT_FOLDER, f"relatorio_{i}.txt")

# === Carregar modelos e scalers ===
mlp = joblib.load("../Modelos/Rede_Neuronal/mlp.pkl")
scaler_mlp = joblib.load("../Modelos/Rede_Neuronal/scaler_mlp.pkl")

ae = load_model("../Modelos/Autoencoder/ae_model.keras")
scaler_ae = joblib.load("../Modelos/Autoencoder/scaler_ae.pkl")
# Carregar Threshold
with open("../Modelos/Autoencoder/ae_threshold.txt", "r") as f:
    VALIDACAO_MSE_LIMIAR = float(f.readlines()[1].split(":")[1]) 

# === Carregar colunas esperadas ===
with open("features.txt", "r") as f:
    ordered_cols = [line.strip() for line in f if line.strip()]

# === Histórico ===
processed_files = set()

print(Fore.CYAN + "[INFO] IDS a correr. À espera de fluxos...")

while True:
    try:
        files = [f for f in os.listdir(FOLDER_PATH) if f.endswith(".csv")]
        for fname in sorted(files):
            fpath = os.path.join(FOLDER_PATH, fname)
            if fpath in processed_files:
                continue

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            df = pd.read_csv(fpath)
            processed_files.add(fpath)

            print(f"\n{Fore.CYAN}[INFO] Ficheiro recebido: {fname} ({len(df)} fluxos)\n")

            # === Aplicar mapeamento de colunas ===
            df.rename(columns=FEATURE_MAP, inplace=True)

            # === Verificar colunas
            missing = [col for col in ordered_cols if col not in df.columns]
            if missing:
                print(Fore.RED + f"[ERRO - Construção] Faltam colunas: {missing}")
                continue

            df_filtrado = df[ordered_cols].copy()

            try:
                # === Etapa 1: MLP (Supervisionado) ===
                print(Fore.YELLOW + "» [MLP] A analisar ficheiro completo com Rede Neuronal...")
                x_mlp = scaler_mlp.transform(df_filtrado.values)
                preds_mlp = mlp.predict(x_mlp)

                num_alertas = np.sum(preds_mlp != 1)

                if num_alertas > 0:
                    print(Fore.RED + f"MLP detetou {num_alertas} fluxos anómalos num total de {len(df)}")
                    classes_alerta = preds_mlp[preds_mlp != 1]
                    unicos, contagens = np.unique(classes_alerta, return_counts=True)
                    mais_frequente = unicos[np.argmax(contagens)]
                    classe_nome = CLASS_NAMES.get(mais_frequente, "Desconhecido")

                    print(Fore.RED + f"↳ Vários fluxos classificados como '{classe_nome}' pela MLP.")
                    resultado = f"ALERT - {num_alertas} fluxos suspeitos ({classe_nome})"
                    modelo = "MLP"

                else:
                    # === Etapa 2: Autoencoder (Não supervisionado) ===
                    print(Fore.YELLOW + "» MLP não detetou ataques. [AE] A analisar ficheiro com Autoencoder...")

                    x_ae = scaler_ae.transform(df_filtrado.values)
                    reconstructions = ae.predict(x_ae, verbose=0)
                    mses = np.mean(np.power(x_ae - reconstructions, 2), axis=1)

                    fluxos_validacao = []
                    fluxos_benignos = []
                    validacoes_ataque = []

                    print("\nResultados por fluxo (Autoencoder):")
                    print("-" * 60)
                    for i, mse in enumerate(mses):
                        if mse >= VALIDACAO_MSE_LIMIAR:
                            print(Fore.RED + f"[⚠] Fluxo {i + 1:03} | MSE = {mse:.4f}  --> Suspeito")
                            fluxos_validacao.append((i + 1, mse))
                        else:
                            print(Fore.GREEN + f"[✓] Fluxo {i + 1:03} | MSE = {mse:.4f}  --> Benigno")
                            fluxos_benignos.append((i + 1, mse))

                    if fluxos_validacao:
                        print(Fore.YELLOW + "\n[VALIDAÇÃO MANUAL] Fluxos suspeitos detetados. É necessária validação por parte do utilizador.\n")
                        for linha, mse in fluxos_validacao:
                            while True:
                                resposta = input(f"Classificar fluxo {linha} (MSE = {mse:.4f}) como anomalia? (s/n): ").strip().lower()
                                if resposta == "s":
                                    validacoes_ataque.append((linha, mse))
                                    break
                                elif resposta == "n":
                                    break
                                else:
                                    print("Resposta inválida. Escreve 's' para sim ou 'n' para não.")

                        if validacoes_ataque:
                            print(Fore.RED + f"\n{len(validacoes_ataque)} fluxos validados como anomalia.")
                            resultado = f"ALERT - {len(validacoes_ataque)} fluxos confirmados"
                            modelo = "Autoencoder"

                            valid_df = df_filtrado.iloc[[linha - 1 for linha, _ in validacoes_ataque]]
                            valid_df["MSE"] = [mse for _, mse in validacoes_ataque]
                            os.makedirs("Validacoes", exist_ok=True)
                            valid_df.to_csv(f"Validacoes/validacao_confirmada_{fname}", index=False)
                        else:
                            print(Fore.GREEN + "\nNenhum fluxo confirmado como anomalia. Considerado benigno.")
                            resultado = "NO ACTION"
                            modelo = "Autoencoder"

                    else:
                        print(Fore.GREEN + "\nTodos os fluxos analisados têm MSE abaixo do limiar. Considerado benigno.")
                        resultado = "NO ACTION"
                        modelo = "Autoencoder"

                # === Output final ===
                output = f"""
{Fore.BLUE}╭────────────────────────────────────────────────────────
│ Ficheiro: {fname:<45}
│ Data/Hora: {timestamp:<40}
│ Resultado: {resultado:<38}
│ Modelo: {modelo:<44}
╰────────────────────────────────────────────────────────
"""
                print(output)

                with open(report_file, "a", encoding="utf-8") as f:
                    f.write(f"{timestamp} | Ficheiro: {fname} | {resultado} | {modelo}\n")

            except Exception as e:
                print(Fore.RED + f"[ERRO - Previsão] {e}")
                with open(report_file, "a", encoding="utf-8") as f:
                    f.write(f"{timestamp} | Ficheiro: {fname} | ERRO: {e}\n")

    except Exception as e:
        print(Fore.RED + f"[ERRO - Geral] {e}")
        with open(report_file, "a", encoding="utf-8") as f:
            f.write(f"[ERRO - Geral] {e}\n")

    time.sleep(CHECK_INTERVAL)
