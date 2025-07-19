import os

# Pasta onde estão os ficheiros de fluxo
PASTA_FLUXOS = "Flow_outputs"

# Contador de ficheiros removidos
removidos = 0

for ficheiro in os.listdir(PASTA_FLUXOS):
    if ficheiro.endswith(".csv"):
        caminho = os.path.join(PASTA_FLUXOS, ficheiro)
        try:
            os.remove(caminho)
            removidos += 1
        except Exception as e:
            print(f"[ERRO] Não foi possível remover {ficheiro}: {e}")

print(f"[INFO] Limpeza concluída. {removidos} ficheiro(s) removido(s) da pasta '{PASTA_FLUXOS}'.")
