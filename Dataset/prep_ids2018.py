import pandas as pd
import numpy as np

# === Definição dos tipos de dados por coluna ===
ids_datatypes = {
    'Dst Port': np.int32,
    'Protocol': np.int8,
    'Flow Duration': np.int64,
    'Tot Fwd Pkts': np.int16,
    'Tot Bwd Pkts': np.int16,
    'TotLen Fwd Pkts': np.int32,
    'TotLen Bwd Pkts': np.int32,
    'Fwd Pkt Len Max': np.int32,
    'Fwd Pkt Len Min': np.int32,
    'Fwd Pkt Len Mean': np.float64,
    'Fwd Pkt Len Std': np.float64,
    'Bwd Pkt Len Max': np.int16,
    'Bwd Pkt Len Min': np.int16,
    'Bwd Pkt Len Mean': np.float64,
    'Bwd Pkt Len Std': np.float64,
    'Flow Byts/s': np.float64,
    'Flow Pkts/s': np.float64,
    'Flow IAT Mean': np.float64,
    'Flow IAT Std': np.float64,
    'Flow IAT Max': np.int64,
    'Flow IAT Min': np.int32,
    'Fwd IAT Tot': np.int32,
    'Fwd IAT Mean': np.float32,
    'Fwd IAT Std': np.float64,
    'Fwd IAT Max': np.int32,
    'Fwd IAT Min': np.int32,
    'Bwd IAT Tot': np.int32,
    'Bwd IAT Mean': np.float64,
    'Bwd IAT Std': np.float64,
    'Bwd IAT Max': np.int64,
    'Bwd IAT Min': np.int64,
    'Fwd PSH Flags': np.int8,
    'Bwd PSH Flags': np.int8,
    'Fwd URG Flags': np.int8,
    'Bwd URG Flags': np.int8,
    'Fwd Header Len': np.int32,
    'Bwd Header Len': np.int32,
    'Fwd Pkts/s': np.float64,
    'Bwd Pkts/s': np.float64,
    'Pkt Len Min': np.int16,
    'Pkt Len Max': np.int32,
    'Pkt Len Mean': np.float64,
    'Pkt Len Std': np.float64,
    'Pkt Len Var': np.float64,
    'FIN Flag Cnt': np.int8,
    'SYN Flag Cnt': np.int8,
    'RST Flag Cnt': np.int8,
    'PSH Flag Cnt': np.int8,
    'ACK Flag Cnt': np.int8,
    'URG Flag Cnt': np.int8,
    'CWE Flag Count': np.int8,
    'ECE Flag Cnt': np.int8,
    'Pkt Size Avg': np.float32,
    'Fwd Seg Size Avg': np.float32,
    'Bwd Seg Size Avg': np.float32,
    'Fwd Byts/b Avg': np.int8,
    'Fwd Pkts/b Avg': np.int8,
    'Fwd Blk Rate Avg': np.int8,
    'Bwd Byts/b Avg': np.int8,
    'Bwd Pkts/b Avg': np.int8,
    'Bwd Blk Rate Avg': np.int8,
    'Subflow Fwd Pkts': np.int16,
    'Subflow Fwd Byts': np.int32,
    'Subflow Bwd Pkts': np.int16,
    'Subflow Bwd Byts': np.int32,
    'Init Fwd Win Byts': np.int32,
    'Init Bwd Win Byts': np.int32,
    'Fwd Act Data Pkts': np.int16,
    'Fwd Seg Size Min': np.int8,
    'Active Mean': np.float64,
    'Active Std': np.float64,
    'Active Max': np.int32,
    'Active Min': np.int32,
    'Idle Mean': np.float64,
    'Idle Std': np.float64,
    'Idle Max': np.int64,
    'Idle Min': np.int64,
    'Label': object
}

used_cols = list(ids_datatypes.keys())

# === Carregamento dos ficheiros ===
df1 = pd.read_csv("../Dataset/CSE-CIC-IDS2018/02-14-2018.csv", dtype=ids_datatypes, usecols=used_cols)
df2 = pd.read_csv("../Dataset/CSE-CIC-IDS2018/02-15-2018.csv", dtype=ids_datatypes, usecols=used_cols)
df5 = pd.read_csv("../Dataset/CSE-CIC-IDS2018/02-21-2018.csv", dtype=ids_datatypes, usecols=used_cols)
df6 = pd.read_csv("../Dataset/CSE-CIC-IDS2018/02-22-2018.csv", dtype=ids_datatypes, usecols=used_cols)
df7 = pd.read_csv("../Dataset/CSE-CIC-IDS2018/02-23-2018.csv", dtype=ids_datatypes, usecols=used_cols)
df10 = pd.read_csv("../Dataset/CSE-CIC-IDS2018/03-02-2018.csv", dtype=ids_datatypes, usecols=used_cols)

merge = [df1, df2, df5, df6, df7, df10]
df_ids2018 = pd.concat(merge)
del merge

# === Limpeza de eventuais valores infinitos e nulos ===
df_ids2018 = df_ids2018.replace([np.inf, -np.inf], np.nan)
df_ids2018 = df_ids2018.dropna()

# === Generalização dos nomes dos ataques ===
df_ids2018['Label'] = df_ids2018['Label'].replace({
    'FTP-BruteForce': 'BruteForce',
    'SSH-Bruteforce': 'BruteForce',
    'Brute Force -Web': 'BruteForce',
    'Brute Force -XSS': 'BruteForce',
    'DDOS attack-HOIC': 'DDOS',
    'DDOS attack-LOIC-UDP': 'DDOS',
    'DoS attacks-GoldenEye': 'DoS',
    'DoS attacks-Slowloris': 'DoS',
    'SQL Injection': 'SQLInjection'
})

# === Codificação das labels generalizadas ===
label_dict = {
    'Benign': 1,
    'BruteForce': 2,
    'DDOS': 3,
    'DoS': 4,
    'SQLInjection': 5,
    'Bot': 6
}
df_ids2018['Label'] = df_ids2018['Label'].map(label_dict)

# === Amostragem balanceada ===
benign = df_ids2018[df_ids2018['Label'] == 1][:100000]
attack_samples = []
for i in range(2, 7):
    attack_samples.append(df_ids2018[df_ids2018['Label'] == i][:10000])
df_sampled = pd.concat([benign] + attack_samples)

# === Exportação ===
df_sampled.to_csv("cleaned_ids2018.csv", index=False)
print("Ficheiro 'cleaned_ids2018.csv' criado com sucesso.")
