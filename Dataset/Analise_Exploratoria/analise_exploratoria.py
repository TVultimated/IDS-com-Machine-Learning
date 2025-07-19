import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# === 1. Carregamento do Dataset Final ===
df = pd.read_csv("../cleaned_ids2018.csv")

# === 2. Dimensão e Estrutura ===
print("Número de registos e atributos:", df.shape)
print("\nTipos de dados por coluna:")
pd.set_option("display.max_rows", None)
print(df.dtypes)

print("\nColunas com valores nulos:")
print(df.isnull().sum()[df.isnull().sum() > 0])

# === 3. Estatísticas Descritivas ===
desc = df.describe().T
desc['median'] = df.median(numeric_only=True)
desc['skewness'] = df.skew(numeric_only=True)
desc['kurtosis'] = df.kurtosis(numeric_only=True)

# Configurações para ver toda a tabela no terminal
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", None)

# Selecionar e imprimir resumo relevante
tabela_resumo = desc[['mean', 'std', 'min', '25%', '50%', '75%', 'max', 'median', 'skewness', 'kurtosis']].round(2)
print("\nEstatísticas descritivas:")
print(tabela_resumo)
tabela_resumo.to_csv("estatisticas_descritivas.csv")

# === 4. Distribuição das Classes ===
print("\nClass Distribution:")
print(df['Label'].value_counts().sort_index())

# Gráfico de barras das classes com percentagens e escala ajustada
class_counts = df['Label'].value_counts().sort_index()
class_percent = (class_counts / class_counts.sum() * 100).round(2)

plt.figure(figsize=(10, 6))
bars = plt.bar(class_counts.index.astype(str), class_counts.values)

# Adicionar percentagens por cima das barras
for i, bar in enumerate(bars):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 1000, f'{class_percent.iloc[i]}%', ha='center', fontsize=9)

plt.xlabel("Class")
plt.ylabel("Number of Records")
plt.title("Class Distribution")
plt.tight_layout()
plt.savefig("distribuicao_classes.png")
plt.show()

# === 5. Correlação entre atributos numéricos ===
variancias = df.var(numeric_only=True).sort_values(ascending=False)
top_atributos = variancias.head(15).index
df_top = df[top_atributos]

corr_top = df_top.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr_top, annot=True, fmt=".2f", cmap="viridis", center=0, square=True,
            cbar_kws={'label': 'Correlation'})
plt.title("Correlation Matrix", fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("matriz_correlacao.png")
plt.show()