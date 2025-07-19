# Sistema de Deteção de Intrusões com Machine Learning

Este repositório contém o código-fonte do projeto final da Licenciatura em Engenharia Informática da Universidade Autónoma de Lisboa, desenvolvido no âmbito da unidade curricular de Laboratório de Projeto. O projeto consistiu na construção de um sistema de deteção de intrusões (IDS) baseado em modelos de Machine Learning, capazes de identificar tráfego normal e tráfego malicioso numa rede virtual simulada.

## 📌 Objetivo

Desenvolver e testar um IDS com recurso a diferentes abordagens de Machine Learning, supervisionadas e não supervisionadas, para deteção de anomalias e ataques em tráfego de rede. O sistema foi validado em ambiente de testes com máquinas virtuais.

## 🧠 Modelos Desenvolvidos

Foram implementados os seguintes modelos:

### Supervisionados:
- Multi-Layer Perceptron (MLP)
- Random Forest (RF)
- Support Vector Machine (SVM)

### Não Supervisionados:
- K-Means
- Isolation Forest
- Autoencoder

## 📂 Estrutura do Projeto

O repositório inclui apenas o **código Python** utilizado para o desenvolvimento, treino e avaliação dos modelos. **As máquinas virtuais utilizadas no ambiente de testes não estão disponíveis** neste repositório.

Caso pretenda obter acesso às máquinas virtuais ou à configuração do ambiente completo, por favor entre em contacto através do email: `viana.tomasmiguel@gmail.com`.

## 📊 Dataset

O dataset utilizado neste projeto foi o **CSE-CIC-IDS2018**, disponível publicamente em:

🔗 [https://www.kaggle.com/datasets/solarmainframe/ids-intrusion-csv](https://www.kaggle.com/datasets/solarmainframe/ids-intrusion-csv)

### Ficheiros utilizados:
Apenas foram usados os seguintes ficheiros do dataset original:

- `02-14-2018.csv`
- `02-15-2018.csv`
- `02-21-2018.csv`
- `02-22-2018.csv`
- `02-23-2018.csv`
- `03-02-2018.csv`

Estes ficheiros devem ser colocados no seguinte caminho:

```
📁 Dataset/
└── 📁 CSE-CIC-IDS2018/
    ├── 📄 02-14-2018.csv
    ├── 📄 02-15-2018.csv
    ├── 📄 02-21-2018.csv
    ├── 📄 02-22-2018.csv
    ├── 📄 02-23-2018.csv
    └── 📄 03-02-2018.csv
```

Para executar o projeto e colocá-lo a funcionar, encontrará tudo o que precisa no **`Manual de Instruções`** presente na secção de **`Anexos`** do Relatório do Projeto, que se encontra neste repositório.

## 👥 Autores

- Afonso Figueiredo Frasquilho (30010929)  
- Guilherme Lopes Fernandes (30010398)  
- Tomás Miguel Rodrigues Viana (30010623)

**Orientador:** Prof. Doutor Mário Marques da Silva

## 📄 Licença

Este projeto é apenas para fins académicos. Qualquer reutilização do código deve referenciar os autores e o projeto original.
