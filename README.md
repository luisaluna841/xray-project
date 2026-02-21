# Classificação de Pneumonia em Raio-X Torácico

> Desafio Individual — Liga Acadêmica de Inteligência Artificial (Ligia)  
> Universidade Federal de Pernambuco — Processo Seletivo 2026  
> Trilha: Visão Computacional

---

## Visão Geral

Este projeto desenvolve um classificador binário de imagens de raio-X torácico para detecção de pneumonia, utilizando Transfer Learning com arquiteturas pré-treinadas no ImageNet. Quatro experimentos foram conduzidos de forma sistemática e controlada, com análise crítica baseada em métricas clínicas.

**Modelo com melhor desempenho no Kaggle:** ResNet18 Baseline (`submission_resnet18.csv`) — ROC-AUC: 0.99543  
**Modelo recomendado clinicamente:** ResNet18 com Class Weighting (H3)  
**Métrica principal:** ROC-AUC  
**Dataset:** [Kaggle – Lígia - CV]([https://www.kaggle.com/competitions/chest-xray-pneumonia-ligia](https://www.kaggle.com/competitions/chest-xray-pneumonia-ligia))

---

## Estrutura do Repositório

```
xray-project/
│
├── data/
│   ├── metadata/
│   │   ├── train_metadata.csv      # Metadados completos do treino
│   │   ├── train_split.csv         # Split de treino (congelado)
│   │   └── val_split.csv           # Split de validação (congelado)
│   ├── train/
│   │   ├── NORMAL/                 # Imagens normais de treino
│   │   └── PNEUMONIA/              # Imagens com pneumonia de treino
│   ├── test_images/                # Imagens de teste (Kaggle)
│   └── test.csv                    # Metadados do conjunto de teste
│
├── models/                         # Pesos dos modelos treinados (.pt)
│   ├── resnet18_light_noCW.pt      # Baseline
│   ├── densenet121_light_noCW.pt   # H1
│   ├── resnet18_strong_noCW.pt     # H2
│   └── resnet18_light_CW.pt        # H3 — modelo recomendado clinicamente
│
├── notebooks/
│   ├── 01_build_metadata_and_split.ipynb   # Construção dos splits
│   ├── 02_eda.ipynb                         # Análise exploratória
│   ├── 02_preprocessing_analysis.ipynb     # Análise de pré-processamento
│   ├── 03_baseline_resnet18.ipynb          # Experimento Baseline
│   ├── 04_h1_densenet121.ipynb             # Hipótese 1
│   ├── 05_h2_strong_augmentation.ipynb     # Hipótese 2
│   ├── 06_h3_classweight.ipynb             # Hipótese 3
│   ├── 07_gradcam.ipynb                    # Interpretabilidade Grad-CAM
│   ├── 08_model_comparison.ipynb           # Comparação final
│   └── 09_generate_submission.ipynb        # Geração do submission.csv
│
├── outputs/
│   ├── figures/                    # Gráficos e visualizações
│   │   └── gradcam/                # Mapas de calor Grad-CAM
│   ├── metrics/                    # Histórico de treinamento (.pkl)
│   └── submissions/                # Arquivos CSV de submissão
│       ├── submission_resnet18.csv                  # ← melhor no Kaggle
│       ├── submission_densenet121_light_noCW.csv
│       ├── submission_resnet18_light_CW.csv
│       └── submission_resnet18_strong_noCW.csv
│
├── src/
│   ├── dataset.py                  # Dataset customizado (XRayDataset)
│   ├── model.py                    # Definição dos modelos
│   ├── train_utils.py              # Loop de treinamento
│   ├── transforms.py               # Transformações de imagem
│   └── utils.py                    # Utilitários gerais
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Instalação e Configuração

### 1. Clonar o repositório

```bash
git clone https://github.com/<seu-usuario>/xray-project.git
cd xray-project
```

### 2. Criar e ativar ambiente virtual

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python -m venv venv
source venv/bin/activate
```

### 3. Instalar dependências

```bash
pip install -r requirements.txt
```

---

## Download dos Dados (Kaggle)

As imagens são fornecidas pela competição no [Kaggle – Lígia - CV]([https://www.kaggle.com/competitions/chest-xray-pneumonia-ligia](https://www.kaggle.com/competitions/chest-xray-pneumonia-ligia)) e **não estão incluídas no repositório** por limitações de tamanho. Siga os passos abaixo para baixá-las automaticamente.

### Pré-requisito: configurar credenciais do Kaggle

1. Acesse [kaggle.com]([https://www.kaggle.com](https://www.kaggle.com/competitions/chest-xray-pneumonia-ligia)) → Account → API → **Create New Token**
2. Salve o arquivo `kaggle.json` baixado em:
   - **Windows:** `C:\Users\<usuario>\.kaggle\kaggle.json`
   - **Linux/macOS:** `~/.kaggle/kaggle.json`

```bash
# Linux/macOS: ajustar permissões
chmod 600 ~/.kaggle/kaggle.json
```

### Executar o script de download

```bash
python download_data.py
```

O script irá:
- Baixar as imagens da competição via API do Kaggle
- Organizar automaticamente nas pastas `data/train/` e `data/test_images/`
- Verificar a integridade dos arquivos após o download

> **Nota:** É necessário ter aceitado os termos da competição no [Kaggle – Lígia - CV](https://www.kaggle.com/competitions/chest-xray-pneumonia-ligia) antes de executar o script.

---

## Script de Download

Salve o arquivo abaixo como `download_data.py` na raiz do projeto:

```python
"""
download_data.py
Script para download e organização automática das imagens da competição.

Uso:
    python download_data.py

Pré-requisito:
    - kaggle.json configurado em ~/.kaggle/kaggle.json
    - Ter aceitado os termos da competição no Kaggle
"""

import os
import zipfile

COMPETITION = "chest-xray-pneumonia-ligia"
DATA_DIR    = os.path.join(os.path.dirname(__file__), "data")

def download_and_extract():
    print("=" * 55)
    print("Download dos dados — Ligia Xray Competition")
    print("=" * 55)

    os.makedirs(DATA_DIR, exist_ok=True)

    print(f"\n[1/3] Baixando dados de: {COMPETITION}")
    os.system(
        f"kaggle competitions download -c {COMPETITION} -p {DATA_DIR}"
    )

    zip_path = os.path.join(DATA_DIR, f"{COMPETITION}.zip")

    if not os.path.exists(zip_path):
        print("\n[ERRO] Download falhou.")
        print("Verifique:")
        print("  1. Credenciais em ~/.kaggle/kaggle.json")
        print("  2. Se aceitou os termos da competição no Kaggle")
        return

    print(f"\n[2/3] Extraindo arquivos em: {DATA_DIR}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(DATA_DIR)
    os.remove(zip_path)
    print("  Arquivo zip removido após extração.")

    print("\n[3/3] Verificando estrutura de pastas...")
    expected = [
        os.path.join(DATA_DIR, "train", "NORMAL"),
        os.path.join(DATA_DIR, "train", "PNEUMONIA"),
        os.path.join(DATA_DIR, "test_images"),
    ]

    all_ok = True
    for folder in expected:
        exists = os.path.isdir(folder)
        status = "OK      " if exists else "FALTANDO"
        print(f"  [{status}] {folder}")
        if not exists:
            all_ok = False

    if all_ok:
        n_normal    = len(os.listdir(os.path.join(DATA_DIR, "train", "NORMAL")))
        n_pneumonia = len(os.listdir(os.path.join(DATA_DIR, "train", "PNEUMONIA")))
        n_test      = len(os.listdir(os.path.join(DATA_DIR, "test_images")))
        print("\nDownload concluído com sucesso!")
        print(f"  Treino — Normal:    {n_normal} imagens")
        print(f"  Treino — Pneumonia: {n_pneumonia} imagens")
        print(f"  Teste:              {n_test} imagens")
    else:
        print("\n[AVISO] Algumas pastas esperadas não foram encontradas.")

    print("=" * 55)

if __name__ == "__main__":
    download_and_extract()
```

---

## Reprodução dos Experimentos

Execute os notebooks **na ordem numérica** a partir da pasta `notebooks/`:

| Ordem | Notebook | Descrição |
|---|---|---|
| 01 | `01_build_metadata_and_split.ipynb` | Constrói metadados e congela os splits |
| 02 | `02_eda.ipynb` | Análise exploratória e justificativas metodológicas |
| 03 | `03_baseline_resnet18.ipynb` | Treinamento do Baseline |
| 04 | `04_h1_densenet121.ipynb` | Hipótese 1: DenseNet121 |
| 05 | `05_h2_strong_augmentation.ipynb` | Hipótese 2: Augmentation Forte |
| 06 | `06_h3_classweight.ipynb` | Hipótese 3: Class Weighting |
| 07 | `07_gradcam.ipynb` | Interpretabilidade Grad-CAM |
| 08 | `08_model_comparison.ipynb` | Comparação final entre experimentos |
| 09 | `09_generate_submission.ipynb` | Geração do arquivo de submissão |

> **Importante:** Os splits estão congelados em `data/metadata/`. Não execute o notebook 01 novamente para garantir reprodutibilidade.

---

## Hipóteses Experimentais

Todos os experimentos compartilham os mesmos splits, hiperparâmetros base (lr=1e-4, batch=32, epochs=10) e pesos pré-treinados no ImageNet. Cada hipótese altera **uma única variável** em relação ao Baseline.

### Baseline — ResNet18 + Augmentation Leve + Sem Class Weight

Modelo de referência. Estabelece o desempenho base do projeto com a arquitetura ResNet18 e configurações conservadoras. **Obteve o melhor ROC-AUC na competição** ([Kaggle – Lígia - CV]([https://www.kaggle.com/competitions/chest-xray-pneumonia-ligia](https://www.kaggle.com/competitions/chest-xray-pneumonia-ligia))): **0.99543**.

### Hipótese 1 (H1) — Arquitetura: DenseNet121

**Pergunta:** Uma arquitetura com conexões densas entre camadas consegue capturar padrões mais sutis de pneumonia em relação à ResNet18?

**Resultado: não confirmada.** AUC de 0.99359 no Kaggle e maior número de Falsos Negativos na validação interna (19 vs 16). Para este dataset de tamanho moderado, a capacidade representacional adicional não trouxe ganhos — e o custo computacional foi significativamente maior.

> Nota metodológica: a DenseNet121 não é apenas "mais profunda" que a ResNet18 — representa um paradigma arquitetural distinto, com reutilização de features por concatenação (dense connections) em vez de adição residual. A comparação é entre paradigmas, não apenas profundidade.

### Hipótese 2 (H2) — Data Augmentation Intenso

**Pergunta:** Estratégias de augmentation mais agressivas (rotação ±15°, affine, jitter de brilho/contraste) atuam como regularização eficaz, reduzindo sobreajuste?

**Resultado: não confirmada.** AUC de 0.99152 no Kaggle. O Recall piorou (0.9725) e os FNs aumentaram para 24 na validação interna. Para imagens médicas, transformações geométricas agressivas podem distorcer padrões patológicos relevantes como consolidações e infiltrados.

### Hipótese 3 (H3) — Ponderação de Classes ✓

**Pergunta:** A aplicação de class weighting na função de perda melhora a sensibilidade para a classe Pneumonia, reduzindo Falsos Negativos?

**Resultado: confirmada.** É o modelo recomendado clinicamente. AUC de 0.99074 no Kaggle — inferior ao Baseline em termos competitivos, mas superior em todas as métricas clínicas relevantes. Ver seção de Resultados.

---

## Resultados

### Desempenho no Kaggle ([Kaggle – Lígia - CV](https://www.kaggle.com/competitions/chest-xray-pneumonia-ligia))

| Submissão | ROC-AUC (Kaggle público) |
|---|---|
| **submission_resnet18.csv** (Baseline) ✓ selecionada | **0.99543** ← melhor resultado |
| submission_densenet121_light_noCW.csv (H1) | 0.99359 |
| submission_resnet18_strong_noCW.csv (H2) | 0.99152 |
| submission_resnet18_light_CW.csv (H3) | 0.99074 |

### Desempenho na Validação Interna (threshold = 0.5)

| Modelo | AUC | F1 | Recall | Precision | FNs |
|---|---|---|---|---|---|
| Baseline (ResNet18) | 0.9990 | 0.9885 | 0.9817 | 0.9953 | 16 |
| H1 — DenseNet121 | 0.9989 | 0.9873 | 0.9782 | 0.9965 | 19 |
| H2 — Strong Aug | 0.9990 | 0.9849 | 0.9725 | 0.9976 | 24 |
| **H3 — Class Weight** | **0.9992** | **0.9931** | **0.9943** | 0.9920 | **5** |

### Por que o H3 é o modelo recomendado clinicamente?

Em termos de ROC-AUC — tanto no Kaggle quanto na validação interna — todos os modelos apresentam desempenho equivalente e alto. Isso indica que qualquer um dos experimentos seria uma solução tecnicamente válida do ponto de vista de capacidade discriminativa global.

No entanto, **ROC-AUC agrega o desempenho ao longo de todos os limiares de decisão possíveis**, não refletindo o comportamento clínico em um limiar fixo de operação. A análise em threshold 0.5 — mais representativa do uso clínico real — revela diferenças substanciais:

O modelo H3 reduz os **Falsos Negativos de 16 para 5 — uma redução de 69%** em relação ao Baseline. Em contexto diagnóstico, um Falso Negativo representa um paciente com pneumonia classificado como Normal e dispensado sem tratamento. O custo assimétrico entre FN e FP justifica a escolha de um modelo com maior Sensitivity (0.9943 vs 0.9817), mesmo que isso implique leve redução de Specificity (0.9738 vs 0.9850) e AUC inferior no leaderboard público.

O H3 é o único experimento **explicitamente desenhado para esse tradeoff clínico**, e a análise Grad-CAM confirma ativações concentradas em regiões anatomicamente plausíveis.

---

## Interpretabilidade (Grad-CAM)

A análise Grad-CAM foi aplicada nos modelos Baseline e H3 para verificar se as decisões são baseadas em regiões anatomicamente plausíveis. Os mapas de calor estão em `outputs/figures/gradcam/`.

Nos casos em que o Baseline erra (FNs), o H3 ativa regiões pulmonares mais difusas e bilaterais — padrão consistente com consolidações por pneumonia. Nos FNs restantes do H3, o modelo demonstra baixa probabilidade de confiança (p < 0.20), indicando incerteza calibrada em vez de erros com alta confiança.

---

## Dependências

```
torch>=1.13.0
torchvision>=0.14.0
numpy>=1.23.0
pandas>=1.5.0
matplotlib>=3.6.0
seaborn>=0.12.0
scikit-learn>=1.1.0
Pillow>=9.3.0
tqdm>=4.64.0
opencv-python>=4.6.0
kaggle>=1.5.12
```

```bash
pip install -r requirements.txt
```

---

## Reprodutibilidade

Seed global fixada em **42** em todos os experimentos via `src/utils.py`:

```python
set_seed(42)
```

Os splits de treino e validação estão congelados nos arquivos CSV e são compartilhados entre todos os experimentos, garantindo comparação justa e controlada.

---

*Liga Acadêmica de Inteligência Artificial — UFPE, 2026*
