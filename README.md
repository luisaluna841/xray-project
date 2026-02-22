# Classificação de Pneumonia em Raio-X Torácico

> Desafio Individual — Liga Acadêmica de Inteligência Artificial (Ligia)  
> Universidade Federal de Pernambuco — Processo Seletivo 2026  
> Trilha: Visão Computacional  
> Autor: [luisaluna841](https://github.com/luisaluna841)

---

## 📄 Relatório Técnico

O relatório completo está disponível em [`relatorio.pdf`](./relatorio.pdf), incluindo análise exploratória, metodologia, resultados, interpretabilidade Grad-CAM e conclusões.

---

## Visão Geral

Este projeto desenvolve um classificador binário de imagens de raio-X torácico para detecção de pneumonia, utilizando Transfer Learning com arquiteturas pré-treinadas no ImageNet. Quatro experimentos foram conduzidos de forma sistemática e controlada, com análise crítica baseada em métricas clínicas.

**Modelo com melhor desempenho no Kaggle:** ResNet18 Baseline (`submission_resnet18.csv`) — ROC-AUC: 0.99543  
**Modelo recomendado clinicamente:** ResNet18 com Class Weighting (H3) — Recall: 0.9943 | FNs: 5  
**Métrica principal:** ROC-AUC  
**Dataset:** [Kaggle – Lígia - CV](https://www.kaggle.com/competitions/ligia-compviz/overview)

---

## Estrutura do Repositório

```
xray-project/
│
├── data/
│   ├── metadata/
│   │   ├── train_metadata.csv      # Metadados completos do treino
│   │   ├── train_split.csv         # Split de treino (congelado, seed=42)
│   │   └── val_split.csv           # Split de validação (congelado, seed=42)
│   ├── train/                      # ← você preenche (não sobe no git)
│   │   ├── NORMAL/
│   │   └── PNEUMONIA/
│   ├── test_images/                # ← você preenche (não sobe no git)
│   └── test.csv
│
├── models/                         # Pesos treinados .pt — não sobem no git
│   ├── resnet18_light_noCW.pt      # Baseline
│   ├── densenet121_light_noCW.pt   # H1
│   ├── resnet18_strong_noCW.pt     # H2
│   └── resnet18_light_CW.pt        # H3 ★ recomendado clinicamente
│
├── notebooks/
│   ├── 01_build_metadata_and_split.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_preprocessing_analysis.ipynb
│   ├── 04_baseline_resnet18.ipynb
│   ├── 05_h1_densenet121.ipynb
│   ├── 06_h2_strong_augmentation.ipynb
│   ├── 07_h3_classweight.ipynb
│   ├── 08_model_comparison.ipynb
│   ├── 09_gradcam.ipynb
│   └── 10_generate_submission.ipynb
│
├── outputs/
│   ├── figures/                    # Gráficos e visualizações (sobem no git)
│   │   └── gradcam/                # Mapas de calor Grad-CAM
│   ├── metrics/                    # Histórico de treinamento .pkl (sobem no git)
│   └── submissions/
│       ├── submission_resnet18.csv                  # ← melhor no Kaggle
│       ├── submission_densenet121_light_noCW.csv
│       ├── submission_resnet18_light_CW.csv
│       └── submission_resnet18_strong_noCW.csv
│
├── src/
│   ├── dataset.py                  # Dataset customizado (XRayDataset)
│   ├── model.py                    # Definição dos modelos
│   ├── train_utils.py              # Loop de treinamento e métricas
│   ├── transforms.py               # Transformações de imagem
│   └── utils.py                    # Seed global e utilitários
│
├── .gitignore
├── relatorio.pdf
├── CONCLUSAO.md
├── README.md
└── requirements.txt
```

---

## Instalação

### 1. Clonar o repositório

```bash
git clone https://github.com/luisaluna841/xray-project.git
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

## Download dos Dados

As imagens **não estão incluídas no repositório**. Para obtê-las:

**1.** Acesse a competição: [Kaggle – Lígia - CV](https://www.kaggle.com/competitions/ligia-compviz/overview)  
**2.** Faça login, aceite os termos (botão **Join Competition**) e vá até a aba **Data**  
**3.** Clique em **Download All** e salve o `.zip` no seu computador  
**4.** Extraia o arquivo e mova as pastas manualmente para dentro de `data/`, respeitando **exatamente** esta estrutura:

```
data/
├── train/
│   ├── NORMAL/        ← cole aqui as imagens da pasta NORMAL
│   └── PNEUMONIA/     ← cole aqui as imagens da pasta PNEUMONIA
├── test_images/       ← cole aqui as imagens de teste
└── test.csv           ← cole aqui o arquivo de metadados
```

> ⚠️ **Não mexa em `data/metadata/`** — essa pasta já está no repositório com os splits congelados que garantem a reprodutibilidade de todos os experimentos.

---

## Reprodução dos Experimentos

Execute os notebooks **na ordem numérica**:

| # | Notebook | Descrição |
|---|---|---|
| 01 | `01_build_metadata_and_split.ipynb` | Constrói metadados e congela os splits |
| 02 | `02_eda.ipynb` | Análise exploratória e justificativas metodológicas |
| 03 | `03_preprocessing_analysis.ipynb` | Análise de pré-processamento e augmentation |
| 04 | `04_baseline_resnet18.ipynb` | Treinamento do Baseline |
| 05 | `05_h1_densenet121.ipynb` | Hipótese 1: DenseNet121 |
| 06 | `06_h2_strong_augmentation.ipynb` | Hipótese 2: Augmentation Forte |
| 07 | `07_h3_classweight.ipynb` | Hipótese 3: Class Weighting |
| 08 | `08_model_comparison.ipynb` | Comparação final entre experimentos |
| 09 | `09_gradcam.ipynb` | Interpretabilidade Grad-CAM |
| 10 | `10_generate_submission.ipynb` | Geração do arquivo de submissão |

> ⚠️ **Não re-execute o notebook 01.** Os splits estão congelados em `data/metadata/` e são compartilhados por todos os experimentos — re-executar alteraria a divisão e tornaria as comparações inválidas.

---

## Reprodutibilidade

Este projeto foi projetado para rodar do mesmo jeito em qualquer máquina:

- **Seed 42** fixada globalmente em todos os experimentos via `src/utils.py`
- **Splits congelados** em `data/metadata/` — mesma divisão treino/validação para todos os modelos
- **Caminhos relativos** em todos os notebooks — nenhum caminho absoluto
- **Versões fixas** de dependências em `requirements.txt`
- **Critério de salvamento determinístico** — melhor época por ROC-AUC de validação
- **Histórico completo** de métricas salvo em `outputs/metrics/*.pkl`

---

## Hipóteses Experimentais

Todos os experimentos compartilham os mesmos splits, hiperparâmetros base (lr=1e-4, batch=32, epochs=10, Adam + ReduceLROnPlateau) e pesos pré-treinados no ImageNet. Cada hipótese altera **uma única variável** em relação ao Baseline.

### Baseline — ResNet18 + Augmentation Leve + Sem Class Weight

Modelo de referência com ResNet18, *flip* horizontal e rotação ±5°. **Obteve o melhor ROC-AUC na competição: 0.99543**.

### Hipótese 1 (H1) — Arquitetura: DenseNet121

**Pergunta:** Conexões densas entre camadas capturam padrões mais sutis de pneumonia do que a adição residual da ResNet18?

**Resultado: não confirmada.** 19 FNs vs 16 do Baseline. A maior complexidade foi desfavorável para o tamanho moderado do dataset — a ResNet18 converge com maior estabilidade.

> A DenseNet121 não é apenas "mais profunda" — representa um paradigma distinto: reutilização de *features* por concatenação vs adição residual. A comparação é entre mecanismos de propagação de informação, não apenas profundidade.

### Hipótese 2 (H2) — Data Augmentation Intenso

**Pergunta:** Augmentation mais agressivo (rotação ±15°, affine, jitter de brilho/contraste) atua como regularização eficaz?

**Resultado: não confirmada.** Pior resultado clínico: 24 FNs e Recall de 0.9725. Transformações geométricas agressivas distorcem consolidações e infiltrados — padrões patológicos sensíveis a deformações.

### Hipótese 3 (H3) — Ponderação de Classes ✓

**Pergunta:** Class weighting na função de perda melhora a Sensitivity para Pneumonia, reduzindo Falsos Negativos?

**Resultado: confirmada.** É o modelo recomendado clinicamente — ver seção de Resultados.

---

## Resultados

### Desempenho no Kaggle

| Submissão | ROC-AUC (público) |
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
| **H3 — Class Weight ★** | **0.9992** | **0.9931** | **0.9943** | 0.9920 | **5** |

### Por que o H3 é o modelo recomendado clinicamente?

Em termos de ROC-AUC — tanto no Kaggle quanto na validação interna — todos os modelos apresentam desempenho equivalente e alto. Isso indica que qualquer experimento seria uma solução tecnicamente válida do ponto de vista de capacidade discriminativa global.

No entanto, **ROC-AUC agrega o desempenho ao longo de todos os limiares de decisão possíveis**, não refletindo o comportamento clínico em um limiar fixo de operação. A análise em threshold 0.5 — mais representativa do uso clínico real — revela diferenças substanciais:

O H3 reduz os **Falsos Negativos de 16 para 5 — redução de 69%**. Em contexto diagnóstico, um Falso Negativo representa um paciente com pneumonia dispensado sem tratamento. O custo assimétrico entre FN e FP justifica a escolha de um modelo com maior Sensitivity (0.9943 vs 0.9817), mesmo que isso implique leve redução de Specificity (0.9738 vs 0.9850) e AUC inferior no leaderboard público.

O H3 é o único experimento **explicitamente desenhado para esse tradeoff clínico**, e a análise Grad-CAM confirma ativações concentradas em regiões anatomicamente plausíveis.

---

## Interpretabilidade (Grad-CAM)

A análise Grad-CAM foi aplicada nos modelos Baseline e H3 para verificar se as decisões se baseiam em regiões anatomicamente plausíveis. Os mapas de calor estão em `outputs/figures/gradcam/`.

O **"p"** nas figuras é a probabilidade de saída do softmax para a classe Pneumonia. Nos FNs do Baseline (p ≈ 0.44), o modelo erra próximo ao limiar sem sinalizar dúvida. Nos FNs do H3 (p < 0.20), o modelo erra com incerteza explícita — em um sistema de triagem real, esses casos podem ser encaminhados para revisão humana, transformando o erro em alerta em vez de dispensa silenciosa.

---

## Dependências

```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.23.0
pandas>=1.5.0
matplotlib>=3.6.0
seaborn>=0.12.0
scikit-learn>=1.2.0
Pillow>=9.3.0
tqdm>=4.64.0
opencv-python>=4.7.0
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