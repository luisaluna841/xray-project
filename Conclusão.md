# Conclusão do Projeto

## Síntese

Este projeto investigou a classificação automática de pneumonia em radiografias torácicas por meio de quatro experimentos controlados com Transfer Learning. O principal achado é que **ROC-AUC não discrimina os modelos** — todos convergiram para valores equivalentes (0.9989–0.9992 na validação; 0.9907–0.9954 no Kaggle). A diferença clinicamente relevante aparece apenas na análise em threshold fixo (0.5):

| Modelo | AUC | Recall | F1 | FNs |
|---|---|---|---|---|
| Baseline (ResNet18) | 0.9990 | 0.9817 | 0.9885 | 16 |
| H1 — DenseNet121 | 0.9989 | 0.9782 | 0.9873 | 19 |
| H2 — Strong Aug | 0.9990 | 0.9725 | 0.9849 | 24 |
| **H3 — Class Weight ★** | **0.9992** | **0.9943** | **0.9931** | **5** |

---

## Resultado por Hipótese

**H1 — Arquitetura (DenseNet121)** ❌ Não confirmada  
A maior complexidade da DenseNet121 foi desfavorável para o tamanho do dataset. FNs aumentaram de 16 para 19 e a convergência foi mais instável que a ResNet18.

**H2 — Data Augmentation Intenso** ❌ Não confirmada  
O augmentation agressivo distorceu padrões patológicos como consolidações e infiltrados — estruturas sensíveis a deformações geométricas. Pior resultado clínico: 24 FNs e Recall de 0.9725.

**H3 — Class Weighting** ✅ Confirmada  
A ponderação de classes reduziu os Falsos Negativos de 16 para 5 — **redução de 69%** — com Recall de 0.9943 e F1 de 0.9931. É o único experimento projetado para o custo assimétrico inerente ao diagnóstico de pneumonia.

---

## Modelo Recomendado

**ResNet18 com Class Weighting** (`resnet18_light_CW.pt`) é o modelo recomendado para aplicações de suporte ao diagnóstico por três razões:

**1. Maior Recall (0.9943):** detecta mais casos de pneumonia, reduzindo o risco de dispensar pacientes doentes sem tratamento.

**2. Interpretabilidade coerente:** a análise Grad-CAM confirma ativações concentradas em regiões pulmonares anatomicamente plausíveis — consolidações e infiltrados — e não em artefatos periféricos.

**3. Incerteza calibrada:** nos Falsos Negativos restantes, o modelo erra com baixa confiança (p < 0.20), sinalizando dúvida em vez de falsa certeza. O Baseline falha com p ≈ 0.44 — próximo ao limiar, sem nenhum sinal de alerta.

> **Nota sobre o Kaggle:** o arquivo submetido na competição foi **`submission_resnet18.csv`** — gerado pelo modelo Baseline (ResNet18 sem class weighting), que obteve o melhor ROC-AUC público: **0.99543**. A recomendação do H3 é baseada exclusivamente na análise clínica em threshold fixo — critério mais representativo do uso diagnóstico real, onde o custo de um Falso Negativo é incomparavelmente maior que o de um alarme falso.

---

## Limitações

- Treinamento limitado a 10 épocas sem busca sistemática de hiperparâmetros
- Threshold fixo em 0.5 — otimização por curva ROC poderia ampliar os ganhos de Sensitivity do H3
- Peso das classes definido proporcionalmente ao desbalanceamento, sem ajuste fino
- Interpretabilidade Grad-CAM qualitativa — uma análise quantitativa de localização (ex.: IoU com máscaras de segmentação pulmonar) fortaleceria as conclusões sobre plausibilidade anatômica

---

## Trabalhos Futuros

- Otimização do threshold de decisão por análise da curva ROC (índice de Youden)
- Ajuste fino do *class weighting* via validação cruzada estratificada
- Avaliação de *focal loss* como alternativa ao class weighting padrão
- Extensão para classificação multiclasse: pneumonia bacteriana vs viral
- Validação em datasets externos para avaliação de generalização entre domínios (*domain shift*) — etapa indispensável para qualquer sistema com pretensão de uso clínico real

---

*Liga Acadêmica de Inteligência Artificial — UFPE, 2026*  
*[luisaluna841](https://github.com/luisaluna841)*