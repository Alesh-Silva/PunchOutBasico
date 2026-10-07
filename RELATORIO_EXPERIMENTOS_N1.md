# RELATÓRIO TÉCNICO DE EXPERIMENTOS DE MACHINE LEARNING (ETAPA N1)

**Projeto:** Sistema de IA Adaptativa com Modelos de Machine Learning em Jogo 2D (Punch-Out)  
**Referência Teórica:** Livro *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* (Aurélien Géron) — Capítulos 1 e 2  
**Disciplina:** Tópicos Especiais / Inteligência Artificial  

---

## 1. INTRODUÇÃO E FORMULAÇÃO DO PROBLEMA

O objetivo deste projeto é desenvolver, treinar e avaliar modelos de **Aprendizado Supervisionado** para prever as ações de combate de um jogador humano em tempo real dentro de um jogo 2D desenvolvido em Pygame, inspirado na jogabilidade clássica de *Punch-Out!!*.

Alinhado ao fluxo de trabalho apresentado nos **Capítulos 1 e 2 de Géron**:
1. **Natureza da Tarefa:** Trata-se de um problema de **Classificação Supervisionada Multiclasse**. O modelo recebe o histórico recente de ações do jogador e prevê a sua próxima ação $y \in \{0, 1, 2, 3, 4, 5\}$.
2. **Classes de Ações Mapeadas:**
   - `0`: Soco Esquerdo (Jab)
   - `1`: Soco Direito (Direto)
   - `2`: Soco Especial (Uppercut)
   - `3`: Esquiva para Esquerda
   - `4`: Esquiva para Direita
   - `5`: Guarda / Bloqueio
3. **Métricas de Desempenho:** Acurácia Global, Precisão, Revocação (Recall), F1-Score Ponderado e Validação Cruzada Estratificada (*Stratified 5-Fold Cross-Validation*).
4. **Linha de Base (Baseline Aleatório):** Um modelo aleatório possui acerto esperado de apenas $\frac{1}{6} \approx 16,67\%$. Modelos eficazes devem superar consistentemente este patamar.

---

## 2. CONJUNTO DE DADOS E ENGENHARIA DE ATRIBUTOS (FEATURES)

O conjunto de dados (`dados_combate.csv`) contém 1.500 registros de combate simulando padrões comportamentais reais (combos frequentes como Soco Esquerdo $\rightarrow$ Soco Direito, repetições de golpes bem-sucedidos e recuos defensivos sob estamina reduzida).

Utilizou-se a técnica de **Janela Temporal Deslizante (*Lag Features*)**:

| Atributo (Feature) | Tipo | Descrição |
| :--- | :--- | :--- |
| `hp_jogador` | Numérico (0 a 100) | Pontos de vida atuais do jogador |
| `stamina_jogador` | Numérico (0 a 100) | Nível de estamina no instante do golpe |
| `acao_t_3` | Categórico (0 a 5) | Ação executada há 3 passos |
| `acao_t_2` | Categórico (0 a 5) | Ação executada há 2 passos |
| `acao_t_1` | Categórico (0 a 5) | Ação executada imediatamente anterior |
| **`proxima_acao`** (Alvo) | Categórico (0 a 5) | **Ação futura a ser prevista pelo modelo** |

Divisão dos dados: **80% para Treino (1.200 amostras)** e **20% para Teste (300 amostras)**, com amostragem estratificada.

---

## 3. CARACTERÍSTICAS DOS MODELOS TRABALHADOS

Foram experimentados e comparados 4 algoritmos fundamentais:

### 3.1. Regressão Logística (Classificador Linear Multiclasse)
- **Funcionamento:** Calcula somatórios ponderados lineares dos atributos de entrada e aplica a função *Softmax* para produzir probabilidades de pertinência a cada classe.
- **Características:** Rápido de ajustar e computacionalmente leve, porém limitado por traçar fronteiras de decisão lineares, tendo dificuldade em modelar ciclos condicionais de combate.

### 3.2. K-Nearest Neighbors (KNN - $k=5$)
- **Funcionamento:** Algoritmo baseado em instâncias (*Lazy Learning*). Para cada nova amostra, localiza os $k$ vizinhos mais próximos no espaço vetorial utilizando Distância Euclidiana e decide por voto majoritário.
- **Características:** Não assume distribuição prévia dos dados, mas é sensível à escala das variáveis e a ruídos em sequências categóricas.

### 3.3. Árvore de Decisão (*Decision Tree* - Profundidade 4)
- **Funcionamento:** Constrói uma árvore hierárquica de decisões binárias particionando o conjunto de dados para minimizar a impureza de Gini.
- **Características:** Modelo de caixa-branca (totalmente interpretável e visualizável). Captura com facilidade regras condicionais de sequências ("Se o último foi soco esquerdo, a chance do próximo ser direito é alta"). A profundidade máxima foi regulada em 4 (`max_depth=4`) para prevenir sobreajuste (*overfitting*).

### 3.4. Random Forest (*Ensemble* - 50 Estimadores)
- **Funcionamento:** Combina 50 árvores de decisão treinadas em subamostras aleatórias dos dados via *Bagging* (*Bootstrap Aggregating*).
- **Características:** Alta robustez e redução de variância em relação a árvores individuais.

---

## 4. RESULTADOS OBTIDOS (RESULTADOS Y E Z)

Os modelos foram submetidos ao conjunto de teste independente e à validação cruzada 5-fold:

| Modelo | Acurácia Teste (%) | F1-Score (%) | Precisão (%) | Recall (%) | Validação Cruzada 5-Fold ($\mu \pm \sigma$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Regressão Logística** | 33,67% | 32,75% | 33,14% | 33,67% | 33,58% ± 4,20% |
| **KNN ($k=5$)** | 26,67% | 26,06% | 26,20% | 26,67% | 27,25% ± 3,01% |
| **Árvore de Decisão ($prof=4$)** | **41,67%** | **40,09%** | **40,43%** | **41,67%** | **39,08% ± 1,85%** |
| **Random Forest ($n=50$)** | 39,67% | 37,81% | 38,12% | 39,67% | 38,92% ± 1,86% |

*(Acurácia teórica de escolha aleatória pura: **16,67%**)*

### Análise dos Resultados:
- **Desempenho da Árvore de Decisão:** O modelo alcançou a maior taxa de acerto no teste (**41,67%**), o que representa **2,5 vezes o desempenho do baseline aleatório**, além de apresentar a menor dispersão na validação cruzada (±1,85%), comprovando consistência e capacidade de generalização.
- **Exportação do Modelo Campeão:** A Árvore de Decisão foi serializada e exportada para o arquivo `modelo_punchout.joblib`, sendo carregada dinamicamente pelo jogo em Pygame.

---

## 5. GRÁFICOS GERADOS PARA O RELATÓRIO

Todos os gráficos foram exportados em alta resolução (300 DPI) na pasta `graficos/`:

1. **`graficos/comparativo_desempenho_modelos.png`**:
   - Gráfico de barras comparativo exibindo Acurácia e F1-Score lado a lado para cada algoritmo.
2. **`graficos/matrizes_confusao.png`**:
   - Matrizes de confusão detalhadas do KNN e da Árvore de Decisão, demonstrando o alinhamento de predições corretas na diagonal principal.
3. **`graficos/importancia_atributos.png`**:
   - Análise de relevância dos atributos na Árvore de Decisão, comprovando que o golpe imediato anterior (`acao_t_1`) responde por mais de **60% da decisão preditiva**, confirmando o peso dos combos sequenciais.
4. **`graficos/arvore_decisao_visual.png`**:
   - Diagrama completo dos nós e regras de decisão aprendidas pela Árvore.

---

## 6. O JOGO 2D EM PYGAME (MECÂNICA PUNCH-OUT E PROGRESSÃO DA IA)

O jogo desenvolvido (`punch_out_game.py`) aplica os conceitos clássicos de combate rítmico do arcade *Punch-Out!!*:

### 6.1. Guarda Neutra, Punição a Spam e Stamina
- O adversário permanece em guarda alta por padrão, bloqueando ataques desferidos no neutro.
- O jogador dispõe de **5 pontos de stamina** que decaem a cada golpe bloqueado. Se a stamina esgotar, o jogador sofre um contra-golpe punitivo automático do adversário. A stamina restaura para 5 sempre que o jogador recebe dano.

### 6.2. Janela de Contra-Ataque (Core Loop)
- O oponente sinaliza previamente seus ataques piscando em **amarelo** (*telegraph*).
- O jogador deve reagir esquivando com precisão (`ESPAÇO`).
- Ao errar o soco, o oponente fica **aberto e vulnerável** (cor alaranjada com braços abaixados) por uma fração de segundo, abrindo a oportunidade de contra-ataque do jogador.

### 6.3. Previsão da IA e Esquiva de Contra-Ataques
- Caso o jogador ataque fora de hora, a Árvore de Decisão prevê o golpe e esquiva/contra-ataca.
- **Evolução a partir do Nível 3.0x:** O oponente passa a ser capaz de prever e desviar até mesmo das tentativas de contra-ataque do jogador durante a janela de abertura, chegando a alta precisão evasiva ao atingir o Nível 4.0x.

### 6.4. Progressão por Rounds e Reinício em Derrota
- Em caso de vitória, ao pressionar `R`, avança-se para o próximo Round (`Round 1` $\rightarrow$ `Round 2`...), aumentando o nível da IA (`1.0x`, `1.5x`, `2.0x`...).
- Caso o jogador seja **nocauteado**, o sistema reseta a partida de volta ao **Round 1 (Nível 1.0x)** e limpa a memória de aprendizado recente do oponente.

---

## 7. INSTRUÇÕES DE EXECUÇÃO

No terminal na raiz do projeto:

```powershell
# 1. Executar o pipeline de Machine Learning (treinamento e gráficos em /graficos):
python experimentos_ml.py

# 2. Executar o jogo 2D:
python punch_out_game.py
```

### Comandos do Jogo:
- **`A`**: Soco Esquerdo
- **`D`**: Soco Direito
- **`ESPAÇO` / `W`**: Esquiva (esperar o adversário piscar em amarelo)
- **`S`**: Defesa (redução de dano recebido)
- **`R`**: Avançar para o Próximo Round / Reiniciar
