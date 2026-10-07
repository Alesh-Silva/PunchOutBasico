import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

os.makedirs("graficos", exist_ok=True)
ACOES = ["Jab_Esq", "Direto_Dir", "Upper", "Esq_Esq", "Esq_Dir", "Bloqueio"]

# 1. Carregar ou Gerar Dados
if os.path.exists("dados_combate.csv"):
    df = pd.read_csv("dados_combate.csv")
else:
    np.random.seed(42)
    dados = []
    a1, a2, a3 = 0, 1, 0
    for r in range(1500):
        # Simula padrões de boxe: combos frequentes (0->1, 1->2) e repetições
        d = np.random.rand()
        prox = 1 if (a1 == 0 and d < 0.45) else (2 if (a1 == 1 and d < 0.45) else (a1 if d < 0.65 else np.random.randint(0, 6)))
        dados.append({"hp_jogador": 100, "stamina_jogador": 100, "acao_t_3": a3, "acao_t_2": a2, "acao_t_1": a1, "proxima_acao": prox})
        a3, a2, a1 = a2, a1, prox
    df = pd.DataFrame(dados)
    df.to_csv("dados_combate.csv", index=False)

# 2. Divisão Treino e Teste
X = df[["hp_jogador", "stamina_jogador", "acao_t_3", "acao_t_2", "acao_t_1"]]
y = df["proxima_acao"]
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3. Modelos
modelos = {
    "Regressão Logística": LogisticRegression(max_iter=1000, random_state=42),
    "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
    "Árvore de Decisão": DecisionTreeClassifier(max_depth=4, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=50, max_depth=6, random_state=42)
}

print(f"{'Modelo':<22} | {'Acurácia':<10} | {'F1-Score':<10} | {'CV 5-Fold':<15}")
print("-" * 65)

res = []
treinados = {}
matrizes = {}

for nome, mod in modelos.items():
    mod.fit(X_treino, y_treino)
    y_pred = mod.predict(X_teste)
    acc = accuracy_score(y_teste, y_pred)
    f1 = f1_score(y_teste, y_pred, average="weighted", zero_division=0)
    cv = cross_val_score(mod, X_treino, y_treino, cv=5).mean()
    
    treinados[nome] = mod
    matrizes[nome] = confusion_matrix(y_teste, y_pred, labels=list(range(6)))
    res.append({"Modelo": nome, "Acurácia": acc, "F1-Score": f1, "CV": cv})
    print(f"{nome:<22} | {acc*100:6.2f}%    | {f1*100:6.2f}%    | {cv*100:6.2f}%")

df_res = pd.DataFrame(res)

# 4. Gráficos
# 4.1 Barras Comparativo
plt.figure(figsize=(8, 4))
x = np.arange(len(df_res))
plt.bar(x - 0.15, df_res["Acurácia"] * 100, 0.3, label="Acurácia (%)", color="#2980b9")
plt.bar(x + 0.15, df_res["F1-Score"] * 100, 0.3, label="F1-Score (%)", color="#e67e22")
plt.xticks(x, df_res["Modelo"])
plt.ylim(0, 100)
plt.legend()
plt.title("Comparativo de Modelos (Punch-Out ML)")
plt.tight_layout()
plt.savefig("graficos/comparativo_desempenho_modelos.png")
plt.close()

# 4.2 Matrizes de Confusão
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ConfusionMatrixDisplay(matrizes["KNN (k=5)"], display_labels=ACOES).plot(ax=ax1, cmap="Blues", colorbar=False)
ConfusionMatrixDisplay(matrizes["Árvore de Decisão"], display_labels=ACOES).plot(ax=ax2, cmap="Blues", colorbar=False)
ax1.set_title("KNN (k=5)")
ax2.set_title("Árvore de Decisão")
plt.tight_layout()
plt.savefig("graficos/matrizes_confusao.png")
plt.close()

# 4.3 Importância dos Atributos
arvore = treinados["Árvore de Decisão"]
plt.figure(figsize=(7, 3.5))
plt.barh(["HP", "Stamina", "Ação t-3", "Ação t-2", "Ação t-1"], arvore.feature_importances_ * 100, color="#16a085")
plt.title("Importância dos Atributos (%)")
plt.tight_layout()
plt.savefig("graficos/importancia_atributos.png")
plt.close()

# 4.4 Visual da Árvore
plt.figure(figsize=(14, 7))
plot_tree(arvore, feature_names=list(X.columns), class_names=ACOES, filled=True, rounded=True, max_depth=3, fontsize=8)
plt.title("Regras da Árvore de Decisão")
plt.tight_layout()
plt.savefig("graficos/arvore_decisao_visual.png")
plt.close()

# 5. Salvar Modelo Campeão
joblib.dump({"modelo": arvore}, "modelo_punchout.joblib")
print("\n[OK] Modelos avaliados, gráficos gerados e 'modelo_punchout.joblib' salvo com sucesso!")
