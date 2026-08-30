import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import classification_report, confusion_matrix

base = pd.read_csv('restaurante.csv', sep=';')

X = base.iloc[:, :-1].copy()
Y = base.iloc[:, -1]

# Codificação dos atributos
mapa_binario = {'Nao': 0, 'Sim': 1}
colunas_binarias = [c for c in X.columns if set(X[c].dropna().unique()).issubset(mapa_binario.keys())]
X[colunas_binarias] = X[colunas_binarias].apply(lambda c: c.map(mapa_binario))

mapa_cliente = {'Nenhum': 0, 'Algum': 1, 'Alguns': 1, 'Cheio': 2}
X['Cliente'] = X['Cliente'].map(mapa_cliente)
X['Preco'] = X['Preco'].map({'R': 0, 'RR': 1, 'RRR': 2})
X['Tempo'] = X['Tempo'].map({'0-10': 0, '10-30': 1, '30-60': 2, '>60': 3})
X = pd.get_dummies(X, columns=['Tipo'], dtype=int)

#Divisão treino e teste (80% treino, 20% teste com random_state=42)
X_treino, X_teste, y_treino, y_teste = train_test_split(X, Y, test_size=0.20, random_state=42)

#Treinamento da arvore
arvore = DecisionTreeClassifier(criterion='entropy', random_state=42)
arvore.fit(X_treino, y_treino)

print("=" * 60)
print("ITEM 1: Árvore Obtida (Estrutura Textual)")
print("=" * 60)
print(export_text(arvore, feature_names=list(X.columns)))

previsoes = arvore.predict(X_teste)
print("=" * 60)
print("ITEM 2: Métricas de Avaliação (Teste)")
print("=" * 60)
print("Matriz de Confusão (Linhas: Real [Nao, Sim], Colunas: Predito [Nao, Sim]):")
print(confusion_matrix(y_teste, previsoes, labels=['Nao', 'Sim']))
print("\nRelatório de Classificação:")
print(classification_report(y_teste, previsoes, labels=['Nao', 'Sim'], target_names=['Nao', 'Sim']))

print("=" * 60)
print("ITENS 3 e 4: Regras Extraídas e Qualidade (Suporte e Confiança no Treino)")
print("=" * 60)
def extrair_regras(tree, feature_names, class_names, total_amostras):
    tree_ = tree.tree_
    regras = []
    def percorrer(node, condicoes):
        if tree_.feature[node] != -2: # Nó interno
            nome_feat = feature_names[tree_.feature[node]]
            limiar = tree_.threshold[node]
            percorrer(tree_.children_left[node], condicoes + [f"({nome_feat} <= {limiar:.2f})"])
            percorrer(tree_.children_right[node], condicoes + [f"({nome_feat} > {limiar:.2f})"])
        else: # Folha
            contagens = tree_.value[node][0]
            classe_pred = class_names[np.argmax(contagens)]
            n_samples = tree_.n_node_samples[node]
            acertos = int(max(contagens))
            confianca = (acertos / n_samples) * 100
            suporte = (n_samples / total_amostras) * 100
            regras.append((condicoes, classe_pred, n_samples, suporte, confianca))
    percorrer(0, [])
    return regras

regras = extrair_regras(arvore, list(X.columns), list(arvore.classes_), len(X_treino))
for i, (conds, classe, n_samp, sup, conf) in enumerate(regras, 1):
    texto_cond = " E ".join(conds)
    print(f"Regra {i}: SE {texto_cond} ENTÃO Conclusão = '{classe}'")
    print(f"   -> Suporte: {n_samp}/9 ({sup:.1f}%) | Confiança/Pureza: {conf:.1f}%\n")

plt.figure(figsize=(16, 8))
plot_tree(arvore, feature_names=X.columns, class_names=list(arvore.classes_), filled=True, rounded=True)
plt.title("Árvore de Decisão - Restaurante")
plt.show()