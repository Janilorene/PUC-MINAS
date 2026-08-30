import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv('train.csv')
print("=" * 65)
print("1. VISUALIZAÇÃO DA BASE DE DADOS")
print("=" * 65)
print(f"Total de registros: {df.shape[0]} linhas, {df.shape[1]} colunas")
print("\nValores nulos por coluna:")
print(df.isnull().sum()[df.isnull().sum() > 0])
print("\nDistribuição da Classe Alvo (Survived):")
print(df['Survived'].value_counts(normalize=True).rename({0: 'Não Sobreviveu (Morto)', 1: 'Sobreviveu'}))

# -------------------------------------------------------------
#  CODIFICAÇÃO E TRATAMENTO DOS ATRIBUTOS
# -------------------------------------------------------------
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df = pd.get_dummies(df, columns=['Embarked'], drop_first=False, dtype=int)

feature_cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_C', 'Embarked_Q', 'Embarked_S']
X = df[feature_cols]
y = df['Survived']

# -------------------------------------------------------------
# TREINAMENTO DA ÁRVORE E EXTRAÇÃO DAS REGRAS
# -------------------------------------------------------------
arvore = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=42)
arvore.fit(X, y)

print("\n" + "=" * 65)
print("2. ESTRUTURA DA ÁRVORE DE DECISÃO")
print("=" * 65)
print(export_text(arvore, feature_names=feature_cols))

print("\n" + "=" * 65)
print("3. REGRAS OBTIDAS E PADRÃO DE MORTALIDADE")
print("=" * 65)
tree_ = arvore.tree_
def extrair_regras(node, caminho):
    if tree_.feature[node] != -2:
        nome = feature_cols[tree_.feature[node]]
        limiar = tree_.threshold[node]
        extrair_regras(tree_.children_left[node], caminho + [f"({nome} <= {limiar:.2f})"])
        extrair_regras(tree_.children_right[node], caminho + [f"({nome} > {limiar:.2f})"])
    else:
        probs = tree_.value[node][0]
        amostras = tree_.n_node_samples[node]
        mortos = int(round(probs[0] * amostras)) if probs[0] <= 1.0 else int(probs[0])
        vivos = int(round(probs[1] * amostras)) if probs[1] <= 1.0 else int(probs[1])
        taxa_morte = (mortos / amostras) * 100
        predicao = "NÃO SOBREVIVEU (MORTO)" if mortos >= vivos else "SOBREVIVEU"
        
        condicoes = " E ".join(caminho)
        print(f"SE {condicoes} ENTÃO -> {predicao}")
        print(f"   Amostras: {amostras} | Mortos: {mortos} ({taxa_morte:.1f}%) | Vivos: {vivos} ({100-taxa_morte:.1f}%)\n")

extrair_regras(0, [])

# Gráfico da Árvore
plt.figure(figsize=(18, 9))
plot_tree(arvore, feature_names=feature_cols, class_names=['Morto', 'Sobreviveu'], filled=True, rounded=True, fontsize=9)
plt.title("Árvore de Decisão - Padrão de Mortalidade Titanic")
plt.tight_layout()
plt.show()