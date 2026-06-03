# 📊 Alinhamento de Sentimento em E-commerce: Do Léxico ao Deep Learning

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Google Colab](https://img.shields.io/badge/Google%20Colab-GPU%20T4-orange.svg)](https://colab.research.google.com/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Transformers-yellow.svg)](https://huggingface.co/)

## 📝 Cenário e Desafio de Negócio

Em plataformas de e-commerce, existe frequentemente um abismo entre a **nota em estrelas (1 a 5)** dada pelo usuário e o **comentário em texto**. Clientes frustrados podem dar notas medianas por engano, ou elogiar o produto mas criticar severamente a entrega.

**O Objetivo:** Construir um pipeline de Processamento de Linguagem Natural (PLN) capaz de analisar o comentário textual do cliente e prever, com máxima precisão, a real intenção de nota do usuário. Isso permite auditar avaliações falsas, disparar gatilhos automáticos para o time de CS (Customer Success) e entender o real humor do consumidor.

---

## 🚀 A Evolução Arquitetural (As 3 Eras do PLN)

O projeto foi construído simulando a evolução histórica das tecnologias de análise de sentimento, dividida em três abordagens computacionais:

### 1️⃣ Era Ancestral: VADER (Abordagem Léxica)
* **Como funciona:** Baseado em dicionários estáticos de palavras com pesos sentimentais.
* **O Problema:** Projetado originalmente para o inglês. Mesmo adaptado, falha severamente em capturar o contexto, gírias e ironias do português brasileiro, jogando a maior parte dos dados na neutralidade (efeito linha reta).

### 2️⃣ Era de Adaptação: LeIA + Regras de Negócio (Léxico Customizado)
* **Como funciona:** Uma variação do VADER otimizada especificamente para o português, combinada com camadas de pós-processamento e regras de negócio para calibrar os scores.
* **O Problema:** Embora melhore o entendimento local, ainda sofre com o "efeito textão" — precisa de um volume muito alto de adjetivos para conseguir cravar uma nota com precisão.

### 3️⃣ Era Estado da Arte: PySentimiento (Deep Learning / Transformers)
* **Como funciona:** Modelo baseado na arquitetura **RoBERTa** (Transformers), treinado especificamente para análises de sentimentos e dinâmicas de redes sociais em espanhol/português.
* **O Ganho:** Entendimento contextual profundo. Ele não apenas lê palavras isoladas, mas entende a ordem, sarcasmos, negações e a semântica completa da frase, independentemente do tamanho do texto.

---

## 📊 Resultados e Benchmarks Estatísticos

Abaixo está o consolidado das métricas de eficiência (Classificação Categórica) contra as métricas de distanciamento e ordenação (Regressão e Estatística Descritiva):

| Métrica Avaliada | VADER (Léxico) | LeIA + RN (Customizado) | PySentimiento (Transformer) | *Direção Ideal* |
| :--- | :---: | :---: | :---: | :---: |
| **Acurácia Geral** | 40.94% | 44.82% | **62.29%** | *(↑ Quanto maior, melhor)* |
| **F1-Score Weighted** | 0.354 | 0.443 | **0.613** | *(↑ Quanto maior, melhor)* |
| **F1-Score Macro (Rígido)** | 0.231 | 0.380 | **0.584** | *(↑ Quanto maior, melhor)* |
| **Coeficiente de Gini** | 0.323 | 0.457 | **0.669** | *(↑ Quanto maior, melhor)* |
| **MAE (Erro Médio Absoluto)** | 0.940 | 0.814 | **0.490** | *(↓ Quanto menor, melhor)* |
| **MSE (Erro Quadrático Médio)** | 1.488 | 1.157 | **0.697** | *(↓ Quanto menor, melhor)* |

### 🔍 Principais Insights do Painel:
* **Quebra do Paradigma Léxico:** O PySentimiento aumentou a taxa de acerto categórico absoluta (**Acurácia**) para **62.29%**, mitigando os problemas de desbalanceamento de classes severos comuns em e-commerce (provado pelo **F1-Macro** de **0.584**).
* **Mitigação de Erros Catastróficos:** O Transformer reduziu o erro médio de proximidade (**MAE**) para **0.490**, garantindo que, quando o modelo desliza, ele erra por "menos de meia estrela" de distância da opinião real do usuário, eliminando distorções grotescas (como classificar um detrator nota 1 como promotor nota 5), o que é corroborado pelo tombo do **MSE para 0.697**.

---

## 🛠️ Otimização de Performance e Engenharia de Dados

Para viabilizar o projeto em produção com grandes volumes de dados, o pipeline do PySentimiento foi refatorado para abandonar o processamento linha a linha tradicional do Pandas (`.apply()`). 

Implementou-se o **Processamento em Lote (Batch Processing)** aproveitando os tensores paralelos da **GPU NVIDIA T4** no ambiente. Isso permitiu que milhares de registros fossem processados simultaneamente em poucos segundos, reduzindo drasticamente o custo computacional e o tempo de esteira.

---

## 📂 Estrutura do Repositório

* `Notebook_Principal.ipynb`: Pipeline ponta a ponta contendo extração, tratamento de nulos, isolamento de regras de negócio, inferência dos modelos e validação estatística.
* `images/`: Gráficos gerados pelas bibliotecas Matplotlib e Seaborn para os relatórios executivos.

## ⚙️ Como Executar o Projeto

1. Abra o `Notebook_Principal.ipynb` no **Google Colab**.
2. Altere o ambiente de execução para utilizar o acelerador de hardware **GPU T4**.
3. Execute todas as células (`Ctrl + F9`) para baixar as dependências do Hugging Face e processar a base de dados automaticamente.

---
