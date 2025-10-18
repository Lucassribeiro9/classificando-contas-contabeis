# Classificador de Contas Contábeis com Machine Learning

## 📖 Visão Geral

Este projeto é uma ferramenta de automação contábil que utiliza Machine Learning para classificar lançamentos financeiros em suas respectivas contas contábeis. A solução consiste em uma aplicação web desenvolvida com Streamlit que permite ao usuário fazer o upload de uma planilha Excel, treinar um modelo de classificação de texto em tempo real e receber de volta a planilha com as contas em branco devidamente preenchidas.

O núcleo da ferramenta é um modelo de Processamento de Linguagem Natural (PLN) que analisa a coluna "DESCRIÇÃO DO LANÇAMENTO" para prever a "CONTA" correta.

## ✨ Funcionalidades Principais

-   **Treinamento Dinâmico:** O modelo é treinado na hora com os dados fornecidos pelo próprio usuário, garantindo que ele se adapte ao contexto específico de cada empresa.
-   **Classificação Inteligente:** Lançamentos com a conta contábil em branco são classificados automaticamente.
-   **Regra de Confiança:**
    -   Se a probabilidade da previsão for **maior ou igual a 70%**, a conta é preenchida diretamente.
    -   Se a probabilidade for **menor que 70%**, a ferramenta preenche com a conta mais provável, mas adiciona um aviso de **"Revisar"**, garantindo um controle de qualidade humano.
-   **Interface Web Simples:** Interface intuitiva criada com Streamlit para upload e download de arquivos.
-   **Download Fácil:** O resultado final é disponibilizado em um novo arquivo Excel, pronto para ser baixado.

## 🛠️ Tecnologias Utilizadas

-   **Linguagem:** Python 3
-   **Análise de Dados e ML:** Pandas, Scikit-learn, NLTK
-   **Interface Web:** Streamlit
-   **Análise Exploratória:** O notebook `analise-contabil.ipynb` contém toda a análise, desenvolvimento e avaliação do modelo, utilizando também Plotly e Seaborn para visualizações.

## 🚀 Como Executar o Projeto Localmente

Siga os passos abaixo para executar a aplicação em sua máquina.

**1. Clone o Repositório**
```bash
git clone https://github.com/seu-usuario/classificando-contas-contabeis.git
cd classificando-contas-contabeis
```

**2. Crie e Ative um Ambiente Virtual**
```bash
# Criar o ambiente
python3 -m venv venv

# Ativar no Linux/macOS
source venv/bin/activate

# Ativar no Windows
.\venv\Scripts\activate
```

**3. Instale as Dependências**
Crie um arquivo `requirements.txt` com o seguinte conteúdo:
```
# filepath: requirements.txt
pandas
numpy
openpyxl
scikit-learn
nltk
streamlit
plotly
seaborn
matplotlib
xlsxwriter
joblib
```
E então instale as dependências:
```bash
pip install -r requirements.txt
```

**4. Execute a Aplicação Streamlit**
```bash
streamlit run app.py
```
Abra o navegador no endereço local fornecido pelo Streamlit (geralmente `http://localhost:8501`).

## 📋 Como Usar a Ferramenta

1.  Na página da aplicação, baixe o **modelo de planilha** para garantir que seus dados estejam no formato correto.
2.  Preencha a planilha com seus dados. As linhas que você deseja classificar devem ter a coluna `CONTA` vazia. As linhas já preenchidas serão usadas para treinar o modelo.
3.  Faça o upload da sua planilha na área de "Envio da planilha para classificação".
4.  Clique no botão **"Processar e classificar planilha"**.
5.  Aguarde o processamento. Ao final, uma prévia dos resultados será exibida.
6.  Clique no botão **"Baixar planilha classificada"** para obter o arquivo final.