import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from io import BytesIO

# Configuração da página
st.set_page_config(page_title="Classificação Contábil 1.1", page_icon="🤖", layout="wide")

# Garantindo que os recursos do NLTK estejam disponíveis
try:
    stopwords.words('portuguese')
except LookupError:
    st.info("Baixando recursos do NLTK...")
    nltk.download('stopwords')

# Função de pré-processamento
@st.cache_data
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    stop_words = set(stopwords.words('portuguese'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Interface do usuário
st.title("🤖 Ferramenta de Classificação Contábil")
st.write("Faça o upload da sua planilha e a ferramenta irá classificar automaticamente as contas contábeis assim que o modelo for treinado.")
st.info(
    "**Instruções:**\n"
    "Sua planilha deve estar no formaoto Excel (.xlsx) e conter o mesmo layout da planilha fornecida\n"
    "A ferramenta usará as linhas preenchidas para aprender e treinar o modelo e depois preencherá as linhas vazias.\n"
)
# Upload do arquivo
uploaded_file = st.file_uploader("Escolha uma planilha Excel", type=["xlsx"])
if uploaded_file is not None:
    if st.button("Processar e classificar planilha"):
        with st.spinner("Lendo a planilha e realizando os testes. Por favor, aguarde..."):
            try:
                if uploaded_file.name.endswith('.xlsx'):
                    df = pd.read_excel(uploaded_file)
                else:
                    st.error("Por favor, envie um arquivo no formato .xlsx")
                    st.stop()
                col_his = 'DESCRIÇÃO DO LANÇAMENTO'
                col_acc = 'CONTA'
                
                # Separando dados para treinamento e classificação
                cond_for_class = df[col_acc].isna() | (df[col_acc].astype(str).str.strip() == '')
                df_for_class = df[cond_for_class].copy()
                df_training = df[~cond_for_class].copy()
                
                # Filtrando classes com poucas amostras
                acc_counts = df_training[col_acc].value_counts()
                cutoff = 5
                valid_accs = acc_counts[acc_counts >= cutoff].index
                df_filtered_training = df_training[df_training[col_acc].isin(valid_accs)].copy()
                df_filtered_training[col_acc] = df_filtered_training[col_acc].astype(int)
                
                # Treinamento do modelo
                df_filtered_training['DESCRIÇÃO LIMPA'] = df_filtered_training[col_his].apply(clean_text)
                X = df_filtered_training['DESCRIÇÃO LIMPA']
                y = df_filtered_training[col_acc]
                text_clf_pipeline = Pipeline([
                    ('tfidf', TfidfVectorizer(ngram_range=(1, 2))),
                    ('clf', LogisticRegression(random_state=42))
                ])
                text_clf_pipeline.fit(X, y)
                
                # Aplicando o modelo para classificação
                if not df_for_class.empty:
                    text_for_class = df_for_class[col_his].apply(clean_text)
                    arr_proba = text_clf_pipeline.predict_proba(text_for_class)
                    
                    filled_accs = []
                    final_proba = []
                    confidence_limit = 0.70 # % de confiança
                    
                    for i in range(len(arr_proba)): # para cada lançamento
                        lanc_proba = arr_proba[i]
                        best_acc_index = lanc_proba.argmax()
                        max_proba = lanc_proba[best_acc_index]
                        predicted_acc = text_clf_pipeline.classes_[best_acc_index]
                        if max_proba >= confidence_limit:
                            filled_accs.append(predicted_acc)
                            final_proba.append(f"{max_proba:.2%}") # convertendo para porcentagem
                        else:
                            filled_accs.append(f"Revisar") # não preencher
                            final_proba.append(f"Sugestão: {predicted_acc} ({max_proba:.2%})") # convertendo para porcentagem
                    df_for_class['CONTA'] = filled_accs
                    df_for_class['PROBABILIDADE'] = final_proba
                # Gerando planilha final com os resultados
                df_final = pd.concat([df_training, df_for_class], ignore_index=True)
                st.success("Classificação concluída!")
                st.write(f"Total de lançamentos processados: {len(df_final)}")
                st.dataframe(df_final.tail(len(df_for_class)+5)) # Mostra os classificados + 5 anteriores
                # Preparando arquivo para download
                output = BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    df_final.to_excel(writer, index=False, sheet_name='Classificação')
                processed_data = output.getvalue()
                st.download_button(
                    label="📥 Baixar planilha classificada",
                    data=processed_data,
                    file_name='classificacao_contas_resultado.xlsx',
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                )
            except Exception as e:
                st.error(f"Erro ao processar a planilha: {e}")
                st.error("Verifique se a planilha está no formato correto e tente novamente.")