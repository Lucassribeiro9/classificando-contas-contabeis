import streamlit as st
import joblib
import pandas as pd

# Configuração da página
st.set_page_config(page_title="Classificação Contábil", page_icon=":chart_with_upwards_trend:", layout="wide")
# Carregando o modelo treinado
# Usando cache do streamlit para carregar o modelo apenas uma vez
@st.cache_resource
def load_model():
    try:
        model = joblib.load('modelo_contabil_pipeline.joblib')
        return model
    except FileNotFoundError:
        st.error("Modelo não encontrado. Certifique-se de que 'modelo_contabil_pipeline.joblib' está no diretório correto.")
        return None
    
model = load_model()

# Interface do usuário
st.title("Classificador Contábil de Lançamentos")
st.write("Insira o histórico do lançamento contábil. O modelo irá sugerir a conta mais provável.")

# Campo de entrada para o histórico do lançamento
story_input = st.text_area("Digite o histórico do Lançamento", height=100, placeholder="Ex: Pagamento Licença Anual Software Adobe")

# Botão para classificar
if st.button("Classificar"):
    if model is not None and story_input.strip() != "":
        # O modelo espera uma lista de entradas
        input_data = [story_input]
        # Fazendo a previsão
        try:
            proba = model.predict_proba(input_data)
            # Obter o nome das classes
            classes = model.classes_
            # Criando dataframe para exibir resultados
            df_results = pd.DataFrame(proba, columns=classes).T
            df_results.rename(columns={0: 'Probabilidade'}, inplace=True)
            df_results['Probabilidade'] = df_results['Probabilidade'] * 100  # Convertendo para porcentagem
            
            # Exibindo os resultados
            df_results_sorted = df_results.sort_values(by='Probabilidade', ascending=False)
            st.subheader("Resultado da Classificação")
            best_acc = df_results_sorted.index[0]
            best_prob = df_results_sorted.iloc[0,0]
            
            # Exibindo a melhor conta sugerida
            st.success(f"**Sugestão de Conta:** {best_acc} com probabilidade de {best_prob:.2f}%")
            
            # Melhores previsões
            st.write("---")
            st.write("**Contas com melhores previsões:**")
            st.dataframe(df_results_sorted.head(5).style.format("{:.2f}%}"), use_container_width=True)
        except Exception as e:
            st.error(f"Erro ao classificar: {e}")
    else:
        st.warning("Por favor, insira um histórico válido para classificação.")