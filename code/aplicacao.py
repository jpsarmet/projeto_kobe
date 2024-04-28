import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.metrics import log_loss, f1_score, confusion_matrix
from mlflow.pyfunc import load_model
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

data_path = "./data/processed/data_filtered.parquet"
data_prod = pd.read_parquet(data_path)
#data_cols = ['lat', 'lon', 'minutes_remaining', 'period', 'playoffs', 'shot_distance', 'shot_made_flag']
#data_prod = data_prod[data_cols]
#data_prod['shot_made_flag'] = pd.to_numeric(data_prod['shot_made_flag'], errors='coerce')
#data_prod = data_prod.dropna()

# Configurar o ambiente do MLflow
mlflow.set_tracking_uri("sqlite:///code/mlruns.db")  
client = MlflowClient()

# Obter o modelo registrado
model_uri = "models:/model_kobe/staging"
model = load_model(model_uri)

# Iniciar a execução do MLflow
with mlflow.start_run(run_name="PipelineAplicacao") as run:
    # Aplicar o modelo para obter previsões
    predictions = model.predict(data_prod.drop('shot_made_flag', axis=1))
    data_prod['predictions'] = predictions

    # Calcular métricas
    f1 = f1_score(data_prod['shot_made_flag'], predictions)
    logloss = log_loss(data_prod['shot_made_flag'], predictions)
    mlflow.log_metrics({'log_loss': logloss, 'f1_score': f1})

    # Gerar a matriz de confusão
    conf_matrix = confusion_matrix(data_prod['shot_made_flag'], predictions)
    
    # Plotar a matriz de confusão
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Blues")
    plt.title('Matriz de Confusão')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('confusion_matrix.png')
    plt.close()

    # Logar a figura da matriz de confusão no MLflow
    mlflow.log_artifact('confusion_matrix.png')

    # Salvar a tabela com os resultados
    results_path = "base_saved.parquet"
    data_prod.to_parquet(results_path)
    mlflow.log_artifact(results_path)

    # Exibir as métricas no console
    print(f"Log Loss: {logloss}, F1 Score: {f1}")

# As informações da execução estão agora disponíveis no objeto 'run'
print(f"Run ID: {run.info.run_id}")

# Função para carregar dados e realizar previsões
def load_data_and_predict(model_uri, data_path):
    data_prod = pd.read_parquet(data_path)
    model = load_model(model_uri)
    predictions = model.predict(data_prod.drop('shot_made_flag', axis=1))
    data_prod['predictions'] = predictions
    return data_prod

# Função para calcular métricas
def calculate_metrics(data):
    # Remover linhas com NaN em 'shot_made_flag' ou 'predictions'
    data = data.dropna(subset=['shot_made_flag', 'predictions'])
    logloss = log_loss(data['shot_made_flag'], data['predictions'])
    f1 = f1_score(data['shot_made_flag'], data['predictions'])
    return logloss, f1

# Função para plotar a matriz de confusão
def plot_confusion_matrix(data):
    conf_matrix = confusion_matrix(data['shot_made_flag'], data['predictions'])
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Blues", ax=ax)
    ax.set_title('Matriz de Confusão')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    return fig

# Inicializando o Streamlit app
st.title("Monitoramento do Modelo em Produção")

# Botão para carregar dados e realizar previsões
if st.button('Carregar dados e atualizar previsões'):
    data_prod = load_data_and_predict(model_uri, data_path)

    # Verificar se há valores NaN em 'shot_made_flag' ou 'predictions'
    if data_prod['shot_made_flag'].isna().any() or data_prod['predictions'].isna().any():
        st.error('Os dados de previsão ou as etiquetas verdadeiras contêm NaNs.')
    else:
        logloss, f1 = calculate_metrics(data_prod)
        
        # Exibindo métricas
        st.metric(label="Log Loss", value=logloss)
        st.metric(label="F1 Score", value=f1)
        
        # Plotando a matriz de confusão
        fig = plot_confusion_matrix(data_prod)
        st.pyplot(fig)

        # Métrica de monitoramento simples (por exemplo: percentual de previsões positivas)
        percent_positive_predictions = (data_prod['predictions'].mean() * 100)
        st.metric(label="Percentual de Previsões Positivas", value=f"{percent_positive_predictions:.2f}%")
