import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
from sklearn.metrics import log_loss, f1_score
import pycaret.classification as pc

dataset_prod_location = 'C:/Projetos/Projeto Kobe/projeto_kobe/data/raw/dataset_kobe_prod.parquet'

def load_model_from_mlflow(registered_model_name):
    client = MlflowClient()
    model_uri = f"models:/{registered_model_name}/staging"
    model = mlflow.pyfunc.load_model(model_uri)
    return model

prod_data = pd.read_parquet(dataset_prod_location)

with mlflow.start_run(run_name='PipelineAplicacao'):
    model = load_model_from_mlflow(registered_model_name)
    
    prod_data['prediction'] = model.predict(prod_data.drop(['shot_made_flag'], axis=1))
    prod_data['prediction_proba'] = model.predict_proba(prod_data.drop(['shot_made_flag'], axis=1))[:, 1]

    new_log_loss = log_loss(prod_data['shot_made_flag'], prod_data['prediction_proba'])
    new_f1_score = f1_score(prod_data['shot_made_flag'], prod_data['prediction'])

    mlflow.log_metrics({'log_loss': new_log_loss, 'f1_score': new_f1_score})

    results_location = "results.parquet"
    prod_data.to_parquet(results_location)
    mlflow.log_artifact(results_location)
