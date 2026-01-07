"""
NYC Taxi Trip Duration Prediction - Airflow DAG
"""
import pickle
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import xgboost as xgb
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import root_mean_squared_error
import mlflow

from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator


# MLflow configuration
MLFLOW_TRACKING_URI = "http://localhost:5001"
MLFLOW_EXPERIMENT_NAME = "nyc-taxi-experiment"
MODELS_FOLDER = Path('models')

# Default arguments for Airflow DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}


def setup_mlflow():
    """Initialize MLflow tracking and experiment."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    MODELS_FOLDER.mkdir(exist_ok=True)


def read_dataframe(year, month):
    """
    Read and preprocess taxi trip data.
    
    Args:
        year: Year of the data
        month: Month of the data
        
    Returns:
        Preprocessed DataFrame
    """
    url = (f'https://d37ci6vzurychx.cloudfront.net/trip-data/'
           f'green_tripdata_{year}-{month:02d}.parquet')
    df = pd.read_parquet(url)
    
    # Calculate trip duration in minutes
    df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)
    
    # Filter outliers
    df = df[(df.duration >= 1) & (df.duration <= 60)]
    
    # Create categorical features
    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']
    
    return df


def create_features(df, dv=None):
    """
    Create feature matrix from DataFrame.
    
    Args:
        df: Input DataFrame
        dv: DictVectorizer (optional, will be created if None)
        
    Returns:
        Tuple of (feature matrix, DictVectorizer)
    """
    categorical = ['PU_DO']
    numerical = ['trip_distance']
    dicts = df[categorical + numerical].to_dict(orient='records')
    
    if dv is None:
        dv = DictVectorizer(sparse=True)
        X = dv.fit_transform(dicts)
    else:
        X = dv.transform(dicts)
    
    return X, dv


def train_model_task(**context):
    """Airflow task to train the model."""
    setup_mlflow()
    
    # Get parameters from Airflow context
    year = context['params']['year']
    month = context['params']['month']
    
    # Load training data
    df_train = read_dataframe(year=year, month=month)
    
    # Load validation data (next month)
    next_year = year if month < 12 else year + 1
    next_month = month + 1 if month < 12 else 1
    df_val = read_dataframe(year=next_year, month=next_month)
    
    # Create features
    X_train, dv = create_features(df_train)
    X_val, _ = create_features(df_val, dv)
    
    # Prepare targets
    target = 'duration'
    y_train = df_train[target].values
    y_val = df_val[target].values
    
    # Train with MLflow tracking
    with mlflow.start_run() as run:
        train = xgb.DMatrix(X_train, label=y_train)
        valid = xgb.DMatrix(X_val, label=y_val)
        
        best_params = {
            'learning_rate': 0.09585355369315604,
            'max_depth': 30,
            'min_child_weight': 1.060597050922164,
            'objective': 'reg:squarederror',  # Updated from deprecated reg:linear
            'reg_alpha': 0.018060244040060163,
            'reg_lambda': 0.011658731377413597,
            'seed': 42
        }
        mlflow.log_params(best_params)
        
        booster = xgb.train(
            params=best_params,
            dtrain=train,
            num_boost_round=30,
            evals=[(valid, 'validation')],
            early_stopping_rounds=50
        )
        
        # Evaluate model
        y_pred = booster.predict(valid)
        rmse = root_mean_squared_error(y_val, y_pred)
        mlflow.log_metric("rmse", rmse)
        
        # Save preprocessor
        preprocessor_path = MODELS_FOLDER / "preprocessor.b"
        with open(preprocessor_path, "wb") as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact(str(preprocessor_path), artifact_path="preprocessor")
        
        # Log model
        mlflow.xgboost.log_model(booster, artifact_path="models_mlflow")
        
        run_id = run.info.run_id
        print(f"MLflow run_id: {run_id}")
        
        # Push run_id to XCom for downstream tasks
        context['task_instance'].xcom_push(key='run_id', value=run_id)
        
        return run_id


def save_run_id_task(**context):
    """Save the MLflow run_id to a file."""
    ti = context['task_instance']
    run_id = ti.xcom_pull(task_ids='train_model', key='run_id')
    
    with open("run_id.txt", "w") as f:
        f.write(run_id)
    
    print(f"Saved run_id: {run_id}")


# Define the DAG
with DAG(
    'nyc_taxi_training_pipeline',
    default_args=default_args,
    description='Train NYC taxi trip duration prediction model',
    schedule='@monthly',  # Run monthly
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['mlops', 'xgboost', 'taxi'],
    params={
        'year': 2024,
        'month': 1
    }
) as dag:
    
    train_model = PythonOperator(
        task_id='train_model',
        python_callable=train_model_task,
    )
    
    save_run_id = PythonOperator(
        task_id='save_run_id',
        python_callable=save_run_id_task,
    )
    
    # Define task dependencies
    train_model >> save_run_id