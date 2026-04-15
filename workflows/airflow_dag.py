from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'mlops',
    'depends_on_past': False,
    'start_date': datetime(2026, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'mlops_training_pipeline_v2',
    default_args=default_args,
    description='Streamlined training pipeline with MLflow tracking',
    schedule_interval=timedelta(days=1),
    catchup=False
) as dag:

    train_flight_pricing = BashOperator(
        task_id='train_flight_pricing',
        bash_command='python ../models/regression_model.py',
    )

    train_gender_classification = BashOperator(
        task_id='train_gender_classification',
        bash_command='python ../models/gender_classification_model.py',
    )

    train_hotel_recommender = BashOperator(
        task_id='train_hotel_recommender',
        bash_command='python ../models/recommendation_model.py',
    )

    # Execute in parallel
    [train_flight_pricing, train_gender_classification, train_hotel_recommender]
