import pandas as pd
import numpy as np
import pickle
import os
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import mlflow
import mlflow.sklearn

warnings.filterwarnings("ignore")

def train():
    print("Training Flight Pricing Regression Model...")
    with mlflow.start_run(run_name="Flight_Pricing_RF"):
        flight_df = pd.read_csv("data/flights.csv")
        flight_df = flight_df.sample(n=50000, random_state=42).reset_index(drop=True)

        flight_df['date'] = pd.to_datetime(flight_df['date'])
        flight_df['week_day'] = flight_df['date'].dt.weekday + 1
        flight_df['week_no'] = flight_df['date'].dt.isocalendar().week
        flight_df['day'] = flight_df['date'].dt.day

        flight_df.rename(columns={"to":"destination"}, inplace=True)

        features_ordering = [
            'from_Florianopolis (SC)', 'from_Sao_Paulo (SP)', 'from_Salvador (BH)', 
            'from_Brasilia (DF)', 'from_Rio_de_Janeiro (RJ)', 'from_Campo_Grande (MS)', 
            'from_Aracaju (SE)', 'from_Natal (RN)', 'from_Recife (PE)',
            
            'destination_Florianopolis (SC)', 'destination_Sao_Paulo (SP)', 
            'destination_Salvador (BH)', 'destination_Brasilia (DF)', 
            'destination_Rio_de_Janeiro (RJ)', 'destination_Campo_Grande (MS)', 
            'destination_Aracaju (SE)', 'destination_Natal (RN)', 'destination_Recife (PE)',
            
            'flightType_economic', 'flightType_firstClass', 'flightType_premium',
            'agency_Rainbow', 'agency_CloudFy', 'agency_FlyingDrops',
            'week_no', 'week_day', 'day'
        ]

        for feature in features_ordering:
            if feature.startswith('from_'):
                val = feature.replace('from_', '').replace('_', ' ')
                flight_df[feature] = (flight_df['from'] == val).astype(int)
            elif feature.startswith('destination_'):
                val = feature.replace('destination_', '').replace('_', ' ')
                flight_df[feature] = (flight_df['destination'] == val).astype(int)
            elif feature.startswith('flightType_'):
                val = feature.replace('flightType_', '')
                flight_df[feature] = (flight_df['flightType'] == val).astype(int)
            elif feature.startswith('agency_'):
                val = feature.replace('agency_', '')
                flight_df[feature] = (flight_df['agency'] == val).astype(int)

        X_flight = flight_df[features_ordering].fillna(0)
        y_flight = flight_df['price']

        travel_scaler = StandardScaler()
        X_flight_scaled = travel_scaler.fit_transform(X_flight)
        
        n_estimators = 50
        mlflow.log_param("n_estimators", n_estimators)
        travel_rf = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
        travel_rf.fit(X_flight_scaled, y_flight)

        os.makedirs("models", exist_ok=True)
        with open("models/scaling_1.pkl", "wb") as f:
            pickle.dump(travel_scaler, f)
        with open("models/rf_model_1.pkl", "wb") as f:
            pickle.dump(travel_rf, f)
        
        mlflow.sklearn.log_model(travel_rf, "flight_pricing_rf_model")
        
        print("Flight models saved successfully.")

if __name__ == "__main__":
    train()
