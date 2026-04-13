import pandas as pd
import numpy as np
import pickle
import os
import warnings
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer

warnings.filterwarnings("ignore")

print("Training Travel ML System (Flight Pricing)...")
# 1. TRAVEL ML SYSTEM
flight_df = pd.read_csv("Datasets/flights.csv")
# Sample to 50k for quick class presentation training to avoid extremely long waits
flight_df = flight_df.sample(n=50000, random_state=42).reset_index(drop=True)

# Parse dates
flight_df['date'] = pd.to_datetime(flight_df['date'])
flight_df['week_day'] = flight_df['date'].dt.weekday + 1 # offset adjusting
flight_df['week_no'] = flight_df['date'].dt.isocalendar().week
flight_df['day'] = flight_df['date'].dt.day

flight_df.rename(columns={"to":"destination"}, inplace=True)

# The frontend passes exactly 27 features ordered appropriately or by dict insertion
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

# One-hot encode manually to perfectly match the 27 dimensions without any dropping
for feature in features_ordering:
    if feature.startswith('from_'):
        val = feature.replace('from_', '')
        # Handle underscores from python dict
        val = val.replace('_', ' ')
        flight_df[feature] = (flight_df['from'] == val).astype(int)
    elif feature.startswith('destination_'):
        val = feature.replace('destination_', '')
        val = val.replace('_', ' ')
        flight_df[feature] = (flight_df['destination'] == val).astype(int)
    elif feature.startswith('flightType_'):
        val = feature.replace('flightType_', '')
        flight_df[feature] = (flight_df['flightType'] == val).astype(int)
    elif feature.startswith('agency_'):
        val = feature.replace('agency_', '')
        flight_df[feature] = (flight_df['agency'] == val).astype(int)

X_flight = flight_df[features_ordering].fillna(0)
y_flight = flight_df['price']

print(f"X_flight shape: {X_flight.shape}")

travel_scaler = StandardScaler()
X_flight_scaled = travel_scaler.fit_transform(X_flight)
travel_rf = RandomForestRegressor(n_estimators=50, random_state=42)
travel_rf.fit(X_flight_scaled, y_flight)

with open("Travel_ML_System/model/scaling_1.pkl", "wb") as f:
    pickle.dump(travel_scaler, f)
with open("Travel_ML_System/model/rf_model.pkl", "wb") as f:
    pickle.dump(travel_rf, f)
with open("Travel_ML_System/model/rf_model_1.pkl", "wb") as f:
    pickle.dump(travel_rf, f)
print("Saved Travel_ML_System models.")


print("\nTraining Gender Classification...")
# 2. GENDER CLASSIFICATION
user_df = pd.read_csv("Datasets/users.csv")
user_df = user_df[user_df['gender'].isin(['male', 'female'])].reset_index(drop=True)

# Encode company (the inference app uses LabelEncoder)
le = LabelEncoder()
user_df['company_encoded'] = le.fit_transform(user_df['company'])

# Target
le_gender = LabelEncoder()
y_gender = le_gender.fit_transform(user_df['gender'])

# NLP embeddings
model = SentenceTransformer('flax-sentence-embeddings/all_datasets_v4_MiniLM-L6')
user_df['name'] = user_df['name'].fillna('').astype(str)
embeddings = np.array(user_df['name'].apply(lambda text: model.encode(text)).tolist())

# PCA
pca = PCA(n_components=23, random_state=42)
text_embeddings_pca = pca.fit_transform(embeddings)

# Concat with code, company_encoded, age
X_numerical = user_df[['code', 'company_encoded', 'age']].fillna(user_df[['code', 'company_encoded', 'age']].median()).values
X_gender = np.hstack((text_embeddings_pca, X_numerical))

print(f"X_gender shape: {X_gender.shape}")

gender_scaler = StandardScaler()
X_gender_scaled = gender_scaler.fit_transform(X_gender)
lr = LogisticRegression(random_state=42, max_iter=500)
lr.fit(X_gender_scaled, y_gender)

with open("Gender Classification Model/model/pca_1.pkl", "wb") as f:
    pickle.dump(pca, f)
with open("Gender Classification Model/model/scaler_1.pkl", "wb") as f:
    pickle.dump(gender_scaler, f)
with open("Gender Classification Model/model/tuned_logistic_regression_model_1.pkl", "wb") as f:
    pickle.dump(lr, f)

print("Saved Gender Classification Models.")
print("\nAll real mathematical models have been successfully trained and serialized!")
