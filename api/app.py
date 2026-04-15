import pandas as pd
import numpy as np
import pickle
import os
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings("ignore")

app = Flask(__name__)

# --- Load Models for Flight Pricing ---
try:
    flight_scaler = pickle.load(open("models/scaling_1.pkl", 'rb'))
    flight_rf = pickle.load(open("models/rf_model_1.pkl", 'rb'))
except Exception as e:
    print(f"Warning: Could not load flight models: {e}")

# --- Load Models for Gender Classification ---
try:
    nlp_model = SentenceTransformer('flax-sentence-embeddings/all_datasets_v4_MiniLM-L6')
    gender_scaler = pickle.load(open("models/scaler_1.pkl", 'rb'))
    gender_pca = pickle.load(open("models/pca_1.pkl", 'rb'))
    gender_lr = pickle.load(open("models/tuned_logistic_regression_model_1.pkl", 'rb'))
except Exception as e:
    print(f"Warning: Could not load gender classification models: {e}")

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "service": "MLOps Unified API"})

@app.route('/', methods=['GET'])
def index():
    return jsonify({
        "service": "MLOps Unified API",
        "status": "running",
        "available_endpoints": [
            "/health",
            "/predict_flight",
            "/predict_gender"
        ]
    })

# --- Predict Flight Price Endpoint ---
def predict_flight_price_logic(input_data):
    df_input = pd.DataFrame([input_data])
    X = flight_scaler.transform(df_input)
    y_pred = flight_rf.predict(X)
    return y_pred[0]

@app.route('/predict_flight', methods=['POST'])
def predict_flight():
    try:
        data = request.json
        boarding = 'from_' + data.get('from', '')
        destination = 'destination_' + data.get('destination', '')
        selected_flight_class = 'flightType_' + data.get('flightType', '')
        selected_agency = 'agency_' + data.get('agency', '')
        
        boarding_city_list = ['from_Florianopolis (SC)', 'from_Sao_Paulo (SP)', 'from_Salvador (BH)',
                              'from_Brasilia (DF)', 'from_Rio_de_Janeiro (RJ)', 'from_Campo_Grande (MS)',
                              'from_Aracaju (SE)', 'from_Natal (RN)', 'from_Recife (PE)']
        destination_city_list = ['destination_Florianopolis (SC)', 'destination_Sao_Paulo (SP)', 'destination_Salvador (BH)',
                                 'destination_Brasilia (DF)', 'destination_Rio_de_Janeiro (RJ)', 'destination_Campo_Grande (MS)',
                                 'destination_Aracaju (SE)', 'destination_Natal (RN)', 'destination_Recife (PE)']
        class_list = ['flightType_economic', 'flightType_firstClass', 'flightType_premium']
        agency_list = ['agency_Rainbow', 'agency_CloudFy', 'agency_FlyingDrops']

        travel_dict = dict()
        for city in boarding_city_list:
            travel_dict[city] = 1 if city == boarding else 0
        for city in destination_city_list:
            travel_dict[city] = 1 if city == destination else 0
        for flight_class in class_list:
            travel_dict[flight_class] = 1 if flight_class == selected_flight_class else 0
        for agency in agency_list:
            travel_dict[agency] = 1 if agency == selected_agency else 0
            
        travel_dict['week_no'] = int(data.get('week_no', 0))
        travel_dict['week_day'] = int(data.get('week_day', 0))
        travel_dict['day'] = int(data.get('day', 0))

        predicted_price = str(round(predict_flight_price_logic(travel_dict), 2))
        return jsonify({'prediction': predicted_price})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# --- Predict Gender Endpoint ---
def predict_gender_logic(input_data):
    df = pd.DataFrame([input_data])
    label_encoder = LabelEncoder()
    df['company_encoded'] = label_encoder.fit_transform(df['company'])
    
    df['name_embedding'] = df['name'].apply(lambda text: nlp_model.encode(text))
    
    n_components = 23
    text_embeddings_pca = np.empty((len(df), n_components * 1))
    embeddings = df['name_embedding'].values.tolist()
    embeddings_pca = gender_pca.transform(embeddings)
    text_embeddings_pca[:, 0:n_components] = embeddings_pca

    X_numerical = df[['code', 'company_encoded', 'age']].fillna(0).values
    X = np.hstack((text_embeddings_pca, X_numerical))
    X = gender_scaler.transform(X)
    
    y_pred = gender_lr.predict(X)
    return y_pred[0]

@app.route('/predict_gender', methods=['POST'])
def predict_gender():
    try:
        data = request.json
        prediction = predict_gender_logic(data)
        gender = 'female' if prediction == 0 else 'male'
        return jsonify({'prediction': gender})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000, debug=False)
