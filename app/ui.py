import streamlit as st
import pandas as pd
import requests
import pickle

st.set_page_config(page_title="MLOps Ecosystem Dashboard", layout="wide")

st.markdown("""
<style>
.stApp { background-color: #f8f9fa; }
h1 { color: #2c3e50; }
</style>
""", unsafe_allow_html=True)

st.title("Travel Analytics MLOps Dashboard")

tabs = st.tabs(["✈️ Flight Price Predictor", "👥 Gender Classification", "🏨 Hotel Recommender"])

API_URL = "http://api:8000"

with tabs[0]:
    st.header("Flight Price Predictor")
    with st.form("flight_form"):
        col1, col2 = st.columns(2)
        with col1:
            boarding = st.selectbox("From (Boarding City)", ["Florianopolis (SC)", "Sao_Paulo (SP)", "Salvador (BH)", "Brasilia (DF)", "Rio_de_Janeiro (RJ)", "Campo_Grande (MS)", "Aracaju (SE)", "Natal (RN)", "Recife (PE)"])
            destination = st.selectbox("To (Destination City)", ["Florianopolis (SC)", "Sao_Paulo (SP)", "Salvador (BH)", "Brasilia (DF)", "Rio_de_Janeiro (RJ)", "Campo_Grande (MS)", "Aracaju (SE)", "Natal (RN)", "Recife (PE)"])
            flightType = st.selectbox("Flight Type", ["economic", "firstClass", "premium"])
        with col2:
            agency = st.selectbox("Agency", ["Rainbow", "CloudFy", "FlyingDrops"])
            week_no = st.number_input("Week Number", min_value=1, max_value=52, value=12)
            week_day = st.number_input("Week Day (1-7)", min_value=1, max_value=7, value=3)
            day = st.number_input("Day of Month", min_value=1, max_value=31, value=15)
        
        submitted = st.form_submit_button("Predict Price")
        if submitted:
            payload = {
                "from": boarding,
                "destination": destination,
                "flightType": flightType,
                "agency": agency,
                "week_no": week_no,
                "week_day": week_day,
                "day": day
            }
            try:
                # We also provide localhost fallback for local testing without Docker
                res = requests.post(f"{API_URL}/predict_flight", json=payload)
                if res.status_code == 200:
                    st.success(f"Predicted Flight Price: ${res.json().get('prediction')}")
                else:
                    st.error("Error from API")
            except requests.exceptions.ConnectionError:
                try:
                    res = requests.post("http://localhost:8000/predict_flight", json=payload)
                    if res.status_code == 200:
                        st.success(f"Predicted Flight Price: ${res.json().get('prediction')}")
                    else:
                        st.error("Error from API")
                except:
                    st.error("Could not connect to the API.")

with tabs[1]:
    st.header("Gender Classification Model")
    with st.form("gender_form"):
        name = st.text_input("Name", value="Charlotte Johnson")
        usercode = st.number_input("User Code", value=123)
        age = st.number_input("Age", value=30, min_value=18, max_value=100)
        company = st.selectbox("Company", ["Acme Factory", "Wonka Company", "Monsters CYA", "Umbrella LTDA", "4You"])
        
        submitted_g = st.form_submit_button("Predict Gender")
        if submitted_g:
            payload = {"name": name, "code": usercode, "age": age, "company": company}
            try:
                res = requests.post(f"{API_URL}/predict_gender", json=payload)
                if res.status_code == 200:
                    st.success(f"Predicted Gender: {res.json().get('prediction').upper()}")
                else:
                    st.error("Error from API")
            except requests.exceptions.ConnectionError:
                try:
                    res = requests.post("http://localhost:8000/predict_gender", json=payload)
                    if res.status_code == 200:
                        st.success(f"Predicted Gender: {res.json().get('prediction').upper()}")
                    else:
                        st.error("Error from API")
                except:
                    st.error("Could not connect to the API.")

with tabs[2]:
    st.header("Hotel Recommender")
    st.markdown("Recommends hotels utilizing sparse matrix factorization mapping behavioral databases.")
    try:
        hotel_df = pd.read_csv("data/hotels.csv")
        usercode_options = hotel_df['userCode'].unique()
        selected_usercode = st.selectbox('Select User Code', usercode_options)
        
        if st.button("Recommend Hotels"):
            try:
                # Need to load the custom class structure or it will fail unpickling if the class isn't in scope.
                # A quick workaround is to import the model script.
                import sys
                sys.path.append('.')
                from models.recommendation_model import CFRecommender
                
                cf_recommender_model = pickle.load(open('models/cf_recommender_model.pkl', 'rb'))
                recommended_hotels = cf_recommender_model.recommend_items(selected_usercode, verbose=True)
                if recommended_hotels.empty:
                    st.warning("No recommendations found.")
                else:
                    st.dataframe(recommended_hotels)
            except Exception as e:
                st.error(f"Error loading recommendations: {e}")
    except FileNotFoundError:
        st.warning("Hotel dataset not found. Ensure data/hotels.csv exists.")
