# Travel Analytics MLOps Ecosystem

An end-to-end MLOps project that combines:
- Flight price prediction
- Gender classification
- Hotel recommendation

The project includes model training scripts, a Flask inference API, a Streamlit UI, and Docker-based deployment support.

## Project Structure

```text
MLOPS-Project-Almabetter/
|- api/
|  |- app.py
|- app/
|  |- ui.py
|- data/
|  |- flights.csv
|  |- users.csv
|  |- hotels.csv
|- models/
|  |- regression_model.py
|  |- gender_classification_model.py
|  |- recommendation_model.py
|  |- *.pkl (generated model artifacts)
|- deployment/
|  |- Dockerfile
|- docker-compose.yml
|- requirements.txt
```

## Features

### 1) Flight Price Prediction
- Model: `RandomForestRegressor` (with feature scaling)
- Training script: `models/regression_model.py`
- API endpoint: `POST /predict_flight`

### 2) Gender Classification
- Model: SentenceTransformer embeddings + PCA + Logistic Regression
- Training script: `models/gender_classification_model.py`
- API endpoint: `POST /predict_gender`

### 3) Hotel Recommendation
- Model: Collaborative filtering using SVD (`CFRecommender`)
- Training script: `models/recommendation_model.py`
- Used directly in Streamlit recommender tab

## API Endpoints

Base URL: `http://localhost:8000`

- `GET /` - Service info and available routes
- `GET /health` - Health status
- `POST /predict_flight` - Predict flight ticket price
- `POST /predict_gender` - Predict gender label

## Run Locally (Windows PowerShell)

### 1. Create environment and install dependencies

```powershell
cd C:\Users\rohan\Desktop\Antigravity_project\MLOPS-Project-Almabetter
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Start Flask API (Terminal 1)

```powershell
.\venv\Scripts\Activate.ps1
python api\app.py
```

API: `http://localhost:8000`

### 3. Start Streamlit UI (Terminal 2)

```powershell
.\venv\Scripts\Activate.ps1
streamlit run app\ui.py --server.port 8501 --server.headless true
```

UI: `http://localhost:8501`

## Train/Rebuild Models

Run these if model artifacts are missing or corrupted:

```powershell
python models\regression_model.py
python models\gender_classification_model.py
python models\recommendation_model.py
```

## Docker Deployment

Run both API and UI with Docker Compose:

```powershell
docker compose up --build
```

Stop containers:

```powershell
docker compose down
```

## Tech Stack

- Python 3.11
- Flask
- Streamlit
- scikit-learn
- SentenceTransformers
- pandas, numpy
- MLflow
- Docker, Docker Compose
