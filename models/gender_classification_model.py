import pandas as pd
import numpy as np
import pickle
import os
import warnings
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
import mlflow
import mlflow.sklearn

warnings.filterwarnings("ignore")

def train():
    print("Training Gender Classification Model...")
    with mlflow.start_run(run_name="Gender_Classification"):
        user_df = pd.read_csv("data/users.csv")
        user_df = user_df[user_df['gender'].isin(['male', 'female'])].reset_index(drop=True)

        le = LabelEncoder()
        user_df['company_encoded'] = le.fit_transform(user_df['company'])

        le_gender = LabelEncoder()
        y_gender = le_gender.fit_transform(user_df['gender'])

        model = SentenceTransformer('flax-sentence-embeddings/all_datasets_v4_MiniLM-L6')
        user_df['name'] = user_df['name'].fillna('').astype(str)
        embeddings = np.array(user_df['name'].apply(lambda text: model.encode(text)).tolist())

        n_components = 23
        mlflow.log_param("pca_n_components", n_components)
        pca = PCA(n_components=n_components, random_state=42)
        text_embeddings_pca = pca.fit_transform(embeddings)

        X_numerical = user_df[['code', 'company_encoded', 'age']].fillna(user_df[['code', 'company_encoded', 'age']].median()).values
        X_gender = np.hstack((text_embeddings_pca, X_numerical))

        gender_scaler = StandardScaler()
        X_gender_scaled = gender_scaler.fit_transform(X_gender)
        
        max_iter = 500
        mlflow.log_param("max_iter", max_iter)
        lr = LogisticRegression(random_state=42, max_iter=max_iter)
        lr.fit(X_gender_scaled, y_gender)

        os.makedirs("models", exist_ok=True)
        with open("models/pca_1.pkl", "wb") as f:
            pickle.dump(pca, f)
        with open("models/scaler_1.pkl", "wb") as f:
            pickle.dump(gender_scaler, f)
        with open("models/tuned_logistic_regression_model_1.pkl", "wb") as f:
            pickle.dump(lr, f)

        mlflow.sklearn.log_model(lr, "gender_classification_lr_model")
        
        print("Gender Classification models saved successfully.")

if __name__ == "__main__":
    train()
