import pandas as pd
import numpy as np
import pickle
import os
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from scipy.sparse.linalg import svds
import mlflow

warnings.filterwarnings("ignore")

class CFRecommender:
    MODEL_NAME = 'Collaborative Filtering'
    def __init__(self, cf_predictions_df , items_df):
        self.cf_predictions_df = cf_predictions_df
        self.items_df = items_df

    def get_model_name(self):
        return self.MODEL_NAME

    def recommend_items(self, user_id, items_to_ignore=[], topn=5, verbose=False):
        if user_id not in self.cf_predictions_df.columns:
            raise KeyError(f"User '{user_id}' not found in prediction data.")
        sorted_user_predictions = self.cf_predictions_df[user_id].sort_values(ascending=False).reset_index().rename(columns={user_id: 'recStrength'})
        recommendations_df = sorted_user_predictions[~sorted_user_predictions['name_encoded'].isin(items_to_ignore)].sort_values('recStrength', ascending=False).head(topn)

        if verbose:
            if self.items_df is None:
                raise Exception('"items_df" is required in verbose mode')
            recommendations_df = recommendations_df.merge(self.items_df, how='left',
                                                          left_on='name_encoded',
                                                          right_on='name_encoded')[['name_encoded','name','recStrength']]
            recommendations_df=pd.DataFrame(recommendations_df.groupby('name').max('recStrength').sort_values('recStrength', ascending=False))
        return recommendations_df

def train():
    print("Training Hotel Recommendation Model...")
    with mlflow.start_run(run_name="Hotel_Recommendation_CF"):
        df=pd.read_csv('data/hotels.csv')
        hotel_df = df.copy()

        users_interactions_count_df = hotel_df.groupby(['userCode','name']).size().groupby('userCode').size()
        users_with_enough_interactions_df = users_interactions_count_df[users_interactions_count_df >= 2].reset_index()[['userCode']]
        
        interactions_from_selected_users_df = hotel_df.merge(users_with_enough_interactions_df,
                       how = 'right',
                       left_on = 'userCode',
                       right_on = 'userCode')

        label_encoder = LabelEncoder()
        interactions_from_selected_users_df['name_encoded'] = label_encoder.fit_transform(interactions_from_selected_users_df['name'])

        interactions_full_df = interactions_from_selected_users_df.groupby(['name_encoded','userCode'])['price'].sum().reset_index()

        interactions_train_df, interactions_test_df = train_test_split(interactions_full_df,
                                                                       stratify=interactions_full_df['userCode'],
                                           test_size=0.25,
                                           random_state=42)

        items_users_pivot_matrix_df = interactions_train_df.pivot(index='userCode',
                                                                  columns='name_encoded',
                                                                  values='price').fillna(0)
        items_users_pivot_matrix = items_users_pivot_matrix_df.values
        user_ids = list(items_users_pivot_matrix_df.index)

        NUMBER_OF_FACTORS_MF = 8
        mlflow.log_param("NUMBER_OF_FACTORS_MF", NUMBER_OF_FACTORS_MF)
        U, sigma, Vt = svds(items_users_pivot_matrix, k = NUMBER_OF_FACTORS_MF)
        sigma = np.diag(sigma)
        all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt)

        cf_preds_df = pd.DataFrame(all_user_predicted_ratings, columns = items_users_pivot_matrix_df.columns,index=user_ids).transpose()
        cf_recommender_model = CFRecommender(cf_preds_df, interactions_from_selected_users_df)

        os.makedirs("models", exist_ok=True)
        with open('models/cf_recommender_model.pkl', 'wb') as f:
            pickle.dump(cf_recommender_model, f)
            
        mlflow.log_artifact('models/cf_recommender_model.pkl', "hotel_recommendation_cf_model")
        
        print("Hotel Recommendation CF model saved successfully.")

if __name__ == "__main__":
    train()
