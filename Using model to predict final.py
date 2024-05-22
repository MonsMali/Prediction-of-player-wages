####Using model to predict

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer
from joblib import load
from data_cleaning import to_snake_case
from data_cleaning import scaler, pca
import pickle

# Load all three models
gb_model, gb_features = load("gradient_boosting_model.joblib")
mlr_model, scaler, pca, pca_features = load("ml_pca_model.joblib")
svr_model, svr_features = load("svr_model.joblib")

# Manually define original_feature_order
original_feature_order = ['age', 'club', 'league_Bundesliga', 'league_La Liga', 'league_Ligue 1', 'league_Premier League', 'league_Serie A', 'position_Attacker', 'position_Defender', 'position_Goalkeeper', 'position_Midfielder', 'minutes_played_overall', 'minutes_played_league', 'minutes_played_cup', 'minutes_played_intl', 'appearances_overall', 'appearances_league', 'appearances_cup', 'appearances_intl', 'goals_overall', 'assists_overall', 'goals_conceded_overall', 'clean_sheets_overall', 'yellow_cards_overall', 'second_yellow_cards_overall', 'red_cards_overall', 'goals_per_90_overall', 'assists_per_90_overall', 'goals_conceded_per_90_overall', 'clean_sheets_%_overall', 'yellow_cards_per_90_overall', 'red_cards_per_90_overall', 'non_penalty_goals', 'npxg_per_shot_overall', 'goals_minus_xG_overall', 'xAG_overall', 'xAG_per_90_overall', 'assists_minus_xAG_overall', 'shots_total_per90_overall', 'shots_on_target_%_overall', 'shot_creating_actions_per_90_overall', 'passes_completed_overall', 'passes_attempted', 'pass_completion_%', 'progressive_passes', 'progressive_carries', 'progressive_passes_rec', 'successful_take_ons', 'take_on_success_%', 'touches_att_pen', 'touches_def_pen', 'touches_def_3rd', 'touches_mid_3rd', 'touches_att_3rd', 'dribbles_completed_overall', 'dribbles_vs_live_opponent_%_overall', 'dribbles_completed_per_90_overall', 'successful_defensive_actions_per_90_overall', 'tackles', 'interceptions', 'blocks', 'clearances', 'aerials_won', 'aerials_won_%', 'shots_touches']

# Load new data
df_new = pd.read_csv("C:/Users/Administrator/OneDrive/Ambiente de Trabalho/Master Classes/2nd Semester/Data Science project/Football project/top5 eur leagues DSP/scraped_data_for_ML_test.csv", encoding='latin-1')  

# Standardize columns
df_new.columns = [to_snake_case(col) for col in df_new.columns]

# Defining features with missing values
variables_with_missing = [
    'non_penalty_goals', 'shots_total', 'shot_creating_actions', 'passes_attempted', 
    'progressive_passes', 'progressive_carries', 'successful_take_ons', 
    'touches_att_pen', 'progressive_passes_rec', 'tackles', 'interceptions', 
    'blocks', 'clearances', 'aerials_won'
]

# Preprocessing function
def preprocess_player_data(player_data, train_features):
    global variables_with_missing
    # Imputing missing values
    imputer = KNNImputer()
    player_data[variables_with_missing] = imputer.fit_transform(player_data[variables_with_missing])
    # Reindex columns to align with training data order
    player_data = player_data.reindex(columns=train_features, fill_value=0)  
    return player_data, imputer

# Removing Target Variable
df_new = df_new.drop('annual_wage_in_e_u_r', axis=1)

# Preprocessing new data for the 3 models
X_new_gb, _ = preprocess_player_data(df_new, gb_features)
X_new_svr, _ = preprocess_player_data(df_new, svr_features)
X_new_scaled = scaler.transform(X_new[pca_features])  # Standardize using saved features
X_new_pca = pca.transform(X_new_scaled)  # Apply PCA

# Make predictions using each model
gb_pred = np.expm1(gb_model.predict(X_new_gb))
mlr_pca_pred = np.expm1(mlr_model.predict(X_new_pca))
svr_pred = np.expm1(svr_model.predict(X_new_svr))

# Print predictions for each model
print("Gradient Boosting Predictions:")
for i, wage in enumerate(gb_pred):
    print(f"Player {i+1}: €{wage:.2f}")

print("\nMLR with PCA Predictions:")
for i, wage in enumerate(mlr_pca_pred):
    print(f"Player {i+1}: €{wage:.2f}")

print("\nSVR Predictions:")
for i, wage in enumerate(svr_pred):
    print(f"Player {i+1}: €{wage:.2f}")

results_df = pd.DataFrame({
    'Player': df_new['player_name'], 
    'Gradient Boosting Prediction': gb_pred,
    'MLR with PCA Prediction': mlr_pca_pred,
    'SVR Prediction': svr_pred
})

# Format predictions to two decimal places (without the € symbol)
for col in ['Gradient Boosting Prediction', 'MLR with PCA Prediction', 'SVR Prediction']:
    results_df[col] = results_df[col].apply(lambda x: f'{x:.2f}')

# Save the results to a CSV file
results_df.to_csv("player_salary_predictions.csv", index=False)