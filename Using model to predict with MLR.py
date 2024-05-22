####Using model to predict

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer
from data_cleaning import scaler, pca
from data_cleaning import to_snake_case
from joblib import load

# Load training model
mlr_model = load("C:/Users/Administrator/OneDrive/Ambiente de Trabalho/Master Classes/2nd Semester/Data Science project/Football project/top5 eur leagues DSP/mlr_model.pkl")

# Load new data
df_new = pd.read_csv("C:/Users/Administrator/OneDrive/Ambiente de Trabalho/Master Classes/2nd Semester/Data Science project/Football project/top5 eur leagues DSP/scraped_data_for_ML_test.csv", encoding='latin-1')  

# Standartizing collumns
df_new.columns = [to_snake_case(col) for col in df_new.columns]

#Defining features with missing values
variables_with_missing = [
    'non_penalty_goals', 'shots_total', 'shot_creating_actions', 'passes_attempted', 
    'progressive_passes', 'progressive_carries', 'successful_take_ons', 
    'touches_att_pen', 'progressive_passes_rec', 'tackles', 'interceptions', 
    'blocks', 'clearances', 'aerials_won'
]

#Preprocessing function
def preprocess_player_data(player_data):
    global variables_with_missing
    # Imputing missing values
    imputer = KNNImputer()
    player_data[variables_with_missing] = imputer.fit_transform(player_data[variables_with_missing])
    # Re-fit scaler and PCA directly on new data
    scaler = StandardScaler()
    pca = PCA(n_components=1)
    # Standardizing and applying PCA
    var_set_standardized = scaler.fit_transform(player_data[['shots_total', 'touches_att_pen']])
    player_data['shots_touches'] = pca.fit_transform(var_set_standardized)
    # Converting to dummy variables
    player_data = pd.get_dummies(player_data, columns=['position', 'league'])

    return player_data, imputer

# Removing Target Variable
df_new = df_new.drop('annual_wage_in_e_u_r', axis=1)

# Preprocessing new data
X_new, _ = preprocess_player_data(df_new)

# Ensure Correct Feature Order and add missing features
original_feature_order = mlr_model.feature_names_in_
for missing_col in mlr_model.feature_names_in_:
    if missing_col not in X_new.columns:
        X_new[missing_col] = 0  # Or fill with another suitable default
        
# Remove Extra Columns:
for extra_col in X_new.columns:
    if extra_col not in mlr_model.feature_names_in_:
        X_new.drop(columns = extra_col, inplace = True)
X_new = X_new[original_feature_order]

# Make prediction and convert it to original scale, then print prediction in EUR for each player
y_new_log_pred = mlr_model.predict(X_new)  
y_new_pred = np.expm1(y_new_log_pred)
print("Predicted Annual Wages:")
for i, wage in enumerate(y_new_pred):
    print(f"Player {i+1}: €{wage:.2f}")

