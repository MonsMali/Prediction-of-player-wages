import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

# Variables from scouting report differ for Goalkeepers and Non-Goalkeepers
# Split the data into two sets
df = pd.read_csv('player_stats_final.csv')
# Identify goalkeepers based on the 'position' column
goalkeeper_positions = ['GK']
# Split the data into goalkeepers and non-goalkeepers
goalkeepers = df[df['position'].isin(goalkeeper_positions)].copy()
players = df[~df['position'].isin(goalkeeper_positions)].copy()

# Remove variables that are no longer relevant for the two groups
goalkeeper_columns_to_drop = [
    'non_penalty_goals', 'shots_total', 'shot_creating_actions',
    'passes_attempted', 'pass_completion', 'progressive_passes',
    'progressive_carries', 'successful_take_ons', 'touches_att_pen',
    'progressive_passes_rec', 'tackles', 'interceptions', 'blocks',
    'clearances', 'aerials_won', 'goals', 'assists', 'penalty_play'
]
players_columns_to_drop = [
    'goals_against', 'save_%', 'save%_penalty_kicks', 'clean_sheet_%',
    'touches', 'launch_%', 'goal_kicks', 'avg_length_goal_kicks', 
    'crosses_stopped_%','def_actions_outside_pen_area', 'avg_distance_def_actions'
]
goalkeepers.drop(columns=goalkeeper_columns_to_drop, inplace=True)
goalkeepers.reset_index(drop=True, inplace=True)
players.drop(columns=players_columns_to_drop, inplace=True)
players.reset_index(drop=True, inplace=True)

# Assign categorical type to 'position' and 'league'
players.dtypes
players['position'] = players['position'].astype('category')
players['league'] = players['league'].astype('category')

# Impute missing variables with KNN
missing_values_count = players.isnull().sum()
missing_values_count
variables_with_missing = ['non_penalty_goals', 'shots_total', 'shot_creating_actions',
                          'passes_attempted', 'progressive_passes', 'progressive_carries',
                          'successful_take_ons', 'touches_att_pen', 'progressive_passes_rec',
                          'tackles', 'interceptions', 'blocks', 'clearances', 'aerials_won']
imputer = KNNImputer()
players[variables_with_missing] = imputer.fit_transform(players[variables_with_missing])

# Check for highly correlated variables for 'players' dataset 
correlation_matrix = players.drop(columns = ['position', 'league']).corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()
# Appy PCA to combine two highly correlated var: 'shots_total' and 'touches_att_pen
scaler = StandardScaler()
var_set = players[['shots_total', 'touches_att_pen']]
var_set_standardized = scaler.fit_transform(var_set)
pca = PCA(n_components=1)
players['shots_touches'] = pca.fit_transform(var_set_standardized)
remove_columns = ['shots_total', 'touches_att_pen']
players.drop(columns=remove_columns, inplace=True)
players.columns

players['annual_wage'].describe()
# One-hot encode the two categorical variables
players_encoded = pd.get_dummies(players, columns=['position', 'league'])

##### MODELS #####
# Continue with modelling using the Non-Goalkeepers('players') dataset, as 'goalkeepers' dataset has only 174 observations 
### Model 1: Multiple Linear Regression
# Make a copy of the dataset
players_reg = players_encoded.copy()

# Split the data into training and testing set 
# Apply log transformation for the depedent variable
y = np.log1p(players_reg['annual_wage'])
X = players_reg.drop(columns=['annual_wage'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model using 10-fold cross validation and evaluate on test data
mlr_model = LinearRegression()
k_fold = KFold(n_splits=10, shuffle=True, random_state=42)
rmse_scores = []
mape_scores = []
for train_index, val_index in k_fold.split(X_train):
    X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
    y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]
    mlr_model.fit(X_train_fold, y_train_fold)
    y_val_pred_log = mlr_model.predict(X_val_fold)
    y_val_pred = np.expm1(y_val_pred_log)
    y_val = np.expm1(y_val_fold)
    rmse_fold = np.sqrt(mean_squared_error(y_val, y_val_pred))
    rmse_scores.append(rmse_fold)
    mape_fold = mean_absolute_percentage_error(y_val, y_val_pred) * 100
    mape_scores.append(mape_fold)
rmse_scores
mape_scores
mean_rmse = np.mean(rmse_scores)
mean_mape = np.mean(mape_scores)
mean_rmse
mean_mape
mlr_model.fit(X_train, y_train)
y_test_pred_log = mlr_model.predict(X_test)
y_test_pred = np.expm1(y_test_pred_log)
test_rmse = np.sqrt(mean_squared_error(np.expm1(y_test), y_test_pred))
test_mape = mean_absolute_percentage_error(np.expm1(y_test), y_test_pred) * 100
test_rmse
# 1783818
test_mape
# 71.50

#### Model 2: Multiple Linear Regression with all PCA components extracted
# Make a new copy of the dataset
players_reg2 = players_encoded.copy()

# Apply log to dependent variable 
y = np.log1p(players_reg2['annual_wage'])
X = players_reg2.drop(columns=['annual_wage'])
# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Apply PCA
pca = PCA(n_components=0.95)  # Retain 95% of the variance
X_pca = pca.fit_transform(X_scaled)
X_pca.shape # 23 principal componenents extracted 
pca.explained_variance_ratio_
cumulative_explained_variance = print(np.cumsum(pca.explained_variance_ratio_))

# Split the data into train and test 
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)
# Define and train the regression model
mlr_model = LinearRegression()
mlr_model.fit(X_train, y_train)
# Predict and evaluate on the test set
y_test_pred_log = mlr_model.predict(X_test)
# Transform predictions and actual values back to the original scale
y_test_pred = np.expm1(y_test_pred_log)
y_test = np.expm1(y_test)
# Calculate RMSE and MAPE on test set
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
test_mape = mean_absolute_percentage_error(y_test, y_test_pred) * 100
test_rmse
# 1799476
test_mape 
# 69.42

### Model 3: Ridge Regression with paremeter tuning for 'aplha'
players_reg3 = players_encoded.copy()

y = np.log1p(players_reg3['annual_wage'])
X = players_reg3.drop(columns=['annual_wage'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.linear_model import RidgeCV, Ridge
# Define a range of alpha values to search
alphas = [0.01, 0.1, 1, 5, 10, 12.5, 15, 20,]
# Initialize RidgeCV with the range of alpha values
ridge_cv_model = RidgeCV(alphas=alphas, cv=10)
# Fit the model to the data
ridge_cv_model.fit(X_train, y_train)
# Get the best alpha value
best_alpha = ridge_cv_model.alpha_
best_alpha
# Initialize Ridge regression with the best alpha value
ridge_model = Ridge(alpha=best_alpha)
# Fit the model to the data
ridge_model.fit(X_train, y_train)
# Predict on the test set
y_test_pred_log = ridge_model.predict(X_test)
y_test_pred = np.expm1(y_test_pred_log)
test_rmse = np.sqrt(mean_squared_error(np.expm1(y_test), y_test_pred))
test_mape = mean_absolute_percentage_error(np.expm1(y_test), y_test_pred) * 100
test_rmse
# 1792377
test_mape
# 71.69

### Model 4: Decision Tree regression
players_tree = players_encoded.copy()

y = np.log1p(players_tree['annual_wage'])
X = players_tree.drop(columns=['annual_wage'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.tree import DecisionTreeRegressor

# Define parameter grid
param_grid = {
    'max_depth': [None, 5, 10, 15, 20],
    'min_samples_split': [2, 5, 10, 15],
    'min_samples_leaf': [1, 2, 4, 8],
    'max_features': [None, 'sqrt', 'log2'],
    'min_impurity_decrease': [0.0, 0.05, 0.1, 0.15],
    'max_leaf_nodes': [None, 5, 10, 20, 50]
}
# Initialize decision tree regressor
dt_regressor = DecisionTreeRegressor(random_state=42)
# Initialize grid search with cross-validation
grid_search = GridSearchCV(dt_regressor, param_grid, cv=5, scoring='neg_mean_squared_error')
# Fit the grid search to the data
grid_search.fit(X_train, y_train)
# Get the best model from the grid search
best_dt_model = grid_search.best_estimator_
best_dt_model
# Predict on the test set
y_test_pred_log = best_dt_model.predict(X_test)
# Convert predictions back to original scale
y_test_pred = np.expm1(y_test_pred_log)
# Evaluate the model
test_rmse = np.sqrt(mean_squared_error(np.expm1(y_test), y_test_pred))
test_mape = mean_absolute_percentage_error(np.expm1(y_test), y_test_pred) * 100
test_rmse
# 2014403
test_mape
# 76.48

### Model 5: Random Forest regression
players_rf = players_encoded.copy()
y = np.log1p(players_rf['annual_wage'])
X = players_rf.drop(columns=['annual_wage'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestRegressor
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}
# Initialize the Random Forest Regressor
rf_model = RandomForestRegressor(random_state=42)
# Perform Grid Search CV to find the best hyperparameters
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, 
                           scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)
# Get the best parameters
best_params = grid_search.best_params_
best_params
# Initialize Random Forest Regressor with the best hyperparameters
best_rf_model = RandomForestRegressor(random_state=42, **best_params)
# Fit the model to the training data
best_rf_model.fit(X_train, y_train)
# Make predictions
y_test_pred_log = best_rf_model.predict(X_test)
# Convert predictions back to original scale
y_test_pred = np.expm1(y_test_pred_log)
y_test_orig = np.expm1(y_test)
# Evaluate the model
test_rmse = np.sqrt(mean_squared_error(y_test_orig, y_test_pred))
test_mape = mean_absolute_percentage_error(y_test_orig, y_test_pred) * 100
test_rmse
# 1896661
test_mape
# 73.92

### Model 6: Gradient Boosting Regression
players_boost = players_encoded.copy()
y = np.log1p(players_boost['annual_wage'])
X = players_boost.drop(columns=['annual_wage'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.ensemble import GradientBoostingRegressor
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.5],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}
# Initialize the Gradient Boosting Regressor
gb_model = GradientBoostingRegressor(random_state=42)
# Perform Grid Search CV to find the best hyperparameters
grid_search = GridSearchCV(estimator=gb_model, param_grid=param_grid, cv=5, 
                           scoring='neg_mean_squared_error', verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)
# Get the best parameters
best_params = grid_search.best_params_
# Initialize Gradient Boosting Regressor with the best hyperparameters
best_gb_model = GradientBoostingRegressor(random_state=42, **best_params)
# Fit the model to the training data
best_gb_model.fit(X_train, y_train)
# Make predictions
y_test_pred_log = best_gb_model.predict(X_test)
# Convert predictions back to original scale
y_test_pred = np.expm1(y_test_pred_log)
y_test_orig = np.expm1(y_test)
# Evaluate the model
test_rmse = np.sqrt(mean_squared_error(y_test_orig, y_test_pred))
test_mape = mean_absolute_percentage_error(y_test_orig, y_test_pred) * 100
test_rmse
# 1772690
test_mape
# 70.51

### Model 6: K-Nearest Neighbors (KNN) Regression
players_knn = players_encoded.copy()
y = np.log1p(players_knn['annual_wage'])
X = players_knn.drop(columns=['annual_wage'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsRegressor

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the KNN model and parameter grid
knn = KNeighborsRegressor()
param_grid = {
    'n_neighbors': [3, 5, 7, 10, 15],
    'weights': ['uniform', 'distance'],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    'leaf_size': [10, 30, 50],
    'p': [1, 2]  # 1 for Manhattan distance, 2 for Euclidean distance
}

# Perform GridSearchCV
grid_search = GridSearchCV(knn, param_grid, cv=10, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)
# Get the best model
best_knn_model = grid_search.best_estimator_
# Predict on the test set
y_test_pred_log = best_knn_model.predict(X_test)
# Convert predictions back to original scale
y_test_pred = np.expm1(y_test_pred_log)
y_test_orig = np.expm1(y_test)
# Evaluate the model
test_rmse = np.sqrt(mean_squared_error(y_test_orig, y_test_pred))
test_mape = mean_absolute_percentage_error(y_test_orig, y_test_pred) * 100
test_rmse
# 2000190
test_mape
# 82.41

### Model 7: Support Vector Regression 
players_svr = players_encoded.copy()
y = np.log1p(players_svr['annual_wage'])
X = players_svr.drop(columns=['annual_wage'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.svm import SVR
# Define the SVR model
svr_model = SVR()
# Define the parameter grid to search
param_grid = {
    'kernel': ['linear', 'rbf', 'poly'],
    'C': [0.1, 1, 10],
    'gamma': ['scale'],
    'epsilon': [0.1, 0.2, 0.5],
    'degree': [2, 3, 4]
}
# Initialize GridSearchCV with parallel computation
grid_search = GridSearchCV(estimator=svr_model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5, n_jobs=-1, verbose=2)
# Fit the grid search to the data
grid_search.fit(X_train, y_train)
# Get the best parameters and model
best_params = grid_search.best_params_
best_params
best_svr_model = grid_search.best_estimator_
# Predict on the test set
y_test_pred_log = best_svr_model.predict(X_test)
# Convert predictions back to original scale
y_test_pred = np.expm1(y_test_pred_log)
y_test_orig = np.expm1(y_test)
# Evaluate the model
test_rmse = np.sqrt(mean_squared_error(y_test_orig, y_test_pred))
test_mape = mean_absolute_percentage_error(y_test_orig, y_test_pred) * 100
test_rmse
# 1748001
test_mape
# 72.55

# Extra Model
### Model 7: Neural Network
import keras_tuner as kt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import random
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)
def build_model(hp):
    model = Sequential()
    model.add(Dense(
        units=hp.Int('units', min_value=32, max_value=512, step=32),
        activation='relu',
        input_dim=X_train.shape[1]
    ))
    for i in range(hp.Int('num_layers', 1, 3)):
        model.add(Dense(
            units=hp.Int('units_' + str(i), min_value=32, max_value=512, step=32),
            activation='relu'
        ))
    model.add(Dense(1, activation='linear'))
    model.compile(
        optimizer=Adam(
            learning_rate=hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
        ),
        loss='mean_squared_error'
    )
    return model
tuner = kt.RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=5,
    executions_per_trial=3,
    directory='my_dir',
    project_name='helloworld'
)
players_nn = players_encoded.copy()
y = np.log1p(players_nn['annual_wage'])
X = players_nn.drop(columns=['annual_wage'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
tuner.search(X_train, y_train, epochs=50, validation_split=0.2, verbose=1)
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
best_model = tuner.hypermodel.build(best_hps)
best_model.fit(X_train, y_train, epochs=100, validation_split=0.2, verbose=1)
# Predict on the test set
y_test_pred_log = best_model.predict(X_test).flatten()
# Convert predictions back to the original scale
y_test_pred = np.expm1(y_test_pred_log)
y_test_orig = np.expm1(y_test)
# Evaluate the model
test_rmse = np.sqrt(mean_squared_error(y_test_orig, y_test_pred))
test_mape = mean_absolute_percentage_error(y_test_orig, y_test_pred) * 100
test_rmse
# 2034867
test_mape
# 67.26