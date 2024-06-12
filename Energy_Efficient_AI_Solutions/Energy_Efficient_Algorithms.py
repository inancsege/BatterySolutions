import numpy as np
import pandas as pd
import perun
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import L2
from concurrent.futures import ProcessPoolExecutor, as_completed
from joblib import parallel_backend
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error


def lstm(data):

    # Define your features and target variable
    target_column = 'Utot (V)'
    features = data.drop(columns=[target_column, 'Time (h)'])
    time_feature = data['Time (h)'].values.reshape(-1, 1)  # Keep Time (h) for later use, if needed

    # Prepare the data
    X = np.hstack([features, time_feature]).reshape(-1, 1, features.shape[1] + 1)
    Y = data[target_column].values

    # Scaling the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.reshape(X.shape[0], -1)).reshape(X.shape)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=50, shuffle=False)

    # Define the LSTM model
    model = Sequential([
        LSTM(20, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2]),
             kernel_regularizer=L2(0.01), recurrent_regularizer=L2(0.01), bias_regularizer=L2(0.01)),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='mse')

    # Train the model with validation split
    history = model.fit(X_train, y_train, epochs=100, batch_size=16, verbose=1, validation_split=0.2)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    # Calculate and print the Mean Absolute Percentage Error
    mape = mean_absolute_percentage_error(y_test, y_pred)
    print(f'Accuracy in lstm: {100 - mape}')
    return y_test, y_pred


def xg_boost(data):

    # Define features and target
    target_column = 'Utot (V)'
    feature_columns = [col for col in data.columns if col != target_column]

    # Simple Feature Engineering: Scale numeric features for example
    scaler = StandardScaler()
    data[feature_columns] = scaler.fit_transform(data[feature_columns])

    # Prepare data matrices
    X = data[feature_columns].values
    Y = data[target_column].values

    # Hyperparameter grid
    param_grid = {
        'max_depth': [5],
        'min_child_weight': [5],
        'learning_rate': [0.2],
        'n_estimators': [200]
    }

    # Initialize the XGBoost regressor
    model = xgb.XGBRegressor(objective='reg:squarederror', seed=42)

    # Setup GridSearchCV
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error',
                               verbose=2)

    # Fit GridSearchCV
    grid_search.fit(X, Y)

    # Best model
    best_model = grid_search.best_estimator_

    # Prediction with best model
    y_pred = best_model.predict(X)
    mape = mean_absolute_percentage_error(Y, y_pred)
    print(f'Accuracy in xgboost: {100 - mape}')
    return Y, y_pred


def random_forest(data):

    # Assuming 'data' is your DataFrame after loading your CSV
    # Using all other features except 'Utot (V)' to predict 'Utot (V)'
    target_column = 'Utot (V)'
    feature_columns = [col for col in data.columns if col != target_column]

    X = data[feature_columns]
    Y = data[target_column]

    # Scaling the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=42, shuffle=False)

    # Initialize and train the RandomForest model
    model = RandomForestRegressor(random_state=42)

    model.fit(X_train, y_train)

    # Predicting on the test set

    y_pred = model.predict(X_test)
    mape = mean_absolute_percentage_error(Y, y_pred)
    print(f'Accuracy in random forest: {100 - mape}')
    return y_test, y_pred


def decision_tree(data):

    # Define your features and target variable
    # Make sure 'Utot (V)' is the correct target column name
    target_column = 'Utot (V)'
    feature_columns = [col for col in data.columns if col != target_column]

    X = data[feature_columns].values
    Y = data[target_column].values

    # Splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.5, random_state=42, shuffle=False)

    # Initialize the Decision Tree Regressor with default parameters
    dt_model = DecisionTreeRegressor(random_state=42)

    # Train the model
    dt_model.fit(X_train, y_train)

    # Predicting on the test set
    y_pred = dt_model.predict(X_test)
    mape = mean_absolute_percentage_error(Y, y_pred)
    print(f'Accuracy in decision tree: {100 - mape}')
    return y_test, y_pred


def main():
    data = pd.read_csv('../Dataset/concatenated_FC1_Ageing.csv')
    lstm(data)
    random_forest(data)
    decision_tree(data)
    xg_boost(data)


def run_with_cores(cores):
    with parallel_backend('loky', n_jobs=cores):
        with ProcessPoolExecutor(max_workers=cores) as executor:
            executor.submit(main)


if __name__ == "__main__":
    for cores in [1, 2, 4]:
        print(f"Running with {cores} cores")
        run_with_cores(cores)