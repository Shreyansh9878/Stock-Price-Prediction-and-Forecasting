import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from quantile_forest import RandomForestQuantileRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import date, timedelta
import base64

best_model = {'ADANIPORTS.NS': {'Model': 'MLPRegressor',
  'Model_params': {'activation': 'relu',
   'alpha': 0.01,
   'early_stopping': True,
   'hidden_layer_sizes': (100,),
   'learning_rate_init': 0.0005,
   'max_iter': 5000,
   'n_iter_no_change': 20,
   'random_state': 42,
   'solver': 'adam',
   'validation_fraction': 0.2}},
 'ASIANPAINT.NS': {'Model': 'MLPRegressor',
  'Model_params': {'activation': 'logistic',
   'alpha': 0.01,
   'early_stopping': True,
   'hidden_layer_sizes': (50, 50),
   'learning_rate_init': 0.0005,
   'max_iter': 5000,
   'n_iter_no_change': 20,
   'random_state': 42,
   'solver': 'adam',
   'validation_fraction': 0.2}},
 'AXISBANK.NS': {'Model': 'DecisionTreeRegressor',
  'Model_params': {'criterion': 'mae',
   'max_depth': 10,
   'min_samples_split': 10}},
 'BAJAJ-AUTO.NS': {'Model': 'MLPRegressor',
  'Model_params': {'activation': 'tanh',
   'alpha': 0.01,
   'early_stopping': True,
   'hidden_layer_sizes': (150, 100, 50),
   'learning_rate_init': 0.0005,
   'max_iter': 5000,
   'n_iter_no_change': 20,
   'random_state': 42,
   'solver': 'adam',
   'validation_fraction': 0.2}},
 'BAJAJFINSV.NS': {'Model': 'MLPRegressor',
  'Model_params': {'activation': 'tanh',
   'alpha': 0.01,
   'early_stopping': True,
   'hidden_layer_sizes': (100, 100),
   'learning_rate_init': 0.0005,
   'max_iter': 5000,
   'n_iter_no_change': 20,
   'random_state': 42,
   'solver': 'adam',
   'validation_fraction': 0.2}},
 'BAJFINANCE.NS': {'Model': 'RandomForestRegressor',
  'Model_params': {'max_depth': 20,
   'min_samples_split': 5,
   'n_estimators': 50,
   'random_state': 42}},
 'BHARTIARTL.NS': {'Model': 'MLPRegressor',
  'Model_params': {'activation': 'tanh',
   'alpha': 0.01,
   'early_stopping': True,
   'hidden_layer_sizes': (150, 100, 50),
   'learning_rate_init': 0.0005,
   'max_iter': 5000,
   'n_iter_no_change': 20,
   'random_state': 42,
   'solver': 'adam',
   'validation_fraction': 0.2}},
 'BPCL.NS': {'Model': 'MLPRegressor',
  'Model_params': {'activation': 'logistic',
   'alpha': 0.001,
   'early_stopping': True,
   'hidden_layer_sizes': (100,),
   'learning_rate_init': 0.0005,
   'max_iter': 5000,
   'n_iter_no_change': 20,
   'random_state': 42,
   'solver': 'adam',
   'validation_fraction': 0.2}},
 'BRITANNIA.NS': {'Model': 'MLPRegressor',
  'Model_params': {'activation': 'tanh',
   'alpha': 0.01,
   'early_stopping': True,
   'hidden_layer_sizes': (100, 100),
   'learning_rate_init': 0.0005,
   'max_iter': 5000,
   'n_iter_no_change': 20,
   'random_state': 42,
   'solver': 'adam',
   'validation_fraction': 0.2}},
 'CIPLA.NS': {'Model': 'MLPRegressor',
  'Model_params': {'activation': 'tanh',
   'alpha': 0.01,
   'early_stopping': True,
   'hidden_layer_sizes': (100, 100),
   'learning_rate_init': 0.0005,
   'max_iter': 5000,
   'n_iter_no_change': 20,
   'random_state': 42,
   'solver': 'adam',
   'validation_fraction': 0.2}},
 'COALINDIA.NS': {'Model': 'MLPRegressor',
  'Model_params': {'activation': 'tanh',
   'alpha': 0.0001,
   'early_stopping': True,
   'hidden_layer_sizes': (100, 100),
   'learning_rate_init': 0.0005,
   'max_iter': 5000,
   'n_iter_no_change': 20,
   'random_state': 42,
   'solver': 'adam',
   'validation_fraction': 0.2}},
 'DRREDDY.NS': {'Model': 'MLPRegressor',
  'Model_params': {'activation': 'tanh',
   'alpha': 0.0001,
   'early_stopping': True,
   'hidden_layer_sizes': (100, 100),
   'learning_rate_init': 0.0005,
   'max_iter': 5000,
   'n_iter_no_change': 20,
   'random_state': 42,
   'solver': 'adam',
   'validation_fraction': 0.2}},
 'EICHERMOT.NS': {'Model': 'MLPRegressor',
  'Model_params': {'activation': 'tanh',
   'alpha': 0.01,
   'early_stopping': True,
   'hidden_layer_sizes': (150, 100, 50),
   'learning_rate_init': 0.0005,
   'max_iter': 5000,
   'n_iter_no_change': 20,
   'random_state': 42,
   'solver': 'adam',
   'validation_fraction': 0.2}},
 'GAIL.NS': {'Model': 'MLPRegressor',
  'Model_params': {'activation': 'tanh',
   'alpha': 0.01,
   'early_stopping': True,
   'hidden_layer_sizes': (100, 100),
   'learning_rate_init': 0.0005,
   'max_iter': 5000,
   'n_iter_no_change': 20,
   'random_state': 42,
   'solver': 'adam',
   'validation_fraction': 0.2}},
 'GRASIM.NS': {'Model': 'MLPRegressor',
  'Model_params': {'activation': 'tanh',
   'alpha': 0.01,
   'early_stopping': True,
   'hidden_layer_sizes': (100,),
   'learning_rate_init': 0.0005,
   'max_iter': 5000,
   'n_iter_no_change': 20,
   'random_state': 42,
   'solver': 'adam',
   'validation_fraction': 0.2}},
 'HCLTECH.NS': {'Model': 'DecisionTreeRegressor',
  'Model_params': {'criterion': 'mae',
   'max_depth': 30,
   'min_samples_split': 5}},
 'HDFC.NS': {'Model': 'MLPRegressor',
  'Model_params': {'activation': 'relu',
   'alpha': 0.0001,
   'early_stopping': True,
   'hidden_layer_sizes': (100, 100),
   'learning_rate_init': 0.0005,
   'max_iter': 5000,
   'n_iter_no_change': 20,
   'random_state': 42,
   'solver': 'adam',
   'validation_fraction': 0.2}},
 'HDFCBANK.NS': {'Model': 'RandomForestRegressor',
  'Model_params': {'max_depth': None,
   'min_samples_split': 2,
   'n_estimators': 100,
   'random_state': 42}},
 'HEROMOTOCO.NS': {'Model': 'SVR',
  'Model_params': {'C': 10.0,
   'epsilon': 0.01,
   'gamma': 'scale',
   'kernel': 'linear'}},
 'HINDALCO.NS': {'Model': 'MLPRegressor',
  'Model_params': {'activation': 'tanh',
   'alpha': 0.001,
   'early_stopping': True,
   'hidden_layer_sizes': (100, 100),
   'learning_rate_init': 0.0005,
   'max_iter': 5000,
   'n_iter_no_change': 20,
   'random_state': 42,
   'solver': 'adam',
   'validation_fraction': 0.2}},
 'HINDUNILVR.NS': {'Model': 'MLPRegressor',
  'Model_params': {'activation': 'tanh',
   'alpha': 0.001,
   'early_stopping': True,
   'hidden_layer_sizes': (100,),
   'learning_rate_init': 0.0005,
   'max_iter': 5000,
   'n_iter_no_change': 20,
   'random_state': 42,
   'solver': 'adam',
   'validation_fraction': 0.2}},
 'ICICIBANK.NS': {'Model': 'MLPRegressor',
  'Model_params': {'activation': 'tanh',
   'alpha': 0.01,
   'early_stopping': True,
   'hidden_layer_sizes': (100,),
   'learning_rate_init': 0.0005,
   'max_iter': 5000,
   'n_iter_no_change': 20,
   'random_state': 42,
   'solver': 'adam',
   'validation_fraction': 0.2}},
 'INDUSINDBK.NS': {'Model': 'MLPRegressor',
  'Model_params': {'activation': 'tanh',
   'alpha': 0.001,
   'early_stopping': True,
   'hidden_layer_sizes': (100, 100),
   'learning_rate_init': 0.0005,
   'max_iter': 5000,
   'n_iter_no_change': 20,
   'random_state': 42,
   'solver': 'adam',
   'validation_fraction': 0.2}},
 'INFY.NS': {'Model': 'MLPRegressor',
  'Model_params': {'activation': 'relu',
   'alpha': 0.0001,
   'early_stopping': True,
   'hidden_layer_sizes': (50, 50),
   'learning_rate_init': 0.0005,
   'max_iter': 5000,
   'n_iter_no_change': 20,
   'random_state': 42,
   'solver': 'adam',
   'validation_fraction': 0.2}},
 'IOC.NS': {'Model': 'MLPRegressor',
  'Model_params': {'activation': 'logistic',
   'alpha': 0.01,
   'early_stopping': True,
   'hidden_layer_sizes': (100,),
   'learning_rate_init': 0.0005,
   'max_iter': 5000,
   'n_iter_no_change': 20,
   'random_state': 42,
   'solver': 'adam',
   'validation_fraction': 0.2}},
 'ITC.NS': {'Model': 'MLPRegressor',
  'Model_params': {'activation': 'tanh',
   'alpha': 0.0001,
   'early_stopping': True,
   'hidden_layer_sizes': (150, 100, 50),
   'learning_rate_init': 0.0005,
   'max_iter': 5000,
   'n_iter_no_change': 20,
   'random_state': 42,
   'solver': 'adam',
   'validation_fraction': 0.2}},
 'JSWSTEEL.NS': {'Model': 'DecisionTreeRegressor',
  'Model_params': {'criterion': 'friedman_mse',
   'max_depth': 10,
   'min_samples_split': 2}},
 'KOTAKBANK.NS': {'Model': 'MLPRegressor',
  'Model_params': {'activation': 'tanh',
   'alpha': 0.0001,
   'early_stopping': True,
   'hidden_layer_sizes': (100,),
   'learning_rate_init': 0.0005,
   'max_iter': 5000,
   'n_iter_no_change': 20,
   'random_state': 42,
   'solver': 'adam',
   'validation_fraction': 0.2}},
 'LT.NS': {'Model': 'MLPRegressor',
  'Model_params': {'activation': 'tanh',
   'alpha': 0.01,
   'early_stopping': True,
   'hidden_layer_sizes': (150, 100, 50),
   'learning_rate_init': 0.0005,
   'max_iter': 5000,
   'n_iter_no_change': 20,
   'random_state': 42,
   'solver': 'adam',
   'validation_fraction': 0.2}},
 'M&M.NS': {'Model': 'MLPRegressor',
  'Model_params': {'activation': 'tanh',
   'alpha': 0.01,
   'early_stopping': True,
   'hidden_layer_sizes': (100, 100),
   'learning_rate_init': 0.0005,
   'max_iter': 5000,
   'n_iter_no_change': 20,
   'random_state': 42,
   'solver': 'adam',
   'validation_fraction': 0.2}},
 'MARUTI.NS': {'Model': 'DecisionTreeRegressor',
  'Model_params': {'criterion': 'mae',
   'max_depth': 10,
   'min_samples_split': 2}},
 'NESTLEIND.NS': {'Model': 'MLPRegressor',
  'Model_params': {'activation': 'tanh',
   'alpha': 0.01,
   'early_stopping': True,
   'hidden_layer_sizes': (100,),
   'learning_rate_init': 0.0005,
   'max_iter': 5000,
   'n_iter_no_change': 20,
   'random_state': 42,
   'solver': 'adam',
   'validation_fraction': 0.2}},
 'NTPC.NS': {'Model': 'MLPRegressor',
  'Model_params': {'activation': 'tanh',
   'alpha': 0.01,
   'early_stopping': True,
   'hidden_layer_sizes': (100, 100),
   'learning_rate_init': 0.0005,
   'max_iter': 5000,
   'n_iter_no_change': 20,
   'random_state': 42,
   'solver': 'adam',
   'validation_fraction': 0.2}},
 'ONGC.NS': {'Model': 'MLPRegressor',
  'Model_params': {'activation': 'logistic',
   'alpha': 0.01,
   'early_stopping': True,
   'hidden_layer_sizes': (100,),
   'learning_rate_init': 0.0005,
   'max_iter': 5000,
   'n_iter_no_change': 20,
   'random_state': 42,
   'solver': 'adam',
   'validation_fraction': 0.2}},
 'POWERGRID.NS': {'Model': 'MLPRegressor',
  'Model_params': {'activation': 'tanh',
   'alpha': 0.001,
   'early_stopping': True,
   'hidden_layer_sizes': (100, 100),
   'learning_rate_init': 0.0005,
   'max_iter': 5000,
   'n_iter_no_change': 20,
   'random_state': 42,
   'solver': 'adam',
   'validation_fraction': 0.2}},
 'RELIANCE.NS': {'Model': 'MLPRegressor',
  'Model_params': {'activation': 'tanh',
   'alpha': 0.01,
   'early_stopping': True,
   'hidden_layer_sizes': (100, 100),
   'learning_rate_init': 0.0005,
   'max_iter': 5000,
   'n_iter_no_change': 20,
   'random_state': 42,
   'solver': 'adam',
   'validation_fraction': 0.2}},
 'SBIN.NS': {'Model': 'RandomForestRegressor',
  'Model_params': {'max_depth': 10,
   'min_samples_split': 2,
   'n_estimators': 50,
   'random_state': 42}},
 'SHREECEM.NS': {'Model': 'MLPRegressor',
  'Model_params': {'activation': 'tanh',
   'alpha': 0.01,
   'early_stopping': True,
   'hidden_layer_sizes': (100, 100),
   'learning_rate_init': 0.0005,
   'max_iter': 5000,
   'n_iter_no_change': 20,
   'random_state': 42,
   'solver': 'adam',
   'validation_fraction': 0.2}},
 'SUNPHARMA.NS': {'Model': 'MLPRegressor',
  'Model_params': {'activation': 'tanh',
   'alpha': 0.01,
   'early_stopping': True,
   'hidden_layer_sizes': (100, 100),
   'learning_rate_init': 0.0005,
   'max_iter': 5000,
   'n_iter_no_change': 20,
   'random_state': 42,
   'solver': 'adam',
   'validation_fraction': 0.2}},
 'TATAMOTORS.NS': {'Model': 'MLPRegressor',
  'Model_params': {'activation': 'tanh',
   'alpha': 0.0001,
   'early_stopping': True,
   'hidden_layer_sizes': (100,),
   'learning_rate_init': 0.0005,
   'max_iter': 5000,
   'n_iter_no_change': 20,
   'random_state': 42,
   'solver': 'adam',
   'validation_fraction': 0.2}},
 'TATASTEEL.NS': {'Model': 'MLPRegressor',
  'Model_params': {'activation': 'tanh',
   'alpha': 0.001,
   'early_stopping': True,
   'hidden_layer_sizes': (100, 100),
   'learning_rate_init': 0.0005,
   'max_iter': 5000,
   'n_iter_no_change': 20,
   'random_state': 42,
   'solver': 'adam',
   'validation_fraction': 0.2}},
 'TCS.NS': {'Model': 'MLPRegressor',
  'Model_params': {'activation': 'tanh',
   'alpha': 0.01,
   'early_stopping': True,
   'hidden_layer_sizes': (100, 100),
   'learning_rate_init': 0.0005,
   'max_iter': 5000,
   'n_iter_no_change': 20,
   'random_state': 42,
   'solver': 'adam',
   'validation_fraction': 0.2}},
 'TECHM.NS': {'Model': 'DecisionTreeRegressor',
  'Model_params': {'criterion': 'mse',
   'max_depth': 30,
   'min_samples_split': 5}},
 'TITAN.NS': {'Model': 'MLPRegressor',
  'Model_params': {'activation': 'tanh',
   'alpha': 0.0001,
   'early_stopping': True,
   'hidden_layer_sizes': (100, 100),
   'learning_rate_init': 0.0005,
   'max_iter': 5000,
   'n_iter_no_change': 20,
   'random_state': 42,
   'solver': 'adam',
   'validation_fraction': 0.2}},
 'ULTRACEMCO.NS': {'Model': 'MLPRegressor',
  'Model_params': {'activation': 'tanh',
   'alpha': 0.01,
   'early_stopping': True,
   'hidden_layer_sizes': (100,),
   'learning_rate_init': 0.0005,
   'max_iter': 5000,
   'n_iter_no_change': 20,
   'random_state': 42,
   'solver': 'adam',
   'validation_fraction': 0.2}},
 'UPL.NS': {'Model': 'RandomForestRegressor',
  'Model_params': {'max_depth': 10,
   'min_samples_split': 5,
   'n_estimators': 100,
   'random_state': 42}},
 'VEDL.NS': {'Model': 'MLPRegressor',
  'Model_params': {'activation': 'tanh',
   'alpha': 0.001,
   'early_stopping': True,
   'hidden_layer_sizes': (100,),
   'learning_rate_init': 0.0005,
   'max_iter': 5000,
   'n_iter_no_change': 20,
   'random_state': 42,
   'solver': 'adam',
   'validation_fraction': 0.2}},
 'WIPRO.NS': {'Model': 'MLPRegressor',
  'Model_params': {'activation': 'tanh',
   'alpha': 0.0001,
   'early_stopping': True,
   'hidden_layer_sizes': (100,),
   'learning_rate_init': 0.0005,
   'max_iter': 5000,
   'n_iter_no_change': 20,
   'random_state': 42,
   'solver': 'adam',
   'validation_fraction': 0.2}},
 'ZEEL.NS': {'Model': 'RandomForestRegressor',
  'Model_params': {'max_depth': 10,
   'min_samples_split': 10,
   'n_estimators': 200,
   'random_state': 42}}}

def get_image_base64(file_path):
    with open(file_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()
        
up_arrow = get_image_base64("uparrow.png")
down_arrow = get_image_base64("downarrow.png")

def get_top_indian_stock_symbols():
    data = pd.read_csv("stock_metadata.csv")
    top_indian_symbols = dict(zip(data['Company Name'], data['Symbol']))
    
    # Use yfinance library to get tickers
    tickers = yf.Tickers(list(top_indian_symbols.values()))

    # Extract symbols and info directly from the dictionary
    symbols_info = {}
    for ticker in top_indian_symbols.values():
        try:
            ticker += ".NS"
            info = yf.Ticker(ticker).info
            long_name = info.get('longName', 'N/A')  # Use 'N/A' if 'longName' is not present
            symbols_info[ticker] = long_name
        except Exception as e:
            st.warning(f'Error fetching info for {ticker}: {str(e)}')
            
    myKeys = list(symbols_info.keys())
    myKeys.sort()
    symbols_info = {i: symbols_info[i] for i in myKeys}
    return symbols_info

def get_stock_data(stock_symbol, start_date, end_date):
    try:
        stock_data = yf.download(stock_symbol, start=start_date, end=end_date)
        return stock_data
    except Exception as e:
        st.error(f'Error fetching stock data for {stock_symbol}: {str(e)}')
        return None
    
# Function to preprocess data and train the model
def train_model(dataset, ticker):
    last_date = dataset.index[-1]
    features = ['Close_lag_1', 'Close_lag_2', 'Close_lag_3', 'Close_lag_4', 'Close_lag_5',
                'Close_lag_6', 'Close_lag_7', 'Close_lag_8', 'Close_lag_9', 'Close_lag_10',
                'MA_20', 'MA_50', 'MA_100', 'Day', 'Month']

    # Split features and target
    X = dataset[features]
    y = dataset["Close"].values

    # Scale features and target
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X = scaler_X.fit_transform(X)
    y = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()  # Flatten target for ML models

    # Split into training and test sets
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    params = best_model[ticker]["Model_params"]

    if best_model[ticker]["Model"] == "SVR":
        model = SVR(**params)
    elif best_model[ticker]["Model"] == "KNeighborsRegressor":
        model = KNeighborsRegressor(**params)
    elif best_model[ticker]["Model"] == "MLPRegressor":
        model = MLPRegressor(learning_rate_init=0.01 ,random_state=42, max_iter=500)
    elif best_model[ticker]["Model"] == "LinearRegression":
        model = LinearRegression()
    elif best_model[ticker]["Model"] == "DecisionTreeRegressor":
        model = DecisionTreeRegressor(**params)
    elif best_model[ticker]["Model"] == "RandomForestRegressor":
        model = RandomForestQuantileRegressor(**params)
        
    model.fit(X_train, y_train)

    # Make predictions
    if isinstance(model, RandomForestQuantileRegressor):
        nn_predictions = model.predict(X_test, quantiles=0.5)
    else:
        nn_predictions = model.predict(X_test)

    # Inverse transform predictions and actuals to original scale
    nn_predictions = scaler_y.inverse_transform(nn_predictions.reshape(-1, 1))
    y_test_inverse_nn = scaler_y.inverse_transform(y_test.reshape(-1, 1))

    # Evaluation
    mae = mean_absolute_error(y_test_inverse_nn, nn_predictions)
    mse = np.sqrt(mean_squared_error(y_test_inverse_nn, nn_predictions))
    
    st.write(f'Mean Absolute Error: {mae:.2f}')
    st.write(f'Root Mean Squared Error: {mse:.2f}')

    if best_model[ticker]["Model"] == "SVR":
        new_model = SVR(**params)
    elif best_model[ticker]["Model"] == "KNeighborsRegressor":
        new_model = KNeighborsRegressor(**params)
    elif best_model[ticker]["Model"] == "MLPRegressor":
        new_model = MLPRegressor(learning_rate_init=0.01 ,random_state=42, max_iter=500)
    elif best_model[ticker]["Model"] == "LinearRegression":
        new_model = LinearRegression()
    elif best_model[ticker]["Model"] == "DecisionTreeRegressor":
        new_model = DecisionTreeRegressor(**params)
    elif best_model[ticker]["Model"] == "RandomForestRegressor":
        new_model = RandomForestQuantileRegressor(**params)

    
    new_model.fit(X,y)

    # Find next market open day
    tomorrow_date = last_date + timedelta(days=1)
    while tomorrow_date.weekday() >= 5:  # Saturday = 5, Sunday = 6
        tomorrow_date += timedelta(days=1)

    # Prepare the input for prediction
    last_10_days = dataset["Close"].tail(10).values
    # Create the feature array for the prediction
    # Assuming you have calculated the necessary features for the last day
    last_day_features = {
        'Close_lag_1': last_10_days[-1],
        'Close_lag_2': last_10_days[-2],
        'Close_lag_3': last_10_days[-3],
        'Close_lag_4': last_10_days[-4],
        'Close_lag_5': last_10_days[-5],
        'Close_lag_6': last_10_days[-6],
        'Close_lag_7': last_10_days[-7],
        'Close_lag_8': last_10_days[-8],
        'Close_lag_9': last_10_days[-9],
        'Close_lag_10': last_10_days[-10],
        'MA_20 ': dataset['Close'].rolling(window=20).mean().iloc[-1],
        'MA_50': dataset['Close'].rolling(window=50).mean().iloc[-1],
        'MA_100': dataset['Close'].rolling(window=100).mean().iloc[-1],
        'Day': tomorrow_date.day,
        'Month': tomorrow_date.month
    }
    x = pd.DataFrame([last_day_features])
    x = scaler_X.transform(x)

    if isinstance(model, RandomForestQuantileRegressor):
        y_pred = new_model.predict(x, quantiles=0.5)
    else:
        y_pred = new_model.predict(x)

    y_pred = scaler_y.inverse_transform(y_pred.reshape(-1, 1))

    # Determine if the predicted price is higher or lower than the last actual price
    flg = 1 if y_pred >= y[-1] else 0
 
    return model, tomorrow_date, y_pred, nn_predictions, flg


def plot_data(data, y_pred, symbol):
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot stock data and predictions
    st.subheader('Stock Price')
    ax.plot(data.index[-len(y_pred):], data["Close"][-len(y_pred):], color='blue', label = "Actual Price", linestyle='-', linewidth=2, markersize=5)
    ax.plot(data.index[-len(y_pred):], y_pred, color='red', label = "Predicted Price", linestyle='-', linewidth=2, markersize=5)
    ax.legend()
    ax.set_title(f'{symbol} Closing Prices')
    ax.set_xlabel('Date')
    ax.set_ylabel('Closing Price (INR)')
    st.pyplot(fig)

def feature_engineering(data):
    lag_days=10
    for lag in range(1,lag_days+1):
        data[f'Close_lag_{lag}'] = data['Close'].shift(lag)

    # Adding rolling window features: Moving averages over different windows (e.g., 5-day, 10-day, 20-day moving average)
    data['MA_20'] = data['Close'].transform(lambda x: x.rolling(window=5).mean())
    data['MA_50'] = data['Close'].transform(lambda x: x.rolling(window=10).mean())
    data['MA_100'] = data['Close'].transform(lambda x: x.rolling(window=20).mean())

    data.dropna(inplace=True)

    data['Day'] = data.index.day
    data['Month'] = data.index.month


def StreamGUI():
    # Streamlit UI
    st.title('Nifty-50 India Stock Prediction')
    st.sidebar.header('User Input')
    
    # Fetch top 50 Indian stock symbols and tickers
    symbols = get_top_indian_stock_symbols()

    # Selectbox for stock symbols and tickers
    selected_stock = st.sidebar.selectbox('Select Stock:', list(symbols.keys()), index=0)  # Set default index to 0

    if selected_stock is not None:
        symbol = selected_stock

        start_date = pd.to_datetime('2012-01-01')
        start_date_pvd = st.sidebar.date_input('Start Date:', pd.to_datetime('2012-01-01'))
        end_date = pd.to_datetime(date.today())

        # Fetch historical data
        try:
            data = get_stock_data(symbol, start_date, end_date)
        except Exception as e:
            st.error(f'Error fetching stock data. Please check the stock symbol. Error details: {str(e)}')

        if 'data' in locals():
            data.index = pd.to_datetime(data.index)
            feature_engineering(data)
            # Train the model and make predictions
            model, tomorrow_date, y_pred, y_test_pred, up = train_model(data, symbol)
            y_pred = y_pred[0]

            st.subheader(f'Stock Data for {selected_stock}')
            st.write(data[::-1])

            plot_data(data[data.index >= pd.to_datetime(start_date_pvd)], y_test_pred, symbol)

            # Display the predicted closing price with an inline arrow image
            arrow = up_arrow if up == 1 else down_arrow
            st.markdown(f"""
                <p style='font-size:24px;'>
                Predicted Closing Price on {tomorrow_date.date()}: INR {y_pred[0]:.2f}
                <img src="data:image/png;base64,{arrow}" style="width:24px; height:24px;" />
                </p>
            """, unsafe_allow_html=True)

    else:
        st.warning('Please select a stock from the list.')


if __name__=='__main__':
    StreamGUI()