from binance.client import Client
import pandas as pd
import ta
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import time
import datetime
import numpy as np
import statistics
from colorama import Fore, Style
import pmdarima as pm

# Function to pause until a specific second of the minute
def pause_until(second=5):
    while True:
        now = datetime.datetime.now()
        if now.second == second:
            break
        time.sleep(0.1)

# Initialize the Binance client
binance_api_key = ''
binance_api_secret = ''
binance_client = Client(binance_api_key, binance_api_secret)

# User input for the trading pair
trading_pair = input("Enter the trading pair: ")

# Function to retrieve historical data
def get_historical_data(pair):
    historical_data = binance_client.get_klines(symbol=pair, interval=Client.KLINE_INTERVAL_1MINUTE, limit=120)
    data_frame = pd.DataFrame(historical_data, columns=['time', 'open', 'high', 'low', 'close', 'vol', 'close_time', 'quote_vol', 'trades', 'taker_buy_base_vol', 'taker_buy_quote_vol', 'ignore'])
    data_frame['time'] = pd.to_datetime(data_frame['time'], unit='ms').astype('int64') // 10**9
    for column in data_frame.columns.drop('time'):
        data_frame[column] = pd.to_numeric(data_frame[column], errors='coerce')
    data_frame.drop(['ignore', 'close_time', 'quote_vol', 'trades', 'taker_buy_base_vol', 'taker_buy_quote_vol'], axis=1, inplace=True)
    return data_frame

# Function to compute technical indicators
def compute_indicators(df):
    df['SMA'] = ta.trend.sma_indicator(df['close'], window=15)
    df['Bollinger_H'] = ta.volatility.bollinger_hband(df['close'], window=15)
    df['Bollinger_M'] = ta.volatility.bollinger_mavg(df['close'], window=15)
    df['Bollinger_L'] = ta.volatility.bollinger_lband(df['close'], window=15)
    df['Force_Index'] = ta.volume.force_index(df['close'], df['vol'], window=15)
    df['CCI'] = ta.trend.cci(df['high'], df['low'], df['close'], window=15)
    distance = ((df['high'] + df['low']) / 2) - ((df['high'].shift(1) + df['low'].shift(1)) / 2)
    box_ratio = (df['vol'] / 100000000) / (df['high'] - df['low'])
    df['EoM'] = distance / (box_ratio + 1e-7)
    return df

# Function to fetch and process new data
def fetch_process_new_data(pair, existing_data):
    new_candles = binance_client.get_klines(symbol=pair, interval=Client.KLINE_INTERVAL_1MINUTE, limit=30)
    new_data_frame = pd.DataFrame(new_candles, columns=['time', 'open', 'high', 'low', 'close', 'vol', 'close_time', 'quote_vol', 'trades', 'taker_buy_base_vol', 'taker_buy_quote_vol', 'ignore'])
    new_data_frame['time'] = pd.to_datetime(new_data_frame['time'], unit='ms').astype('int64') // 10**9
    for column in new_data_frame.columns.drop('time'):
        new_data_frame[column] = pd.to_numeric(new_data_frame[column], errors='coerce')
    new_data_frame.drop(['ignore', 'close_time', 'quote_vol', 'trades', 'taker_buy_base_vol', 'taker_buy_quote_vol'], axis=1, inplace=True)
    updated_data = pd.concat([existing_data, new_data_frame]).drop_duplicates(subset=['time']).reset_index(drop=True)
    updated_data = compute_indicators(updated_data)
    return updated_data.tail(15)

# Function to get the current market price
def get_current_market_price(pair):
    ticker_info = binance_client.get_symbol_ticker(symbol=pair)
    return float("{:.15f}".format(float(ticker_info['price'])))

# Function to check if the difference is significant
def is_significant_diff(future, predicted, actual, threshold=0.0003):
    diff = abs(future - predicted) / actual
    return diff > threshold

# Function to update the model with new data
def update_model(model, new_features, new_targets):
    if len(new_features) >= 15 and len(new_targets) >= 15:
        new_features = new_features.iloc[-15:]
        new_targets = new_targets[-15:]
        data_matrix = xgb.DMatrix(new_features, label=new_targets)
        booster = model.get_booster()
        updated_booster = xgb.train(params=model.get_xgb_params(), dtrain=data_matrix, xgb_model=booster)
        model._Booster = updated_booster
    return model

# Fetch initial data and process it
initial_data = get_historical_data(trading_pair)
processed_data = compute_indicators(initial_data)
processed_data['target'] = (processed_data['close'].shift(-1) > processed_data['close']).astype(int)
processed_data['historical_price'] = processed_data['close']
processed_data.dropna(inplace=True)

# Display the current market price
market_price = get_current_market_price(trading_pair)
print(Style.BRIGHT + f'{trading_pair} ' + Style.RESET_ALL + 'Current Market Price:' + Style.RESET_ALL + Fore.CYAN + f' {market_price}'+ Style.RESET_ALL)

# Prepare data for regression
X_reg = processed_data[['open', 'high', 'low', 'close', 'vol', 'SMA', 'Bollinger_H', 'Bollinger_M', 'Bollinger_L', 'Force_Index', 'CCI', 'EoM']]
y_reg = processed_data['historical_price']

# Split the data for training and testing
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.1, random_state=42)

# Train the regression model
reg_model = xgb.XGBRegressor(enable_categorical=True, learning_rate=0.05, max_depth=8)
reg_model.fit(X_train_reg, y_train_reg)

# Initial prediction using the regression model
processed_data['predicted_future_price'] = reg_model.predict(X_reg)

# Prepare data for classification
X_cls = processed_data[['open', 'high', 'low', 'close', 'vol', 'SMA', 'Bollinger_H', 'Bollinger_M', 'Bollinger_L', 'Force_Index', 'CCI', 'EoM', 'predicted_future_price']]
y_cls = processed_data['target']
X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(X_cls, y_cls, test_size=0.3, shuffle = False, random_state=42)

# Train the classification model
cls_model = xgb.XGBClassifier(enable_categorical=True)
cls_model.fit(X_train_cls, y_train_cls)

# Calculate classification accuracy
cls_predictions = cls_model.predict(X_test_cls)
cls_accuracy = accuracy_score(y_test_cls, cls_predictions)
print(f"Classification Model Accuracy: {cls_accuracy}")

# Calculate RMSE for the regression model
reg_predictions = reg_model.predict(X_test_reg)
reg_rmse = mean_squared_error(y_test_reg, reg_predictions, squared=False)
print(f"Regression Model RMSE: {reg_rmse}")

# Initialize list for storing predicted future prices
future_prices_list = []

def train_arima_model(historical_data, processed_data):
    if not isinstance(historical_data, pd.Series):
        historical_data = pd.Series(historical_data)
    if 'predicted_future_price' in processed_data:
        predictions = pd.Series(processed_data['predicted_future_price'])
    else:
        raise ValueError("The processed_data DataFrame does not contain a 'predicted_future_price' column.")
    # Combining historical data and predictions for ARIMA model training
    combined_data_arima = pd.concat([historical_data, predictions]).reset_index(drop=True)
    model = pm.auto_arima(combined_data_arima, m=60, seasonal=True, trace=True, error_action='ignore', suppress_warnings=True)
    return model

# Assuming 'data' is your DataFrame containing historical prices to be used for ARIMA
historical_prices = processed_data['close']  # or any other column representing historical prices

arima_model = train_arima_model(historical_prices, processed_data)

# To make future predictions with ARIMA
n_periods = 5  # For example, predicting the next 5 time steps
arima_forecast = arima_model.predict(n_periods=n_periods)
print(arima_forecast)

# Main script execution
pause_until(5)

# Main loop for real-time predictions
if reg_rmse < 1:
    update_interval = 29
    iteration_count = 0  

    while True:
        iteration_count += 1
        
        # Fetch and preprocess new data
        new_market_data = fetch_process_new_data(trading_pair, processed_data)
        new_features_reg = new_market_data[['open', 'high', 'low', 'close', 'vol', 'SMA', 'Bollinger_H', 'Bollinger_M', 'Bollinger_L', 'Force_Index', 'CCI', 'EoM']].copy()

        # Predict future price using the regression model
        future_price_pred = reg_model.predict(new_features_reg)
        
        # Extract the first minute prediction
        first_minute_prediction = future_price_pred[0] if len(future_price_pred) > 0 else None

        # Fetch the actual last minute price
        actual_last_price = get_current_market_price(trading_pair)
        
        # Compare the prediction with the actual price
        if first_minute_prediction is not None:
            print(f"First Minute Prediction: " + Fore.GREEN + f"{first_minute_prediction}" + Style.RESET_ALL + ", Actual Last Price: " + Fore.MAGENTA + f"{actual_last_price}" + Style.RESET_ALL)

        # Calculate and print the difference if needed
        difference = abs(first_minute_prediction - actual_last_price)
        print(f"Difference between predicted and actual price: " + Fore.RED + f"{difference}" + Style.RESET_ALL)
        
        # Add the predicted future price to new_features for classification prediction
        new_features_cls = new_features_reg.copy()
        new_features_cls['predicted_future_price'] = future_price_pred
        
        predicted_price_mean = statistics.mean(future_price_pred)
        print(f"Predicted Price (Mean calculation of the 15 min predicted prices): {predicted_price_mean}, Actual Last Price for {trading_pair}: {actual_last_price}")
        
        # Get the actual current market price
        current_market_price = get_current_market_price(trading_pair)

        # Predict buy or sell signal using the classification model
        buy_sell_signal = cls_model.predict(new_features_cls)[0]
        signal_prob = cls_model.predict_proba(new_features_cls)[0][buy_sell_signal]
        
        # Display the buy or sell signal
        if buy_sell_signal == 1:
            signal_text = "BUY LONG"
        elif buy_sell_signal == 0:
            signal_text = "SELL SHORT"
        print(f"{signal_text} signal based on XGB-Classifier prediction - Probability: "  + Fore.GREEN +  f"{signal_prob}" + Style.RESET_ALL)

        # Check if the prediction needs correction
        if abs(future_price_pred[0] - current_market_price) > 0.002:
            processed_data = fetch_process_new_data(trading_pair, processed_data)
            new_features_correction = processed_data[['open', 'high', 'low', 'close', 'vol', 'SMA', 'Bollinger_H', 'Bollinger_M', 'Bollinger_L', 'Force_Index', 'CCI', 'EoM']]
            reg_model = update_model(reg_model, new_features_correction, [current_market_price] * 15)
            print('Model updated due to significant difference between prediction and actual price.')
        else:
            print(Style.DIM + f'No significant difference. Continuing prediction...' + Style.RESET_ALL)

        # Decision making based on predicted price change
        price_change_threshold = 0.010
        predicted_price_change = statistics.mean(future_price_pred) - current_market_price
        print(f"Check for difference > : {predicted_price_change}")
        print(future_price_pred)
        # current_market_price - add logic to buy or sell based on the current market price and the price_change_threshold
        
        if buy_sell_signal == 1 and predicted_price_change > price_change_threshold:
            print(Style.BRIGHT + f'Action: BUY LONG - Confidence: {signal_prob}' + Style.RESET_ALL, 'Predicted Price Change:' + f'{predicted_price_change}')
        elif buy_sell_signal == 0 and predicted_price_change < -price_change_threshold:
            print(Style.BRIGHT + f"Action: SELL SHORT - Confidence: {signal_prob}" + Style.RESET_ALL, "Predicted Price Change:" + f"{predicted_price_change}")
        else:
            print(Style.DIM + f'No clear signal - Waiting for new data' + Style.RESET_ALL)

        # Update and display predicted prices for each minute
        for i in range(15):
            if i < len(future_prices_list):
                future_prices_list[i] = future_price_pred[i]
            else:
                future_prices_list.append(future_price_pred[i])
            print(f"Minute {i+1} Predicted Price: {future_prices_list[i]}")
        
        # Keep only the latest predictions
        max_predictions_to_keep = 90
        future_prices_list = future_prices_list[-max_predictions_to_keep:]

        print(Style.BRIGHT + f"Iteration count: {iteration_count}" + Style.RESET_ALL)
        
        # Define the stop loss and take profit percentages
        stop_loss_percent = 0.5  # 0.5%
        take_profit_percent = 88  # 88%
        
        price_difference = abs(actual_last_price - first_minute_prediction)
        
        print(f"Signal: {buy_sell_signal} Price difference: {price_difference} Last price: {actual_last_price} Predicted price mean: {predicted_price_mean}")
        
        # Check if the first minute prediction is the same as the actual last price
        
        # Add logic to print BUY OR SELL signal based on the buy_sell signal
        if buy_sell_signal == 1 and price_difference <= 0.006 and actual_last_price == predicted_price_mean:
            print("Entering the if condition.")
            stop_loss_value = actual_last_price * (1 - stop_loss_percent)
            take_profit_value = actual_last_price * (1 + take_profit_percent)

            print(Fore.BLUE + f"Stop Loss Value:" + Style.RESET_ALL + Fore.YELLOW + f'{stop_loss_value}' + Style.RESET_ALL)
            print(Fore.BLUE + f"Take Profit Value:" + Style.RESET_ALL + Fore.YELLOW + f'{take_profit_value}' + Style.RESET_ALL)
        
        else:
            print(Style.BRIGHT + f'FUTURE 1 BETA ' + Style.RESET_ALL + Fore.YELLOW + f'Prediction Lottery' + Style.RESET_ALL)
        
        pause_until(5)  # Wait until the next cycle

        # Fetch and prepare current features for prediction
        current_features = fetch_process_new_data(trading_pair, processed_data)
        features_for_prediction = current_features[['open', 'high', 'low', 'close', 'vol', 'SMA', 'Bollinger_H', 'Bollinger_M', 'Bollinger_L', 'Force_Index', 'CCI', 'EoM']]
        
        future_price_pred = reg_model.predict(features_for_prediction)
        
        # Fetch actual current market price
        actual_market_price = get_current_market_price(trading_pair)

        # Regularly update the model
        if iteration_count % update_interval == 0:
            reg_model = update_model(reg_model, features_for_prediction, [actual_market_price]*15)
            print('Reached update interval. Model updated with new data.')
        else:
            print(Style.BRIGHT + 'Continuing prediction...' + Style.RESET_ALL)

        # Update model if significant difference is found
        if is_significant_diff(future_prices_list[-1], future_price_pred[-1], actual_market_price):
            reg_model = update_model(reg_model, features_for_prediction, [actual_market_price]*15)
            print(Style.BRIGHT + f'Model updated due to significant difference.' + Style.RESET_ALL)

else:
    print("Model RMSE is above the threshold, no action taken.")