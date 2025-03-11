# Functions for the main script

import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import project_02.Functions.time_series_functions as fun
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

## Time series

# For time series forecasting, we will use the Prophet library
def forecast_crypto(df, crypto_id, periods=30):
    # Filter the data for the specified cryptocurrency
    df_crypto = df[df['crypto_id'] == crypto_id]

    # Prepare the data for Prophet
    df_crypto = df_crypto[['date', 'volatility', 'close']]
    df_crypto.columns = ['ds', 'y']

    # Convert the 'ds' column to datetime
    df_crypto['ds'] = pd.to_datetime(df_crypto['ds'])

    # Aggregate data to daily level (if necessary)
    df_crypto = df_crypto.groupby('ds').mean().reset_index()

    # Initialize the Prophet model
    model = Prophet()

    # Fit the model on the data
    model.fit(df_crypto)

    # Create a dataframe to hold predictions
    future = model.make_future_dataframe(periods=periods)

    # Make predictions
    forecast = model.predict(future)

    # Plot the forecast
    model.plot(forecast)
    plt.title(f'{crypto_id} Close Price Forecast')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.show()

    # Plot the forecast components
    model.plot_components(forecast)
    plt.show()

    # Evaluate the model
    mae = mean_absolute_error(df_crypto['y'], forecast['yhat'][:len(df_crypto)])
    print(f'Mean Absolute Error for {crypto_id}: {mae}')

    return forecast, mae

def plot_real_vs_predicted(df, forecast, crypto_id, year):
    # Filter the data for the specified cryptocurrency
    df_crypto = df[df['crypto_id'] == crypto_id]
    
    # Prepare the data for Prophet
    df_crypto = df_crypto[['date', 'close']]
    df_crypto.columns = ['ds', 'y']
    df_crypto['ds'] = pd.to_datetime(df_crypto['ds'])
    df_crypto = df_crypto.groupby('ds').mean().reset_index()
    
    # Filter the real data for the specified year
    df_real = df_crypto[(df_crypto['ds'] >= f'{year}-01-01') & (df_crypto['ds'] <= f'{year}-12-31')]
    
    # Filter the forecasted data for the specified year
    df_forecast = forecast[(forecast['ds'] >= f'{year}-01-01') & (forecast['ds'] <= f'{year}-12-31')]
    
    # Plot the real vs predicted data
    plt.figure(figsize=(12, 6))
    plt.plot(df_real['ds'], df_real['y'], label='Real Close Price', color='blue')
    plt.plot(df_forecast['ds'], df_forecast['yhat'], label='Predicted Close Price', color='red', linestyle='--')
    plt.fill_between(df_forecast['ds'], df_forecast['yhat_lower'], df_forecast['yhat_upper'], color='pink', alpha=0.3, label='Uncertainty Interval')
    plt.title(f'Real vs Predicted Close Price for {crypto_id} in {year}')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Ensure the real and predicted data have the same length
    df_real = df_real.set_index('ds')
    df_forecast = df_forecast.set_index('ds')
    df_combined = df_real.join(df_forecast[['yhat']], how='inner')
    
    # Calculate MAE and RMSE
    mae = mean_absolute_error(df_combined['y'], df_combined['yhat'])
    rmse = np.sqrt(mean_squared_error(df_combined['y'], df_combined['yhat']))
    
    print(f'Mean Absolute Error (MAE) for {crypto_id} in {year}: {mae}')
    print(f'Root Mean Squared Error (RMSE) for {crypto_id} in {year}: {rmse}')
    
    
    
    
