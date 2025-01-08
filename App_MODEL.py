import pandas as pd
import numpy as np
from numpy.fft import fft
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from darts import TimeSeries    # Import the TimeSeries class from the Darts library for handling time series data
from darts.models import TFTModel  # Temporal Fusion Transformer model for time series forecasting
from darts.utils.likelihood_models import QuantileRegression  # For quantile regression in probabilistic forecasting
from darts.dataprocessing.transformers import Scaler  # For scaling and normalizing time series data
from darts.utils.timeseries_generation import datetime_attribute_timeseries  # For creating time series based on datetime attributes
import optuna
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import plotly.graph_objs as go
import plotly.express as px
import streamlit as st


def load_data(df):
    loaded_data = df
    return loaded_data

# Define preprocessing functions
def create_lag_features(data, lag_steps=3):
    for i in range(1, lag_steps + 1):
        data[f'lag_{i}'] = data['AMOUNT'].shift(i)
    return data

def create_rolling_mean(data, window_size=3):
    data['rolling_mean'] = data['AMOUNT'].rolling(window=window_size).mean()
    return data

def apply_fourier_transform(data):
    values = data['AMOUNT'].values
    fourier_transform = fft(values)
    data['fourier_transform'] = np.abs(fourier_transform)
    return data

def pred_xgboost_model(df):
    data = df.copy()
    data['DATE'] = pd.to_datetime(data['DATE'])
    data.set_index('DATE', inplace=True)
    data = create_lag_features(data)
    data = create_rolling_mean(data)
    data = apply_fourier_transform(data)
    data.fillna(0)
    
    train_size = int(len(data) * 0.8)
    train_data, test_data = data[:train_size], data[train_size:]

    X_train = train_data[['AMOUNT', 'lag_1', 'lag_2', 'lag_3', 'rolling_mean', 'fourier_transform']]
    X_test = test_data[['AMOUNT', 'lag_1', 'lag_2', 'lag_3', 'rolling_mean', 'fourier_transform']]
    y_train = train_data['AMOUNT']
    y_test = test_data['AMOUNT']

    param_grid = {
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 0.9, 1.0]
    }

    grid_search = GridSearchCV(XGBRegressor(), param_grid, cv=3)
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_

    xgb_model = XGBRegressor(**best_params)
    xgb_model.fit(X_train, y_train)
    predictions_xgb = xgb_model.predict(X_test)

    return X_test ,predictions_xgb

def pred_random_forest_model(df):
    data = df.copy()
    data['DATE'] = pd.to_datetime(data['DATE'])
    data.set_index('DATE', inplace=True)
    # data = create_lag_features(data)
    # data = create_rolling_mean(data)
    # data = apply_fourier_transform(data)
    # data.fillna(0)
    train_size = int(len(data) * 0.8)
    train_data, test_data = data[:train_size], data[train_size:]

    X_train = train_data[['AMOUNT']]
    X_test = test_data[['AMOUNT']]
    y_train = train_data['AMOUNT']
    y_test = test_data['AMOUNT']
    

    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    grid_search = GridSearchCV(RandomForestRegressor(), param_grid, cv=3)
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_

    rf_model = RandomForestRegressor(**best_params)
    rf_model.fit(X_train, y_train)
    predictions_rf = rf_model.predict(X_test)

    return X_test ,predictions_rf

def pred_tft_model(data):
    data['DATE'] = pd.to_datetime(data['DATE'])   
    data.set_index('DATE', inplace=True)    
    data.drop(columns=['APP_NAME','MARKET','ACCOUNT_ID','CHANNEL_ID','MPG_ID','CURRENCY_ID','month_index'], inplace=True)    

    series = TimeSeries.from_dataframe(data, value_cols="AMOUNT")    
    train_size = int(len(series) * 0.8)   
    train, val = series[:train_size], series[train_size:]    

    quantiles = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]   

    transformer = Scaler()    # Initialize and fit the Scaler
    train_transformed = transformer.fit_transform(train)    # Transform the training data
    val_transformed = transformer.fit_transform(val)    # Transform the validation data
    series_transformed = transformer.fit_transform(series)    # Transform the entire series


    covariates = datetime_attribute_timeseries(series, attribute="year", one_hot=False)   # Create covariates based on datetime attributes
    covariates = covariates.stack(datetime_attribute_timeseries(series, attribute="month", one_hot=False))  # Create covariates based on datetime attributes
    covariates = covariates.stack(TimeSeries.from_times_and_values(times=series.time_index, values=np.arange(len(series)), columns=["linear_increase"] ))  # Create covariates based on linear increase
    covariates = covariates.astype(np.float32)  # Convert the covariates to float32 data type


    scaler_covs = Scaler()    # Initialize and fit the Scaler
    train_size = int(len(data) * 0.8)    # Calculate the training set size
    cov_train, cov_val = covariates[:train_size], covariates[train_size:]    # Split the data into training and validation sets
    scaler_covs.fit(cov_train)    # Fit the Scaler on the training data
    covariates_transformed = scaler_covs.transform(covariates)    # Transform the covariates

    my_model = TFTModel(
        input_chunk_length=20,
        output_chunk_length=6,
        hidden_size=64,
        lstm_layers=2,
        num_attention_heads=4,
        dropout=0.1,
        batch_size=16,
        n_epochs=10,
        add_relative_index=False,
        likelihood=QuantileRegression(
            quantiles=quantiles
        ),
        random_state=42,
    )

    my_model.fit(train_transformed, future_covariates=covariates_transformed, verbose=True)

    prediction = my_model.predict(len(val), num_samples=1)
    actual_prediction = transformer.inverse_transform(prediction)

    actual = val.values().flatten()
    predictions_tft = actual_prediction.values().flatten()

    return actual, predictions_tft



def plot_actual_vs_predicted(y_test, predictions_xgb, predictions_rf, predictions_tft):
    plt.style.use('dark_background')  # Set plot style to dark background

    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plotting lines without grid
    ax.plot(y_test.index, y_test.values, label='Actual', marker='o', color='white')
    ax.plot(y_test.index, predictions_rf, label='Random Forest Predicted', marker='s', color='blue')
    ax.plot(y_test.index, predictions_xgb, label='XGBoost Predicted', marker='s', color='red')
    ax.plot(y_test.index, predictions_tft, label='Temporal Fusion Transformer Predicted', marker='s', color='green')

    # Customize plot labels and legend
    ax.set_xlabel('Date')
    ax.set_ylabel('Amount')
    ax.set_title('Actual vs Predicted Values')
    ax.legend()
    
    # Remove grid
    ax.grid(False)
    
    # Ensure x-axis labels are readable
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()

    return fig



 #--------------------------------------------------------------------------------------------------------------
def forecast_xgboost_values(df):
    data = df.copy()
    data['DATE'] = pd.to_datetime(data['DATE'])
    data.set_index('DATE', inplace=True)
    data = create_lag_features(data)
    data = create_rolling_mean(data)
    data = apply_fourier_transform(data)
    data.fillna(0)
    X_test ,predictions_xgb = pred_xgboost_model(df)

    selected_columns = ['AMOUNT', 'lag_1', 'lag_2', 'lag_3', 'rolling_mean', 'fourier_transform']
    X = data[selected_columns]
    y = data['AMOUNT']

    param_grid = {
       'learning_rate': [0.01, 0.1, 0.2],
       'max_depth': [3, 5, 7],
       'subsample': [0.8, 0.9, 1.0]
    }

    grid_search = GridSearchCV(XGBRegressor(), param_grid, cv=3)
    grid_search.fit(X, y)
    best_params = grid_search.best_params_


    xgb_model_entire = XGBRegressor(**best_params)
    xgb_model_entire.fit(X, y)

    y = y.values
    last_seq = y[-6:]
    new_predictions = []

    for _ in range(24):
        xgb_next_pred = xgb_model_entire.predict(last_seq.reshape(1, -1))
        new_predictions.append(xgb_next_pred )
        last_seq = np.concatenate((last_seq[1:], xgb_next_pred), axis=0)
    xgb_Forecast = np.array(new_predictions).squeeze().T
    return xgb_Forecast

def forecast_random_forest_values(df):
    data = df.copy()
    data['DATE'] = pd.to_datetime(data['DATE'])
    data.set_index('DATE', inplace=True)
    data = create_lag_features(data)
    data = create_rolling_mean(data)
    data = apply_fourier_transform(data)
    data.dropna(inplace=True)
    X_test ,predictions_rf = pred_random_forest_model(df)

    selected_columns = ['AMOUNT', 'lag_1', 'lag_2', 'lag_3', 'rolling_mean', 'fourier_transform']
    X = data[selected_columns]
    y = data['AMOUNT']
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    grid_search = GridSearchCV(RandomForestRegressor(), param_grid, cv=3)
    grid_search.fit(X, y)
    best_params = grid_search.best_params_

    # Train RandomForest model with best parameters
    rf_model_entire = RandomForestRegressor(**best_params)
    rf_model_entire.fit(X, y)

    y = y.values
    last_seq = y[-6:]
    new_predictions = []

    for _ in range(24):
        rf_next_pred = rf_model_entire.predict(last_seq.reshape(1, -1))
        new_predictions.append(rf_next_pred )
        last_seq = np.concatenate((last_seq[1:], rf_next_pred), axis=0)
    rf_Forecast = np.array(new_predictions).squeeze().T
    return rf_Forecast

def forecast_tft_values(df):
      data = df.copy()
      data['DATE'] = pd.to_datetime(data['DATE'])
      data.set_index('DATE', inplace=True)
      data.drop(columns=['APP_NAME','MARKET','ACCOUNT_ID','CHANNEL_ID','MPG_ID','CURRENCY_ID','month_index'], inplace=True)

      y_test ,predictions_tft = pred_tft_model(df)
      train_size = int(len(data) * 0.8)
      train_set, val_set = data[:train_size], data[train_size:]  

      transformer = Scaler()  # Initialize and fit the Scaler
      series = TimeSeries.from_dataframe(data, value_cols="AMOUNT")  # Create a TimeSeries object from the DataFrame
      series_transformed = transformer.fit_transform(series)  # Transform the entire series
      quantiles = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]

      # Set the lookback window and forecast horizon
      look_back = 6
      forecast_horizon = 24
      # Create future time index
      future_dates = [series.end_time() + pd.DateOffset(months=i+1) for i in range(forecast_horizon)]  # Generate a list of future dates for the next 24 months
      future_series = TimeSeries.from_times_and_values(pd.DatetimeIndex(future_dates), np.zeros(forecast_horizon))  # Create a TimeSeries object with the future dates and zeros

      # Extend covariates to cover the forecast horizon
      covariates = datetime_attribute_timeseries(series, attribute="year", one_hot=False)   # Create covariates based on datetime attributes
      covariates = covariates.stack(datetime_attribute_timeseries(series, attribute="month", one_hot=False))  # Create covariates based on datetime attributes
      covariates = covariates.stack(TimeSeries.from_times_and_values(times=series.time_index, values=np.arange(len(series)), columns=["linear_increase"]))  # Create covariates based on linear increase
      future_covariates = datetime_attribute_timeseries(future_series, attribute="year", one_hot=False)  # Create future covariates based on datetime attributes
      future_covariates = future_covariates.stack(datetime_attribute_timeseries(future_series, attribute="month", one_hot=False))  # Create future covariates based on datetime attributes
      future_covariates = future_covariates.stack(TimeSeries.from_times_and_values(times=future_series.time_index, values=np.arange(len(future_series)), columns=["linear_increase"]))  # Create future covariates based on linear increase

      covariates = covariates.append(future_covariates).astype(np.float32)  # Append the future covariates to the existing covariates

      scaler_covs = Scaler()  # Initialize and fit the Scaler
      scaler_covs.fit(covariates)  # Fit the Scaler on the covariates
      covariates_transformed = scaler_covs.fit_transform(covariates)  # Transform the covariates

      # Initialize the TFTModel with the specified parameters
      my_model = TFTModel(
            input_chunk_length=20,   # Set the input chunk length to 20
            output_chunk_length=6,  # Set the output chunk length to 6
            hidden_size=64,  # Set the hidden size to 64
            lstm_layers=2,  # Set the number of LSTM layers to 2
            num_attention_heads=4,  # Set the number of attention heads to 4
            dropout=0.1,  # Set the dropout rate to 0.1
            batch_size=16,  # Set the batch size to 16
            n_epochs=10,  # Set the number of epochs to 10
            add_relative_index=False,  # Set add_relative_index to False
            likelihood=QuantileRegression(quantiles=quantiles),  # QuantileRegression is set per default
            random_state=42,  # Set the random state to 42
      )

      my_model.fit(series_transformed, future_covariates=covariates_transformed, verbose=True)  # Fit the model on the training data

      new_prob_forecast = my_model.predict(forecast_horizon, num_samples=1, future_covariates=covariates_transformed)  # Make predictions on the validation set
      new_actual_forecast = transformer.inverse_transform(new_prob_forecast)  # Inverse transform the predictions to get the actual values
      new_actual_forecast = new_actual_forecast.values().flatten()
      return new_actual_forecast


def forecast_xgb(df):
          filtered_grouped = df.groupby(['MARKET', 'ACCOUNT_ID', 'CHANNEL_ID', 'MPG_ID'])
          all_results_df = pd.DataFrame()
          for name, group in filtered_grouped:
            data = df.copy()
            data['DATE'] = pd.to_datetime(data['DATE'])
            data.set_index('DATE', inplace=True)
            X_test ,predictions_xgb = pred_xgboost_model(df)
            xgb_Forecast = forecast_xgboost_values(df)
            future_index = [max(data.index) + pd.DateOffset(months=i+1) for i in range(24)]
            forecast_df = pd.DataFrame({'DATE': future_index, 'PREDICTED_AMOUNT': xgb_Forecast ,'ACTUAL_AMOUNT':[0] * len(xgb_Forecast)})
            predict_df = pd.DataFrame({'DATE': X_test.index, 'PREDICTED_AMOUNT': predictions_xgb ,'ACTUAL_AMOUNT':X_test['AMOUNT'] })

            result_df = pd.concat([predict_df,forecast_df, ], ignore_index=True)

            result_df['TYPE'] = ['Predicted'] * len(predict_df) +['Forecasted'] * len(forecast_df)
            result_df['APP_NAME'] = 'LM_EU_WB'
            result_df['MARKET'] = name[0]
            result_df['ACCOUNT_ID'] = name[1]
            result_df['CHANNEL_ID'] = name[2]
            result_df['MPG_ID'] = name[3]
            result_df['CURRENCY_ID'] = 181
            result_df['MODEL'] = 'XGBOOST'


            xgb_results = result_df[['MARKET', 'ACCOUNT_ID', 'CHANNEL_ID', 'MPG_ID', 'DATE','ACTUAL_AMOUNT','TYPE','MODEL','PREDICTED_AMOUNT']]
            all_results_df = pd.concat([all_results_df, result_df], ignore_index=True)
            all_results_df.to_csv('xgb_Results.csv', index=False)
          return data, X_test, predictions_xgb, xgb_Forecast,all_results_df

def forecast_rf(df):
          filtered_grouped = df.groupby(['MARKET', 'ACCOUNT_ID', 'CHANNEL_ID', 'MPG_ID'])
          all_results_df = pd.DataFrame(columns=['MARKET', 'ACCOUNT_ID', 'CHANNEL_ID', 'MPG_ID', 'DATE','ACTUAL_AMOUNT','TYPE','MODEL','PREDICTED_AMOUNT'])
          for name, group in filtered_grouped:
            data = df.copy()
            data['DATE'] = pd.to_datetime(data['DATE'])
            data.set_index('DATE', inplace=True)
            X_test ,predictions_rf = pred_random_forest_model(df)
            rf_Forecast = forecast_random_forest_values(df)
            future_index = [max(data.index) + pd.DateOffset(months=i+1) for i in range(24)]
            forecast_df = pd.DataFrame({'DATE': future_index, 'PREDICTED_AMOUNT': rf_Forecast ,'ACTUAL_AMOUNT':[0] * len(rf_Forecast)})
            predict_df = pd.DataFrame({'DATE': X_test.index, 'PREDICTED_AMOUNT': predictions_rf ,'ACTUAL_AMOUNT':X_test['AMOUNT'] })

            result_df = pd.concat([predict_df,forecast_df, ], ignore_index=True)

            result_df['TYPE'] = ['Predicted'] * len(predict_df) +['Forecasted'] * len(forecast_df)
            result_df['APP_NAME'] = 'LM_EU_WB'
            result_df['MARKET'] = name[0]
            result_df['ACCOUNT_ID'] = name[1]
            result_df['CHANNEL_ID'] = name[2]
            result_df['MPG_ID'] = name[3]
            result_df['CURRENCY_ID'] = 181
            result_df['MODEL'] = 'Random Forest Regressor'


            rf_results = result_df[['MARKET', 'ACCOUNT_ID', 'CHANNEL_ID', 'MPG_ID', 'DATE','ACTUAL_AMOUNT','TYPE','MODEL','PREDICTED_AMOUNT']]
            all_results_df = pd.concat([all_results_df, result_df], ignore_index=True)
            all_results_df.to_csv('rf_Results.csv', index=False)
          return data ,X_test ,predictions_rf ,X_test ,predictions_rf ,rf_Forecast ,all_results_df


def forecast_tft(df):
          df2 = df.copy()
          filtered_grouped = df.groupby(['MARKET', 'ACCOUNT_ID', 'CHANNEL_ID', 'MPG_ID'])
          all_results_df = pd.DataFrame(columns=['MARKET', 'ACCOUNT_ID', 'CHANNEL_ID', 'MPG_ID', 'DATE','ACTUAL_AMOUNT','TYPE','MODEL','PREDICTED_AMOUNT'])
          for name, group in filtered_grouped:
            data = df.copy()
            data['DATE'] = pd.to_datetime(data['DATE'])
            data.set_index('DATE', inplace=True)
            y_test ,predictions_tft = pred_tft_model(df)
            train_size = int(len(data) * 0.8)
            train_set, val_set = data['AMOUNT'][:train_size], data[train_size:]
            X_test = val_set 
            tft_Forecast = forecast_tft_values(df2)
            future_index = [max(data.index) + pd.DateOffset(months=i+1) for i in range(24)]
            forecast_df = pd.DataFrame({'DATE': future_index, 'PREDICTED_AMOUNT': tft_Forecast ,'ACTUAL_AMOUNT':[0] * len(tft_Forecast)})
            predict_df = pd.DataFrame({'DATE': X_test.index, 'PREDICTED_AMOUNT': predictions_tft ,'ACTUAL_AMOUNT':X_test['AMOUNT'] })

            result_df = pd.concat([predict_df,forecast_df, ], ignore_index=True)

            result_df['TYPE'] = ['Predicted'] * len(predict_df) +['Forecasted'] * len(forecast_df)
            result_df['APP_NAME'] = 'LM_EU_WB'
            result_df['MARKET'] = name[0]
            result_df['ACCOUNT_ID'] = name[1]
            result_df['CHANNEL_ID'] = name[2]
            result_df['MPG_ID'] = name[3]
            result_df['CURRENCY_ID'] = 181
            result_df['MODEL'] = 'Temporal Fusion Transformer'
            rf_results = result_df[['MARKET', 'ACCOUNT_ID', 'CHANNEL_ID', 'MPG_ID', 'DATE','ACTUAL_AMOUNT','TYPE','MODEL','PREDICTED_AMOUNT']]
            all_results_df = pd.concat([all_results_df, result_df], ignore_index=True)
            all_results_df.to_csv('tft_Results.csv', index=False)
          return data ,y_test ,predictions_tft ,tft_Forecast ,all_results_df


def plot_Forecasted(data, y_test, predictions_xgb, predictions_rf, predictions_tft, xgb_Forecast, rf_Forecast, tft_Forecast):
    fig = go.Figure()
    # Plot actual values
    fig.add_trace(go.Scatter(x=data[-24:].index, y=data['AMOUNT'][-24:], mode='lines+markers', name='Actual (2 Years)', marker=dict(color='silver')))
    # Plot predicted values for test set
    fig.add_trace(go.Scatter(x=data.index[-len(y_test):], y=predictions_rf, mode='lines+markers', name='Random Forest', marker=dict(color='blue')))
    fig.add_trace(go.Scatter(x=data.index[-len(y_test):], y=predictions_xgb, mode='lines+markers', name='XGBoost', marker=dict(color='red')))
    fig.add_trace(go.Scatter(x=data.index[-len(y_test):], y=predictions_tft, mode='lines+markers', name='Temporal Fusion Transformer', marker=dict(color='green')))

    # Generate future indices for forecasting
    future_index = [data.index[-1] + pd.DateOffset(months=i+1) for i in range(12)]
    # Plot forecasted values
    fig.add_trace(go.Scatter(x=future_index, y=rf_Forecast[:12], mode='lines+markers', line=dict(dash='dash', color='blue')))
    fig.add_trace(go.Scatter(x=future_index, y=xgb_Forecast[:12], mode='lines+markers', line=dict(dash='dash', color='red')))
    fig.add_trace(go.Scatter(x=future_index, y=tft_Forecast[:12], mode='lines+markers', line=dict(dash='dash', color='green')))
    # Set plot labels and title
    fig.update_layout(
        title='Actual vs Predicted and Forecasted Values',
        xaxis_title='Date',
        yaxis_title='Amount',
        plot_bgcolor='black',
        paper_bgcolor='black',
        font=dict(color='white')
    )
    # Adjust plot layout
    fig.update_xaxes(tickangle=45)
    fig.update_yaxes(showgrid=False)
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.show()
    return fig



def plot_Results(df):
    # Filter out rows where ACTUAL_AMOUNT is 0 (since these are forecasted)
    df_filtered = df[df['ACTUAL_AMOUNT'] != 0]

    # Calculate MAE for each model
    mae_scores = df_filtered.groupby('MODEL').apply(lambda x: mean_absolute_error(x['ACTUAL_AMOUNT'], x['PREDICTED_AMOUNT']))
    best_model = mae_scores.idxmin()

    # Create Plotly figure
    fig = go.Figure()

    # Add actual amounts
    fig.add_trace(go.Scatter(x=df['DATE'], y=df['ACTUAL_AMOUNT'], mode='lines+markers', name='Actual Amount', line=dict(color='black')))

    # Add predicted and forecasted amounts for each model
    for model in df['MODEL'].unique():
        model_data = df[df['MODEL'] == model]
        fig.add_trace(go.Scatter(x=model_data['DATE'], y=model_data['PREDICTED_AMOUNT'], mode='lines+markers', name=f'Predicted ({model})'))

    # Highlight the best model
    best_model_data = df[df['MODEL'] == best_model]
    fig.add_trace(go.Scatter(x=best_model_data['DATE'], y=best_model_data['PREDICTED_AMOUNT'], mode='lines+markers', name=f'Best Model ({best_model})', line=dict(width=2.5, dash='dash')))

    # Update layout
    fig.update_layout(
        title='Actual vs Predicted and Forecasted Amounts',
        xaxis_title='Date',
        yaxis_title='Amount',
        legend_title='Legend',
        template='plotly_white'
    )

    st.plotly_chart(fig)  # Display the plot in Streamlit

    # Print the MAE scores and best model
    st.write("MAE Scores by Model:")
    st.write(mae_scores)
    st.write(f"Best Model: {best_model}")


