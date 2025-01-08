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
from sklearn.metrics import mean_absolute_error
import plotly.graph_objs as go
import plotly.express as px
import streamlit as st



def load_Result(file_path):
    df = pd.read_csv(file_path)
    df['DATE'] = pd.to_datetime(df['DATE'])
    return df

def load_Best_Model(file_path):
    df = pd.read_csv(file_path)
    return df

# df = load_Result(file_path)
# best_models = load_Best_Model(file_path)
# df = df.merge(best_models, on=['MARKET', 'ACCOUNT_ID', 'CHANNEL_ID', 'MPG_ID'], how='left')






def find_best_model(df_predictions):
    df_predicted = df_predictions[df_predictions['TYPE'] == 'Predicted']
    mae_per_model = df_predicted.groupby('MODEL').apply(lambda x: mean_absolute_error(x['ACTUAL_AMOUNT'], x['PREDICTED_AMOUNT']))
    best_model = mae_per_model.idxmin()
    return best_model

def plot_actual_vs_predicted(df_amount, df_predictions):
    """
    Plots AMOUNT from df_amount and PREDICTED_AMOUNT for 3 models from df_predictions using Plotly.

    Parameters:
    df_amount (pd.DataFrame): DataFrame containing actual amounts with columns ['APP_NAME', 'ACCOUNT_ID', 'MARKET', 'CHANNEL_ID', 'MPG_ID', 'CURRENCY_ID', 'AMOUNT', 'month_index', 'DATE']
    df_predictions (pd.DataFrame): DataFrame containing predicted amounts with columns ['MARKET', 'ACCOUNT_ID', 'CHANNEL_ID', 'MPG_ID', 'DATE', 'ACTUAL_AMOUNT', 'TYPE', 'MODEL', 'PREDICTED_AMOUNT']
    """

    # Ensure DATE columns are datetime
    df_amount['DATE'] = pd.to_datetime(df_amount['DATE'])
    df_predictions['DATE'] = pd.to_datetime(df_predictions['DATE'])

    # Find the best model based on a specific metric, e.g., MAE
    best_model = find_best_model(df_predictions)

    # Create traces for actual amounts
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_amount['DATE'], y=df_amount['AMOUNT'], mode='lines+markers', name='Actual Amount', line=dict(color='blue')))

    # Create traces for predicted amounts and highlight the best model
    models = df_predictions['MODEL'].unique()
    colors = px.colors.qualitative.Plotly

    for model, color in zip(models, colors):
        df_model = df_predictions[df_predictions['MODEL'] == model]
        fig.add_trace(go.Scatter(x=df_model['DATE'], y=df_model['PREDICTED_AMOUNT'], mode='lines+markers', name=f'Predicted Amount ({model})', line=dict(color=color)))

        # Highlight the best model with markers and annotations
        if model == best_model:
            fig.add_trace(go.Scatter(
                x=df_model['DATE'],
                y=df_model['PREDICTED_AMOUNT'],
                mode='markers',
                name=f'Best Model: {model}',
                marker=dict(color='black', size=10, symbol='star'),
                text=[f'Best Model: {model}' for _ in range(len(df_model))],
                hoverinfo='text'
            ))

    # Update layout with black background and no grid
    fig.update_layout(
        title='Actual Amount vs Predicted Amounts for Different Models',
        xaxis_title='Date',
        yaxis_title='Amount',
        legend_title='Legend',
        template='plotly_dark',
        #plot_bgcolor='black',
        #paper_bgcolor='black',
        font=dict(color='white'),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False)
    )

    return fig

