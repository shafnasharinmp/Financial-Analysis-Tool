# DATA.py
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import io
import base64
from sklearn.metrics import mean_absolute_error

def load_Result_df(file_path):
    df = pd.read_csv(file_path)
    df['DATE'] = pd.to_datetime(df['DATE'])
    return df

def load_Best_Model_df(file_path):
    df = pd.read_csv(file_path)
    return df



def find_best_model(df_predictions):
    """
    Finds the best model based on the least MAE (Mean Absolute Error) between predicted and actual values.
    """
    df_predicted = df_predictions[df_predictions['TYPE'] == 'Predicted']
    mae_per_model = df_predicted.groupby('MODEL').apply(lambda x: mean_absolute_error(x['ACTUAL_AMOUNT'] , x['PREDICTED_AMOUNT']))
    best_model = mae_per_model.idxmin()
    return best_model

def calculate_confidence_score(best_predictions_df):
    """
    Calculate confidence score and confidence intervals from best model predictions.
    """
    actual = best_predictions_df['ACTUAL_AMOUNT'].values
    mean_predictions = best_predictions_df['PREDICTED_AMOUNT'].values
    
    std_dev = np.std(mean_predictions)
    mae = np.mean(np.abs(mean_predictions - actual))
    confidence_score = max(0, min(100, 100 - (mae / std_dev * 100)))

    residuals = np.abs(actual - mean_predictions)
    confidence_percentage = 100 - (np.mean(residuals) / np.mean(actual) * 100)
    confidence_percentage = max(0, min(100, confidence_percentage))

    
    confidence_interval = 1.96 * std_dev
    lower_bound = mean_predictions - confidence_interval
    upper_bound = mean_predictions + confidence_interval
    
    return mean_predictions, actual, std_dev, mae, confidence_score, lower_bound, upper_bound

def plot_results(mean_predictions, actual, lower_bound, upper_bound, time_index, confidence_score):
    """
    Plot the results with confidence intervals.
    """
    data = pd.DataFrame({
        'Time': time_index,
        'Actual': actual,
        'Mean Prediction': mean_predictions,
        'Lower Bound': lower_bound,
        'Upper Bound': upper_bound
    })
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=data['Time'],
        y=data['Mean Prediction'],
        mode='markers+lines',
        marker=dict(size=8, color='blue', opacity=0.7),
        name='Mean Prediction'
    ))
    
    fig.add_trace(go.Scatter(
        x=data['Time'],
        y=data['Actual'],
        mode='lines',
        line=dict(color='red'),
        name='Actual Data'
    ))
    
    fig.add_trace(go.Scatter(
        x=data['Time'],
        y=data['Lower Bound'],
        mode='lines',
        line=dict(color='gray', dash='dash'),
        name='Lower Bound'
    ))
    
    fig.add_trace(go.Scatter(
        x=data['Time'],
        y=data['Upper Bound'],
        mode='lines',
        line=dict(color='gray', dash='dash'),
        name='Upper Bound'
    ))
    
    fig.update_layout(
        title=f'Chart with Confidence Intervals (Confidence Score: {confidence_score:.2f}%)',
        xaxis_title='Time',
        yaxis_title='Amount',
        template='plotly_dark'
    )
    
    return fig

def process_best_model_and_plot(df_amount, df_predictions):
    """
    Processes the best model, calculates confidence scores and intervals, and plots the results.
    """
    # Find the best model
    best_model = find_best_model(df_predictions)
    
    # Filter predictions for the best model
    best_predictions_df = df_predictions[(df_predictions['MODEL'] == best_model) & (df_predictions['TYPE'] == 'Predicted')]
    
    # Calculate confidence scores and intervals
    mean_predictions, actual, std_dev, mae, confidence_score, lower_bound, upper_bound = calculate_confidence_score(best_predictions_df)
    
    # Plot results
    time_index = best_predictions_df['DATE']
    fig = plot_results(mean_predictions, actual, lower_bound, upper_bound, time_index, confidence_score)
    
    return fig