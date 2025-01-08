# DATA.py
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import io
import base64

def calculate_confidence_score(predictions_rf, predictions_xgb, predictions_tft, y_test):
    predictions_rf = predictions_rf.flatten()
    predictions_xgb = predictions_xgb.flatten()
    predictions_tft = predictions_tft.flatten()
    actual = y_test.flatten()
    predictions = np.array([predictions_rf, predictions_xgb, predictions_tft])
    mean_predictions = np.mean(predictions, axis=0)
    
    # Calculate the standard deviation of the predictions
    std_dev = np.std(mean_predictions)
    
    # Calculate the mean absolute error
    mae = np.mean(np.abs(mean_predictions - actual))
    
    # Calculate the confidence score
    confidence_score = max(0, min(100, 100 - (mae / std_dev * 100)))
    
    # Calculate confidence intervals (assuming 95% CI)
    confidence_interval = 1.96 * std_dev
    lower_bound = mean_predictions - confidence_interval
    upper_bound = mean_predictions + confidence_interval
    
    return mean_predictions, actual, std_dev, mae, confidence_score, lower_bound, upper_bound

def plot_results(mean_predictions, actual, lower_bound, upper_bound, time_index, confidence_score):
    # Data for plotting
    data = pd.DataFrame({
        'Time': time_index,
        'Actual': actual,
        'Mean Prediction': mean_predictions,
        'Lower Bound': lower_bound,
        'Upper Bound': upper_bound
    })
    
    # Create a Plotly bubble chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=data['Time'],
        y=data['Mean Prediction'],
        mode='markers+lines',
        marker=dict(size=20, color='blue', opacity=0.7),
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
        yaxis_title='AMOUNT',
        template='plotly_dark'
    )
    
    return fig
