import numpy as np
import pandas as pd
from tsmoothie.smoother import LowessSmoother
import yfinance as yf
from statsmodels.nonparametric.smoothers_lowess import lowess
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
import plotly.subplots as sp
import streamlit as st

def detect_outliers(column_data):
    # Ensure there are enough data points
    if len(column_data) < 3:
        return 0

    Amt = column_data.values

    # Standardize data
    Amt_mean = np.mean(Amt)
    Amt_std = np.std(Amt)
    if Amt_std == 0:
        return 0
    Amt_standardized = (Amt - Amt_mean) / Amt_std

    try:
        smoother = lowess(Amt_standardized, np.arange(len(Amt_standardized)), frac=0.1, return_sorted=False)
        low, up = np.percentile(smoother, [2.5, 97.5])
        points = smoother
        up_points = np.full_like(points, up)
        low_points = np.full_like(points, low)

        outliers = []
        for i in range(len(points)):
            current_point = points[i]
            current_up = up_points[i]
            current_low = low_points[i]
            if current_point > current_up or current_point < current_low:
                outliers.append(current_point)

        return len(outliers)
    except:
        return 0

def detect_outliers_values(column_data, dates):
    if len(column_data) < 3:
        return []

    Amt = column_data.values
    Amt_mean = np.mean(Amt)
    Amt_std = np.std(Amt)
    if Amt_std == 0:
        return []
    Amt_standardized = (Amt - Amt_mean) / Amt_std

    try:
        smoother = lowess(Amt_standardized, np.arange(len(Amt_standardized)), frac=0.1, return_sorted=False)
        low, up = np.percentile(smoother, [2.5, 97.5])
        points = smoother
        up_points = np.full_like(points, up)
        low_points = np.full_like(points, low)

        outlier_mask = (Amt_standardized < low_points) | (Amt_standardized > up_points)
        outliers_df = dates[outlier_mask]
        return ', '.join(pd.to_datetime(outliers_df).dt.strftime("%Y-%m-%d"))
    except Exception as e:
        print(f"Error in detect_outliers: {e}")
        return []

# Function to calculate summary statistics
def calculate_summary_statistics(df):
    data = df.copy()
    months_seq = pd.date_range(start=min(df['Date']), end=max(df['Date']), freq='D')

    summary_list = []

    # Calculate statistics for each group
    group = df.copy()
    zero_count = (group['Close'] == 0).sum()
    negative_count = (group['Close'] < 0).sum()
    outlier_count = detect_outliers(group['Close'])
    outlier_dates = detect_outliers_values(group['Close'], group['Date'])

    missing_count =   len(months_seq) - len(group)
    missing_values = ', '.join([date.strftime("%Y-%m-%d") for date in months_seq[~months_seq.isin(group['Date'])]])

    summary_list.append({
        'Zero_Count': zero_count,
        'Negative_Count': negative_count,
        'Outlier_Count': outlier_count,
        'Outlier_Dates': outlier_dates,
        'Missing_Count': missing_count,
        'Missing_Values': missing_values
    })

    # Create DataFrame from the summary list
    summ_acc = pd.DataFrame(summary_list)
    return summ_acc

def filter_negative_counts(df1, summary_stats):
    negative_counts = summary_stats[summary_stats['Negative_Count'] > 0]
    if not negative_counts.empty:
        filter_conditions =  (df1['Close'] < 0)
        
        filtered_df = df1[filter_conditions]
        
        if not filtered_df.empty:
            st.write("DataFrame :")
            st.dataframe(filtered_df ,hide_index=True)
        else:
            st.write("No Negative Close values found in the filtered data.")
    else:
        st.write("Sorry!...No Negative values present in the Data")
def filter_zero_counts(df1, summary_stats):
    zero_counts = summary_stats[summary_stats['Zero_Count'] > 0]
    if not zero_counts.empty:
        filter_conditions = (df1['Close'] == 0)
        
        filtered_zero_df = df1[filter_conditions]
        
        if not filtered_zero_df.empty:
            st.write("DataFrame :")
            st.dataframe(filtered_zero_df ,hide_index=True)
        else:
            st.write("No Zero Close values found in the filtered data.")
    else:
        st.write("Sorry!...No Zero values present in the Data.")

def filter_outliers(df1, summary_stats):
    outliers = summary_stats[summary_stats['Outlier_Count'] > 0]
    if not outliers.empty:
        outlier_dates = outliers['Outlier_Dates'].dropna().apply(lambda x: x.split(', '))
        outlier_dates_set = set(date for sublist in outlier_dates for date in sublist)
        
        outlier_dates_dt = pd.to_datetime(list(outlier_dates_set))
        
        filter_conditions = (df1['Date'].isin(outlier_dates_dt.strftime('%Y-%m-%d')))
        
        filtered_outlier_df = df1[filter_conditions]
        filtered_outlier_df['Date'] = filtered_outlier_df['Date'].dt.strftime('%Y-%m-%d')
        filtered_outlier_df = filtered_outlier_df.drop('20-day MA', axis=1)
        
        if not filtered_outlier_df.empty:
            st.dataframe(filtered_outlier_df , hide_index=True)
            
        else:
            st.write("No outlier Close values found in the filtered data.")
    else:
        st.write("Sorry!...No Outliers present in the Data.")





# Streamlit UI
def display_summary_statistics(df):
    summary_stats = calculate_summary_statistics(df)
    st.subheader("Summary Statistics")
    df1 = df.copy()

    col = st.columns((5,5), gap='medium')

    with col[0]:                  
        st.subheader("")
        for index, row in summary_stats.iterrows():
            st.info(f"  - Zero Count: {row['Zero_Count']}")
        with st.popover("Zero values"):
            filter_zero_counts(df1, summary_stats)

        st.subheader("")
        for index, row in summary_stats.iterrows():
            outlier_count = row['Outlier_Count']
            st.info(f"  - Outlier Count: {outlier_count}")
            with st.popover("Outlier values"):
                st.write("Dataframe :")
                filter_outliers(df1, summary_stats)

    with col[1]:                    
        st.subheader("")
        for index, row in summary_stats.iterrows():
            st.info(f"  - Negative Count: {row['Negative_Count']}")

        with st.popover("Negative values"):
            filter_negative_counts(df1, summary_stats)

        st.subheader("")
        for index, row in summary_stats.iterrows():
            missing_count = row['Missing_Count']
            st.info(f"  - Missing Count: {missing_count}")
            with st.popover("Missing Dates "):
                if missing_count > 0:
                    st.write(f"  - Missing Dates: {row['Missing_Values']}")
                else:
                    st.write("Sorry!....No Missing Dates Present In the Data")


def plot_time_series_decomposition(df, date_col='Date', value_col='Close'):

    # Ensure the date column is in datetime format
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col)

    # Resample the data to monthly frequency if needed (optional)
    df = df.resample('M').sum()

    # Decompose the time series
    decomposition = sm.tsa.seasonal_decompose(df[value_col], model='additive')

    # Extract components
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid

    # Create subplots
    fig = sp.make_subplots(rows=4, cols=1, shared_xaxes=True,
                           subplot_titles=('Original Time Series', 'Trend Component', 'Seasonality Component', 'Residual Component'))

    # Define x-axis range for all subplots
    xaxis_range = [df.index.min(), df.index.max()]

    # Plot original time series
    fig.add_trace(go.Scatter(x=df.index, y=df[value_col],
                             mode='lines',
                             name='Original',
                             line=dict(color='blue')),
                  row=1, col=1)

    # Plot trend component
    fig.add_trace(go.Scatter(x=trend.index, y=trend,
                             mode='lines',
                             name='Trend',
                             line=dict(color='green')),
                  row=2, col=1)

    # Plot seasonal component
    fig.add_trace(go.Scatter(x=seasonal.index, y=seasonal,
                             mode='lines',
                             name='Seasonality',
                             line=dict(color='orange')),
                  row=3, col=1)

    # Plot residual component
    fig.add_trace(go.Scatter(x=residual.index, y=residual,
                             mode='lines',
                             name='Residual',
                             line=dict(color='red')),
                  row=4, col=1)

    # Update layout for all subplots
    fig.update_layout(height=800, width=1000,
                      title='Time Series Decomposition',
                      xaxis_title='Date',
                      yaxis_title='Amount',
                      showlegend=True,
                      xaxis=dict(
                          tickformat='%b %Y',  # Show month and year on x-axis
                          range=xaxis_range  # Ensure x-axis covers full range of dates
                      ),
                      xaxis2=dict(
                          tickformat='%b %Y',
                          range=xaxis_range
                      ),
                      xaxis3=dict(
                          tickformat='%b %Y',
                          range=xaxis_range
                      ),
                      xaxis4=dict(
                          tickformat='%b %Y',
                          range=xaxis_range
                      ))

    st.plotly_chart(fig)

