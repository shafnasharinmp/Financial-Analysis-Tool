import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import altair as alt
import plotly.express as px
from LOAD import upload_file
import plotly.express as px
import plotly.graph_objects as go 
from tsmoothie.utils_func import sim_randomwalk
from tsmoothie.smoother import LowessSmoother
import statsmodels.api as sm
import plotly.subplots as sp


def LOAD_summary_statistics(uploaded_file):
        summary_stats = pd.read_csv(uploaded_file)
        return summary_stats
  
def plot_frequency_range(summary_stats):
    # Prepare data for bar chart
    frequency_ranges = summary_stats['Frequency_Range'].value_counts().sort_index()
    frequency_range_df = frequency_ranges.reset_index()
    frequency_range_df.columns = ['Frequency_Range', 'Count']

    # Create a bar chart for frequency ranges
    # Create a bar chart with Altair
    chart = alt.Chart(frequency_range_df).mark_bar().encode(
        x=alt.X('Frequency_Range:O', title='Frequency Range', axis=alt.Axis(labelAngle=-0)),
        y=alt.Y('Count:Q', title='Count'),
    ).properties(
        width=600,
        height=350,
        title='Frequency Range Distribution'  
    )
    st.altair_chart(chart, use_container_width=True)

def plot_donut_chart(summary_stats,selected_feature):
    frequency_counts = summary_stats.groupby(selected_feature)['Frequency_Range'].value_counts().reset_index()
    frequency_counts.columns = [selected_feature, 'Frequency_Range', 'Count']
    fig = px.pie(frequency_counts, names='Frequency_Range', values='Count', hole=0.5, 
                 title=f'Donut Chart of Frequency Ranges by {selected_feature}')
    fig.update_layout(
        width=300,  
        height=300
    )
    return fig





def display_performance_analysis(df):
    aggregated_data = df.groupby(['MARKET', 'ACCOUNT_ID', 'CHANNEL_ID', 'MPG_ID'])['AMOUNT'].sum().reset_index()
    sorted_data = aggregated_data.sort_values(by='AMOUNT', ascending=False)
    #st.dataframe(sorted_data)
    top_5_products = sorted_data.head(5).reset_index(drop=True)
    bottom_5_products = sorted_data.tail(5).reset_index(drop=True)


    with st.container():
        col1, col2 = st.columns(2, gap='medium')

        with col1:
            with st.expander("Top Performing Products"):
                st.write(top_5_products ,hide_index=True)

        with col2:
            with st.expander("Least Performing Products"):
                st.write(bottom_5_products ,hide_index=True)

def display_performance_analysis_filtered(df):
    aggregated_data = df.groupby(['MARKET', 'ACCOUNT_ID', 'CHANNEL_ID', 'MPG_ID'])['AMOUNT'].sum().reset_index()
    sorted_data = aggregated_data.sort_values(by='AMOUNT', ascending=False)
    #st.dataframe(sorted_data)
    top_5_products = sorted_data.head(1).reset_index(drop=True)
    bottom_5_products = sorted_data.tail(1).reset_index(drop=True)


    with st.container():
        col1, col2 = st.columns(2, gap='medium')

        with col1:
            with st.expander("Top Performing Product"):
                st.write(top_5_products,hide_index=True)

        with col2:
            with st.expander("Least Performing Product"):
                st.write(bottom_5_products ,hide_index=True)
  

def filter_negative_counts(df1, summary_stats):
    negative_counts = summary_stats[summary_stats['Negative_Count'] > 0]
    if not negative_counts.empty:
        filter_conditions = (df1['MARKET'].isin(negative_counts['MARKET'])) & \
                            (df1['ACCOUNT_ID'].isin(negative_counts['ACCOUNT_ID'])) & \
                            (df1['CHANNEL_ID'].isin(negative_counts['CHANNEL_ID'])) & \
                            (df1['MPG_ID'].isin(negative_counts['MPG_ID'])) & \
                            (df1['AMOUNT'] < 0)
        
        filtered_df = df1[filter_conditions]
        
        if not filtered_df.empty:
            st.write("DataFrame :")
            st.dataframe(filtered_df ,hide_index=True)
        else:
            st.write("No Negative AMOUNT values found in the filtered data.")
    else:
        st.write("Sorry!...No Negative values present in the Data")
def filter_zero_counts(df1, summary_stats):
    zero_counts = summary_stats[summary_stats['Zero_Count'] > 0]
    if not zero_counts.empty:
        filter_conditions = (df1['MARKET'].isin(zero_counts['MARKET'])) & \
                            (df1['ACCOUNT_ID'].isin(zero_counts['ACCOUNT_ID'])) & \
                            (df1['CHANNEL_ID'].isin(zero_counts['CHANNEL_ID'])) & \
                            (df1['MPG_ID'].isin(zero_counts['MPG_ID'])) & \
                            (df1['AMOUNT'] == 0)
        
        filtered_zero_df = df1[filter_conditions]
        
        if not filtered_zero_df.empty:
            st.write("DataFrame :")
            st.dataframe(filtered_zero_df ,hide_index=True)
        else:
            st.write("No Zero AMOUNT values found in the filtered data.")
    else:
        st.write("Sorry!...No Zero values present in the Data.")

def filter_outliers(df1, summary_stats):
    outliers = summary_stats[summary_stats['Outlier_Count'] > 0]
    if not outliers.empty:
        outlier_dates = outliers['Outlier_Dates'].dropna().apply(lambda x: x.split(', '))
        outlier_dates_set = set(date for sublist in outlier_dates for date in sublist)
        
        outlier_dates_dt = pd.to_datetime(list(outlier_dates_set))
        
        filter_conditions = (df1['DATE'].isin(outlier_dates_dt.strftime('%Y-%m-%d')))
        
        filtered_outlier_df = df1[filter_conditions]
        
        if not filtered_outlier_df.empty:
            st.dataframe(filtered_outlier_df , hide_index=True)
        else:
            st.write("No outlier AMOUNT values found in the filtered data.")
    else:
        st.write("Sorry!...No Outliers present in the Data.")


def filter_missing_values(df1, summary_stats):
    missing_counts = summary_stats[summary_stats['Missing_Count'] > 0]
    if not missing_counts.empty:
        missing_dates = missing_counts['Missing_Values'].dropna().apply(lambda x: x.split(', '))
        missing_dates_set = set(date for sublist in missing_dates for date in sublist)
        
        missing_dates_dt = pd.to_datetime(list(missing_dates_set))
        
        filter_conditions = (df1['DATE'].isin(missing_dates_dt.strftime('%Y-%m-%d')))
        
        filtered_missing_df = df1[filter_conditions]
        
        if not filtered_missing_df.empty:
            st.write("DataFrame :")
            st.dataframe(filtered_missing_df , hide_index=True)
        else:
            st.write("No missing AMOUNT values found in the filtered data.")
    else:
        st.write("Sorry!...No missing values present in the Data.")



def plot_outliers_and_imputed_values(df1, summary_stats, Imputed_Outlier_df):
    """
    Filters outliers from df1 based on summary_stats and plots outliers and imputed values.

    Parameters:
    - df1 (pd.DataFrame): The original DataFrame containing outlier data.
    - summary_stats (pd.DataFrame): The summary statistics DataFrame with outlier information.
    - Imputed_Outlier_df (pd.DataFrame): The DataFrame containing imputed values for outliers.
    """

    # Filter summary_stats for outliers
    outliers = summary_stats[summary_stats['Outlier_Count'] > 0]
    if not outliers.empty:
        # Extract outlier dates and convert to datetime
        outlier_dates = outliers['Outlier_Dates'].dropna().apply(lambda x: x.split(', '))
        outlier_dates_set = set(date for sublist in outlier_dates for date in sublist)
        outlier_dates_dt = pd.to_datetime(list(outlier_dates_set))
        
        # Filter df1 based on outlier dates
        filter_conditions = (df1['DATE'].isin(outlier_dates_dt.strftime('%Y-%m-%d')))
        filtered_outlier_df = df1[filter_conditions]
        
        if not filtered_outlier_df.empty:
            st.write("DataFrame with Outliers:")
            st.dataframe(filtered_outlier_df , hide_index=True)
            
            # Merge with Imputed Outliers DataFrame for plotting
            imputed_outliers = Imputed_Outlier_df[Imputed_Outlier_df['DATE'].isin(outlier_dates_dt)]
            
            # Create Plotly figure
            fig = go.Figure()
            
            # Plot actual outlier values
            fig.add_trace(go.Scatter(x=df1['DATE'],
                                     y=df1['AMOUNT'],
                                     mode='lines+markers',
                                     name='Actual Data',
                                     line=dict(color='blue'),
                                     marker=dict(size=10)))
            
            # Plot imputed outlier values
            fig.add_trace(go.Scatter(x=Imputed_Outlier_df['DATE'],
                                     y=Imputed_Outlier_df['AMOUNT'],
                                     mode='lines+markers',
                                     name='Imputed Data',
                                     line=dict(color='red'),
                                     marker=dict(size=10)))
            
            fig.update_layout(title='Actual vs Imputed Outliers',
                              xaxis_title='Date',
                              yaxis_title='Amount',
                              legend_title='Legend',
                              xaxis=dict(tickformat='%Y-%m'),
                              yaxis=dict(title='Amount'))
            
            st.plotly_chart(fig)

        else:
            st.write("No outlier AMOUNT values found in the filtered data.")
    else:
        st.write("Sorry!...No Outliers present in the Data.")



def plot_missing_dates_and_comparisons(df1, imputed_dates_df):
    # Ensure DATE columns are in datetime format
    df1['DATE'] = pd.to_datetime(df1['DATE'])
    imputed_dates_df['DATE'] = pd.to_datetime(imputed_dates_df['DATE'])
    
    # Identify missing dates in df1
    actual_dates = set(df1['DATE'])
    imputed_dates = set(imputed_dates_df['DATE'])
    missing_dates = imputed_dates - actual_dates
    
    if missing_dates:
        missing_dates_df = imputed_dates_df[imputed_dates_df['DATE'].isin(missing_dates)]
        
        # Create Plotly figure
        fig = go.Figure()
        
        # Plot actual data
        fig.add_trace(go.Scatter(x=df1['DATE'],
                                 y=df1['AMOUNT'],
                                 mode='lines+markers',
                                 name='Actual Data',
                                 marker=dict(color='blue'),
                                 line=dict(color='blue')))
        
        # Plot imputed data for missing dates
        fig.add_trace(go.Scatter(x=missing_dates_df['DATE'],
                                 y=missing_dates_df['AMOUNT'],
                                 mode='markers',
                                 name='Imputed Data',
                                 marker=dict(color='red', size=10)))
        
        fig.update_layout(title='Actual vs Imputed Data for Missing Dates',
                           xaxis_title='Date',
                           yaxis_title='Amount',
                           legend_title='Legend',
                          xaxis=dict(tickformat='%Y-%m'),
                          yaxis=dict(title='Amount'))
        
        st.plotly_chart(fig)
        
        st.write("DataFrame with Imputed Data:")
        st.dataframe(missing_dates_df , hide_index=True)
    else:
        st.write("Sorry!...No Missing Dates present in the Data.")






def plot_time_series_decomposition(df, date_col='DATE', value_col='AMOUNT'):
    """
    Decomposes the time series into trend, seasonality, and residuals and plots the results in separate graphs.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing time series data.
    - date_col (str): The name of the date column.
    - value_col (str): The name of the value column.
    """

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

