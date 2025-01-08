import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import altair as alt
import plotly.express as px
from LOAD import upload_file


def freq_range(freq):
    if freq < 12:
        range_freq = 1
    elif freq >= 12 and freq < 24:
        range_freq = 2
    elif freq >= 24 and freq < 36:
        range_freq = 3
    else:
        range_freq = 4
    return range_freq

from tsmoothie.utils_func import sim_randomwalk
from tsmoothie.smoother import LowessSmoother

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
        smoother = LowessSmoother(smooth_fraction=0.1, iterations=1)
        smoother.smooth(Amt_standardized)

        low, up = smoother.get_intervals('prediction_interval')
        points = smoother.data[0]
        up_points = up[0]
        low_points = low[0]

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
        smoother = LowessSmoother(smooth_fraction=0.1, iterations=1)
        smoother.smooth(Amt_standardized)

        low, up = smoother.get_intervals('prediction_interval')
        points = smoother.data[0]
        up_points = up[0]
        low_points = low[0]

        outlier_mask = (Amt_standardized < low_points) | (Amt_standardized > up_points)
        outliers_df = dates[outlier_mask]
        return ', '.join(pd.to_datetime(outliers_df).dt.strftime("%Y-%m-%d"))
    except Exception as e:
        print(f"Error in detect_outliers: {e}")
        return []

    

def freq_range(freq):
    if freq <= 12:
        return 1
    elif freq > 12 and freq <= 24:
        return 2
    elif freq > 24 and freq <= 36:
        return 3
    else:
        return 4


# Function to calculate summary statistics and save the result

def calculate_summary_statistics(df):
    data = df.copy()
    df.reset_index(inplace=True)
    months_seq = pd.date_range(start=min(df['DATE']), end=max(df['DATE']), freq='MS')

    # Grouping the data
    grouped = df.groupby(['MARKET', 'ACCOUNT_ID', 'CHANNEL_ID', 'MPG_ID'])

    summary_list = []

    for name, group in grouped:
        frequency = len(group)
        frequency_range = freq_range(frequency)
        zero_count = (group['AMOUNT'] == 0).sum()
        negative_count = (group['AMOUNT'] < 0).sum()
        outlier_count = detect_outliers(group['AMOUNT'])
        outlier_dates = detect_outliers_values(group['AMOUNT'], group['DATE'])

        missing_count = len(months_seq) - len(group)
        missing_values = ', '.join([date.strftime("%Y-%m-%d") for date in months_seq[~months_seq.isin(group['DATE'])]])

        summary_list.append({
            'MARKET': name[0],
            'ACCOUNT_ID': name[1],
            'CHANNEL_ID': name[2],
            'MPG_ID': name[3],
            'Frequency': frequency,
            'Frequency_Range': frequency_range,
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




# Example of usage
# data = pd.DataFrame(...)  # Load your DataFrame here
# summary_stats = calculate_summary_statistics(data)
# plot_frequency_range(summary_stats)



def display_performance_analysis(df):
    # Create a copy of the DataFrame to avoid modifying the original
    data = df.copy()
    data['DATE'] = pd.to_datetime(data['DATE'] )
    data.reset_index(drop=True, inplace=True)

    # Grouping the data by specified columns
    combinations = data.groupby(['MARKET', 'ACCOUNT_ID', 'CHANNEL_ID', 'MPG_ID']).size().reset_index()
    combinations['Combination'] = combinations.apply(
        lambda row: f"MARKET: {row['MARKET']}, ACCOUNT_ID: {row['ACCOUNT_ID']}, CHANNEL_ID: {row['CHANNEL_ID']}, MPG_ID: {row['MPG_ID']}",
        axis=1
    )

    # Streamlit app
    st.title('Analysis ')

    # Dropdown menu for selecting a combination
    selected_combination = st.selectbox('Select Combination', combinations['Combination'].unique())

    # Extract selected combination details
    selected_combination_details = combinations[combinations['Combination'] == selected_combination].iloc[0]
    selected_market = selected_combination_details['MARKET']
    selected_account_id = selected_combination_details['ACCOUNT_ID']
    selected_channel_id = selected_combination_details['CHANNEL_ID']
    selected_mpg_id = selected_combination_details['MPG_ID']

    # Filter the DataFrame based on the selected combination
    filtered_data = data[
        (data['MARKET'] == selected_market) & 
        (data['ACCOUNT_ID'] == selected_account_id) & 
        (data['CHANNEL_ID'] == selected_channel_id) & 
        (data['MPG_ID'] == selected_mpg_id)
    ]

    filtered_data['DATE'] = pd.to_datetime(filtered_data['DATE'])

    best_Performing_mpg = (
        filtered_data.groupby(['MARKET', 'ACCOUNT_ID','CHANNEL_ID','MPG_ID'])
        .agg(Total_Amount=('AMOUNT', 'sum'))
        .reset_index()
    )
    best_Performing_mpg = best_Performing_mpg.loc[
    best_Performing_mpg.groupby(['MARKET', 'ACCOUNT_ID', 'CHANNEL_ID'])['Total_Amount'].idxmax()
    ]
    top_5_mpg = best_Performing_mpg.nlargest(5, 'Total_Amount')

    least_Performing_mpg = best_Performing_mpg.loc[
        best_Performing_mpg.groupby(['MARKET', 'ACCOUNT_ID', 'CHANNEL_ID'])['Total_Amount'].idxmin()
    ]
    bottom_5_mpg = least_Performing_mpg.nsmallest(5, 'Total_Amount')



    # # Extract Year and Month
    # filtered_data['Year'] = filtered_data['DATE'].dt.year
    # filtered_data['Month'] = filtered_data['DATE'].dt.month
    # # 1. Best Performing Month for Each Year
    # best_month_per_year = (
    #     filtered_data.groupby(['Year', 'Month'])
    #     .agg(Total_Amount=('AMOUNT', 'sum'))
    #     .reset_index()
    # )
    # best_month_per_year = best_month_per_year.loc[best_month_per_year.groupby('Year')['Total_Amount'].idxmax()]

    # # 2. Best Performing Year
    # best_year = (
    #     filtered_data.groupby('Year')
    #     .agg(Total_Amount=('AMOUNT', 'sum'))
    #     .reset_index()
    #     .nlargest(1, 'Total_Amount')
    # )

    # Get top 5 AMOUNT values
    # top_5_rows = filtered_data.nlargest(5, 'AMOUNT')
    # top_5_rows['DATE'] = top_5_rows['DATE'].dt.strftime("%Y-%m-%d")
    # top_5_rows.reset_index(drop=True)
    # bottom_5_rows = filtered_data.nsmallest(5, 'AMOUNT')
    # bottom_5_rows['DATE'] = bottom_5_rows['DATE'].dt.strftime("%Y-%m-%d")
    # bottom_5_rows.reset_index(drop=True)    


    # Display results in columns
    #col = st.columns((4), gap='medium')
    #with col[0]:
        # st.write("Best Performing Year")
        # st.dataframe(best_year)
        # st.write("Best Performing Month for Each Year")
        # st.dataframe(best_month_per_year)

    #with col[0]:
    st.write("Top Perfoming Products")
    st.dataframe(top_5_mpg)
    st.write("Least Perfoming Products")
    st.dataframe(bottom_5_mpg)
        







