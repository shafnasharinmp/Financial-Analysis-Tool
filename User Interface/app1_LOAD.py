import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt


def upload_file(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
        df.drop(columns=['PROD_ID','PERIOD_DATE','Unnamed: 0'], inplace=True)

        date_column = 'UPDATED_DATE'
        df.rename(columns={date_column: 'DATE'}, inplace=True)
    except Exception as e:
        st.error(f"File processing failed: {e}")
        st.toast("Failed!")
        df = None
    return df

def filter_data(df, market, account_ids=None, channel_ids=None, mpg_ids=None):
    """Filter the dataframe based on the selected filters."""
    filtered_data = df[df['MARKET'] == market]
    
    if account_ids:
        filtered_data = filtered_data[filtered_data['ACCOUNT_ID']==(account_ids)]
    
    if channel_ids:
        filtered_data = filtered_data[filtered_data['CHANNEL_ID']==(channel_ids)]
    
    if mpg_ids:
        filtered_data = filtered_data[filtered_data['MPG_ID'].isin(mpg_ids)]
    
    return filtered_data

def filter_page(df, market):
    filtered_df = df[df['MARKET'] == market]
    df = filtered_df.groupby(['MARKET', 'ACCOUNT_ID', 'CHANNEL_ID', 'MPG_ID', 'CURRENCY_ID'])    # Group the DataFrame by specified columns
    return df
