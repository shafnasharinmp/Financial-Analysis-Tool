# -*- coding: utf-8 -*-

import pandas as pd
import google.generativeai as genai
import os
import google.generativeai as genai

# Configure API key
secret = 'AIzaSyDrkreHTnxikbUthuD5HOOQizo7jlWDaiw'  # Replace with your actual API key
genai.configure(api_key=secret)

data = pd.read_csv('C:\Users\hp\OneDrive\Documents\@@@Internship2-MResult\Proj1\Datasets\Updated_date_EU.csv')
data

df = data[
    (data['APP_NAME'] == 'LM_EU_WB') &
    (data['ACCOUNT_ID'] == 35624) &
    (data['MARKET'] == 8252) &
    (data['CHANNEL_ID'] == 9653) &
    (data['MPG_ID'] == 380306) &
    (data['CURRENCY_ID'] == 181)
]


def dataframe_to_string(df):
    """Convert DataFrame to a string format suitable for model input."""
    return df.to_csv(index=False)

def generate_gemini_content(df_string, prompt):
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(f"{prompt}\n\n{df_string}")
    return response.text

# Sample DataFrame
df = df

# Convert DataFrame to string
df_string = dataframe_to_string(df)

# Define a context-specific prompt
prompt = "Analyze the following data and provide insights.:"

# Get insights from the model
insights = generate_gemini_content(df_string, prompt)
print(insights)