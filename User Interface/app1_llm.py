import os
import pandas as pd
import google.generativeai as genai

# Load data
def load_To_ll_data(path):
    df = upload_file(path)
    return df

def load_data_string(df):
    df = df.copy()
    return df.to_csv(index=False)


# Configure API key
secret = ' '  # Replace with your actual API key
genai.configure(api_key=secret)

def dataframe_to_string(df):
    """Convert DataFrame to a string format suitable for model input."""
    return df.to_csv(index=False)

def generate_insights_and_recommendations(df):
    df_string = load_data_string(df)

    prompt = "Analyze the following data and provide insights.:"
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(f"{prompt}\n\n{df_string}")
    return response.text





