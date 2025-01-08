
import pandas as pd
import numpy as np
from numpy.fft import fft

from llama_index.core.prompts.base import PromptTemplate
from llama_index.core.prompts.prompt_type import PromptType
from llama_index.llms.gemini import Gemini


# Load data
def load_To_ll_data(path):
    df = upload_file(path)
    return df

def load_data_string(df):
    df = df.to_string()
    return df

# Function to generate insights and recommendations
def generate_insights_and_recommendations(df_actul,df_result):
    # Load data for main and result contexts
    data_context_main = load_data_string(df_actul)
    data_context_result = load_data_string(df_result)

    # Initialize the LLM
    llm = Gemini(api_key="AIzaSyDrkreHTnxikbUthuD5HOOQizo7jlWDaiw", model='models/gemini-pro')

    # Define templates for insights and recommendations
    datasum_tmpl = (
        "Insights Extraction:\n"
        "Extract the insights about data with a title Insights:\n"
        "Also extract basic informations about data specifying below contents "
        "Also extract the outliers and the dates corresponding to that, also extract the missing dates "
        "Also give information about trend, seasonality, irregularity, stationarity check"
        "3. Provide actionable recommendations to increase sales.\n"
        "Note that the AMOUNT displayed should be in integers.\n"
        "Provide the details in the following format:\n"
            "insights: [insights of Data]"
            "Actionable Recommendations : [Actionable Recommendations of Data],\n"
        "{context_str}\n"
    )

    datares_tmpl = (
        "1. Extract insights of the data  with the title insights over prediction.\n"
        " -Also Identify the best performing MODEL .\n"
        "2. Also Provide actionable recommendations to increase sales with title 'Recommendation over prediction'.\n"
        "Provide the details in the following format:\n"
            "insights: [insights of Data]"
            "basic information : [basic information of Data]"
        "{context_str}\n"
    )

    # Data summary insights
    datasum_prompt = PromptTemplate(datasum_tmpl, prompt_type=PromptType.KEYWORD_EXTRACT)
    prompt_datasum = datasum_prompt.format(context_str=data_context_main)
    data_summary = llm.complete(prompt_datasum)

    # Data result insights
    datares_prompt = PromptTemplate(datares_tmpl, prompt_type=PromptType.KEYWORD_EXTRACT)
    prompt_datares = datares_prompt.format(context_str=data_context_result)
    data_result = llm.complete(prompt_datares)

    return data_summary, data_result


