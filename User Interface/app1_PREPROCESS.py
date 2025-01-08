# PREPROCESS.py
import pandas as pd
import numpy as np
from tsmoothie.smoother import LowessSmoother, DecomposeSmoother 

def LOAD_impute_dates_and_values(uploaded_file):
        df = pd.read_csv(uploaded_file)
        df['DATE'] = pd.to_datetime(df['DATE'])
        return df



def LOAD_impute_outliers(uploaded_file):
        df = pd.read_csv(uploaded_file)
        df['DATE'] = pd.to_datetime(df['DATE'])
        return df
   
def preprocessed_data(df,df1):
        Prep_df = impute_dates_and_values(df)
        Prep2_df = impute_outliers(Prep_df)
        return Prep2_df
