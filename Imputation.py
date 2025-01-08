from tsmoothie.utils_func import sim_randomwalk
from tsmoothie.smoother import LowessSmoother, DecomposeSmoother

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import altair as alt
import plotly.express as px
from statsmodels.tsa.seasonal import seasonal_decompose
import time

def load_data():
    data = pd.read_csv('/content/Updated_date_EU.csv')
    return data

data = load_data()
data

filtered_grouped = data
all_results_df = pd.DataFrame(columns=['APP_NAME', 'ACCOUNT_ID', 'MARKET', 'CHANNEL_ID', 'MPG_ID', 'CURRENCY_ID','month_index', 'AMOUNT'])
months_seq = pd.date_range(start=min(df['DATE']), end=max(df['DATE']), freq='MS')
filtered_grouped = filtered_grouped.groupby(['MARKET', 'ACCOUNT_ID', 'CHANNEL_ID','MPG_ID'])
filtered_grouped.query('MARKET == 8223 and ACCOUNT_ID == 35561 and CHANNEL_ID ==9813 and MPG_ID == 380340')
for i, (name, group) in enumerate(filtered_grouped, start=1):
  print(f"Group_{i}:")
  data = group.copy()
  data['DATE'] = pd.to_datetime(data['DATE'])
  data.set_index('DATE', inplace=True)
  amount = data['AMOUNT'].copy()


  print(len(months_seq))
  print(len(data.index))
  Missing_Count = (len(months_seq) - len(data.index))
  print("Missing Count:", Missing_Count)


  missing_dates = [date for date in months_seq if date not in data.index]
  missing_dates_str = ', '.join([date.strftime("%Y/%m/%d") for date in missing_dates])
  print("Missing Values:", missing_dates_str)

  missing_df = pd.DataFrame({
          'APP_NAME': [data['APP_NAME'].iloc[0]] * len(missing_dates),
          'ACCOUNT_ID': [data['ACCOUNT_ID'].iloc[0]] * len(missing_dates),
          'MARKET': [data['MARKET'].iloc[0]] * len(missing_dates),
          'CHANNEL_ID': [data['CHANNEL_ID'].iloc[0]] * len(missing_dates),
          'MPG_ID': [data['MPG_ID'].iloc[0]] * len(missing_dates),
          'CURRENCY_ID': [data['CURRENCY_ID'].iloc[0]] * len(missing_dates),
          'AMOUNT': [0] * len(missing_dates),
          'month_index': [None] * len(missing_dates)
  })

  missing_df.set_index('DATE', inplace=True)
  imputed_df = pd.concat([data, missing_df])
  imputed_df = imputed_df.sort_index().reset_index()

  amount_actual = data['AMOUNT'].copy()
  amount = imputed_df['AMOUNT'].copy()
  Amt = imputed_df['AMOUNT'].values
  assert len(imputed_df) == len(months_seq), "The length of imputed_df should be equal to the length of months_seq."


  seasonal_smoother = DecomposeSmoother(smooth_type='lowess', periods=45, smooth_fraction=0.1)
  seasonal_smoother.smooth(Amt)
  seasonal_smoothed_data = seasonal_smoother.smooth_data[0]
  amount = imputed_df['AMOUNT'].copy()
  Amt = imputed_df['AMOUNT'].values
  missing_idxs = np.where(Amt == 0)[0]

  for idx in missing_idxs:
    Amt[idx] = seasonal_smoothed_data[idx]

  # Handling condition where 2 nearest indices have the same amount
  for idx in range(1, len(imputed_df) - 1):
            if imputed_df.at[idx, 'AMOUNT'] == imputed_df.at[idx - 1, 'AMOUNT']:
                if imputed_df.at[idx, 'AMOUNT'] == imputed_df.at[idx + 1, 'AMOUNT']:
                    # Sum the values if both previous and next values are the same
                    imputed_df.at[idx, 'AMOUNT'] += imputed_df.at[idx + 1, 'AMOUNT']
                else:
                    # Introduce a small variation based on trend if not possible
                    trend = (imputed_df.at[idx + 1, 'AMOUNT'] - imputed_df.at[idx - 1, 'AMOUNT']) / 2
                    imputed_df.at[idx, 'AMOUNT'] += trend

  # Impute missing values for 'month_index' using incrementing backfill
  imputed_df['month_index'] = imputed_df['month_index'].fillna(method='bfill').fillna(method='ffill')
  imputed_df['month_index'] = imputed_df['month_index'].astype(int)
  imputed_df['month_index'] = np.arange(imputed_df['month_index'].min(), imputed_df['month_index'].min() + len(imputed_df))

  print('________________________________')
  #if Missing_Count > 0:
  print(Amt)
      #print(missing_idxs)
      #print(len(seasonal_smoothed_data2))
      #print(len(imputed_df['AMOUNT'].values))
      #pd.set_option('display.float_format', lambda x: '%.2f' % x)
      #comparison = pd.DataFrame({'Actual': amount, 'Imputed': imputed_df['AMOUNT']})
      #print(comparison.head(45))
      #print(imputed_df.head(45))
  print('________________________________')

  # Plot the actual and imputed 'AMOUNT' values
  plt.figure(figsize=(10, 5))
  plt.plot(data.index, data['AMOUNT'], label='Actual', color='blue')
  plt.scatter(imputed_df['DATE'], imputed_df['AMOUNT'], label='Imputed', color='red')
  plt.xlabel('Date')
  plt.ylabel('Amount')
  plt.title(f'Group {i}: Actual vs Imputed Amounts')
  plt.legend()
  plt.show()

all_results_df = pd.DataFrame(columns=['APP_NAME', 'ACCOUNT_ID', 'MARKET', 'CHANNEL_ID', 'MPG_ID', 'CURRENCY_ID','month_index', 'AMOUNT'])
months_seq = pd.date_range(start=min(df['DATE']), end=max(df['DATE']), freq='MS')

for i, (name, group) in enumerate(filtered_grouped, start=1):
  print(f"Group_{i}:")
  data = group.copy()
  data['DATE'] = pd.to_datetime(data['DATE'])  e
  data.set_index('DATE', inplace=True)
  amount = data['AMOUNT'].copy()


  print(len(months_seq))
  print(len(data.index))
  Missing_Count = (len(months_seq) - len(data.index))
  print("Missing Count:", Missing_Count)


  missing_dates = [date for date in months_seq if date not in data.index]
  missing_dates_str = ', '.join([date.strftime("%Y/%m/%d") for date in missing_dates])
  print("Missing Values:", missing_dates_str)

  missing_df = pd.DataFrame({
          'DATE': missing_dates,
          'APP_NAME': [data['APP_NAME'].iloc[0]] * len(missing_dates),
          'ACCOUNT_ID': [data['ACCOUNT_ID'].iloc[0]] * len(missing_dates),
          'MARKET': [data['MARKET'].iloc[0]] * len(missing_dates),
          'CHANNEL_ID': [data['CHANNEL_ID'].iloc[0]] * len(missing_dates),
          'MPG_ID': [data['MPG_ID'].iloc[0]] * len(missing_dates),
          'CURRENCY_ID': [data['CURRENCY_ID'].iloc[0]] * len(missing_dates),
          'AMOUNT': [0] * len(missing_dates),
          'month_index': [None] * len(missing_dates)
  })

  missing_df.set_index('DATE', inplace=True)
  imputed_df = pd.concat([data, missing_df])
  imputed_df = imputed_df.sort_index().reset_index()


  amount_actual = data['AMOUNT'].copy()   # Extract the 'AMOUNT' column for later use if needed
  amount = imputed_df['AMOUNT'].copy()    # Extract the 'AMOUNT' column with null values for missing dates for later use if needed
  Amt = imputed_df['AMOUNT'].values   # Extract the 'AMOUNT' values with null values for missing dates for later use if needed(Used for imputation)
  assert len(imputed_df) == len(months_seq), "The length of imputed_df should be equal to the length of months_seq."

  # Apply a Lowess smoothing technique to the 'AMOUNT' values
  seasonal_smoother = DecomposeSmoother(smooth_type='lowess', periods=45, smooth_fraction=0.1)
  seasonal_smoother.smooth(Amt)
  seasonal_smoothed_data = seasonal_smoother.smooth_data[0]
  amount = imputed_df['AMOUNT'].copy()
  Amt = imputed_df['AMOUNT'].values
  missing_idxs = np.where(Amt == 0)[0]

  for idx in missing_idxs:
    Amt[idx] = seasonal_smoothed_data[idx]


  # Impute missing values for 'month_index' using incrementing backfill
  imputed_df['month_index'] = imputed_df['month_index'].fillna(method='bfill').fillna(method='ffill')
  imputed_df['month_index'] = imputed_df['month_index'].astype(int)
  imputed_df['month_index'] = np.arange(imputed_df['month_index'].min(), imputed_df['month_index'].min() + len(imputed_df))
  print('________________________________')
  #if Missing_Count > 0:
      #print(len(Amt))
      #print(missing_idxs)
      #print(len(seasonal_smoothed_data2))
      #print(len(imputed_df['AMOUNT'].values))
      #pd.set_option('display.float_format', lambda x: '%.2f' % x)
      #comparison = pd.DataFrame({'Actual': amount, 'Imputed': imputed_df['AMOUNT']})
      #print(comparison.head(45))
      #print(imputed_df.head(45))
  print('________________________________')

  # Plot the actual and imputed 'AMOUNT' values
  plt.figure(figsize=(10, 5))
  plt.plot(data.index, data['AMOUNT'], label='Actual', color='blue')
  plt.scatter(imputed_df['DATE'], imputed_df['AMOUNT'], label='Imputed', color='red')
  plt.ylabel('Amount')
  plt.title(f'Group {i}: Actual vs Imputed Amounts')
  plt.legend()
  plt.show()

  print('__________________________________________________________________________________________________________________________________')

  # Add metadata to the imputed DataFrame
  imputed_df['APP_NAME'] = 'LM_EU_WB'
  imputed_df['MARKET'] = name[0]
  imputed_df['ACCOUNT_ID'] = name[1]
  imputed_df['CHANNEL_ID'] = name[2]
  imputed_df['MPG_ID'] = name[3]
  imputed_df['CURRENCY_ID'] = name[4]

  imputed_df = imputed_df[[ 'APP_NAME', 'ACCOUNT_ID', 'MARKET', 'CHANNEL_ID', 'MPG_ID', 'CURRENCY_ID','month_index', 'AMOUNT', 'DATE']]
  print(f"Group_{i}: {imputed_df.head(3)}")
  all_results_df = pd.concat([all_results_df, imputed_df], ignore_index=True)
  print('__________________________________________________________________________________________________________________________________')


all_results_df.to_csv('Imputed_Dates_EU.csv', index=False)   e
print("All results saved to 'Imputed_Dates_EU.csv'")

