import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

df = pd.read_csv("/content/gm_input_EU.csv")

df['PERIOD_DATE'] = pd.to_datetime(df['PERIOD_DATE'])
df

total_months = df['PERIOD_DATE'].dt.to_period('M').nunique()
print('Total number of unique months used in the dataset for the years 2017-2020:', total_months)

df.info()

print("Shape of DataFrame:", df.shape)

df.isnull().sum()

df.duplicated().sum()

df[['ACCOUNT_ID', 'MARKET', 'CHANNEL_ID', 'MPG_ID', 'CURRENCY_ID']] = df[['ACCOUNT_ID', 'MARKET', 'CHANNEL_ID', 'MPG_ID', 'CURRENCY_ID']].astype(str)
print(df.dtypes)

columns_to_check = df.select_dtypes(include=['object']).columns

for column in columns_to_check:
    unique_count = df[column].nunique()
    print(f"No of unique values in '{column}'   :{unique_count}")

print(df['MARKET'].unique())

"""DATE CONVERSION"""

from datetime import date
new_date = date.today()
new_date = pd.to_datetime(new_date)
new_date

df['DIFF'] = (new_date - (df['PERIOD_DATE'].max()))
df['UPDATED_DATE'] = df['PERIOD_DATE'] + pd.to_timedelta(df['DIFF'], unit='D')

def replace_days(date):
  return date.replace(day=1)

df['UPDATED_DATE'] = df['UPDATED_DATE'].apply(replace_days)
df = df.drop('DIFF', axis = 1)
df['UPDATED_DATE'].dtype

df['UPDATED_DATE'] = df['UPDATED_DATE'].dt.strftime('%Y/%m/%d')
df['PERIOD_DATE'] = df['PERIOD_DATE'].dt.strftime('%Y/%m/%d')
df[df['UPDATED_DATE']== '2020/09/01' ]


df_filtered = df.sort_values('UPDATED_DATE')
df_filtered.shape

df_filtered.to_csv('Updated_date_EU.csv')

df_filtered.query("ACCOUNT_ID == 35315 and MARKET == 8233 and CHANNEL_ID == 9774	and MPG_ID == 380436")







"""EXTRA"""

grouped_df = df.groupby(['APP_NAME','MARKET', 'ACCOUNT_ID', 'CHANNEL_ID', 'MPG_ID'])['PERIOD_DATE'].unique()
grouped_df

grouped_df_updated = df_filtered.groupby(['APP_NAME','MARKET', 'ACCOUNT_ID', 'CHANNEL_ID', 'MPG_ID'])['PERIOD_DATE'].unique()
grouped_df_updated

grouped_updated = df_filtered.groupby(['APP_NAME','MARKET', 'ACCOUNT_ID', 'CHANNEL_ID', 'MPG_ID'])['UPDATED_DATE'].apply(list)
grouped_updated

grouped_updated = df_filtered.groupby(['APP_NAME','MARKET', 'ACCOUNT_ID', 'CHANNEL_ID', 'MPG_ID'])['UPDATED_DATE'].apply(list)

# Validate the groupby result with the updated date column
for group_name, updated_dates in grouped_updated.items():

    original_dates = df_filtered.loc[(df_filtered['APP_NAME'] == group_name[0]) &
                                      (df_filtered['MARKET'] == group_name[1]) &
                                      (df_filtered['ACCOUNT_ID'] == group_name[2]) &
                                      (df_filtered['CHANNEL_ID'] == group_name[3]) &
                                      (df_filtered['MPG_ID'] == group_name[4]), 'PERIOD_DATE'].tolist()


    assert len(updated_dates) == len(original_dates), f"Length mismatch for group {group_name}"


    for updated_date, original_date in zip(updated_dates, original_dates):
        assert updated_date >= original_date, f"Updated date {updated_date} is not greater than or equal to original date {original_date} for group {group_name}"

print("Validation successful.")

df[['UPDATED_DATE','PERIOD_DATE']].drop_duplicates()























print(df['PERIOD_DATE'].max())
df['PERIOD_DATE'] = pd.to_datetime(df['PERIOD_DATE'], format='%Y-%m-%d')
df['diff'] = pd.to_datetime('2024-04-01') - df['PERIOD_DATE'].max()

# Adjusting the updated date to the first day of the next month if it's not already
df['UPDATED_DATE'] = df['PERIOD_DATE'] + pd.to_timedelta(df['diff'], unit='D')
print(df)

def replace_days(date):
    return date.replace(day=1)

# Apply replace_days function to update days to 1st of the month
df['UPDATED_DATE'] = df['UPDATED_DATE'].apply(replace_days)

def outliers(NORMALIZED_AMOUNT):
    quart1 = np.quantile(NORMALIZED_AMOUNT, 0.25)
    quart3 = np.quantile(NORMALIZED_AMOUNT, 0.75)
    iqr = quart3 - quart1
    return np.sum((NORMALIZED_AMOUNT < (quart1 - 1.5 * iqr)) | (NORMALIZED_AMOUNT > (quart3 + 1.5 * iqr)))

outlier_count = outliers(df['NORMALIZED_AMOUNT'])
print(outlier_count)

if outlier_count > 0:
    print("Outliers are present.")
else:
    print("No outliers are present.")

Q1 = df['AMOUNT'].quantile(0.25)
Q3 = df['AMOUNT'].quantile(0.75)
IQR = Q3 - Q1
outliers = (df['AMOUNT'] < (Q1 - 1.5 * IQR)) | (df['AMOUNT'] > (Q3 + 1.5 * IQR))
print(outliers)

from sklearn.preprocessing import MinMaxScaler
def normalize_amount(amount_column):
    scaler = MinMaxScaler()
    normalized_amount = scaler.fit_transform(amount_column.values.reshape(-1, 1))
    return normalized_amount.flatten()

# Apply normalization to the 'AMOUNT' column
df['NORMALIZED_AMOUNT'] = normalize_amount(df['AMOUNT'])
df





