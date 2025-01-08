import pandas as pd
import numpy as np

df = pd.read_csv("/content/Updated_date_EU.csv")
df

df.rename(columns={'UPDATED_DATE': 'DATE'}, inplace=True)
df.head()

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

summ_acc = calculate_summary_statistics(df)

summ_acc

summ_acc.to_csv('Summary_statistics_EU.csv')





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

from scipy import stats
df['zscore'] = stats.zscore(df['AMOUNT'])

from tsmoothie.utils_func import sim_randomwalk
from tsmoothie.smoother import LowessSmoother

def detect_outliers(column_name):
    Amt = column_name.values
    smoother = LowessSmoother(smooth_fraction=0.1, iterations=1)
    smoother.smooth(Amt)

    low, up = smoother.get_intervals('prediction_interval')
    points = smoother.data[0]
    up_points = up[0]
    low_points = low[0]


    outliers = []
    for i in range(len(points) - 1, 0, -1):
        current_point = points[i]
        current_up = up_points[i]
        current_low = low_points[i]
        if current_point > current_up or current_point < current_low:
            outliers.append(current_point)
    return len(outliers)

import numpy as np
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
        print(outliers_df)
        outliers_df = pd.DataFrame(outliers_df)
        outliers_df.columns = ['DATE']
        return outliers_df['DATE']

    except Exception as e:
        print(f"Error in detect_outliers: {e}")
        return []

column_data = df['AMOUNT']
dates = df['DATE']
detect_outliers_values(column_data, dates)

df['DATE'] = pd.to_datetime(df['DATE'])
months_seq = pd.date_range(start=min(df['DATE']), end=max(df['DATE']), freq='MS')













summ_acc = df.groupby(['MARKET', 'ACCOUNT_ID', 'CHANNEL_ID', 'MPG_ID']).agg(
        Frequency=('MPG_ID', 'count'),
        Frequency_Range = ('MPG_ID', lambda x: freq_range(len(x))),
        Zero_Count=('AMOUNT', lambda x: (x == 0).sum()),
        Negative_Count=('AMOUNT', lambda x: (x < 0).sum()),
        outlier_ZScore_count=('zscore', lambda x: ((x < -3) | (x > 3)).sum()),
        outlier_count=('AMOUNT', lambda x: detect_outliers(x)),
        outlier_dates = ('AMOUNT', lambda x: detect_outliers_values(x, df.loc[x.index, 'DATE'])),
        missing_count=('DATE', lambda x: len(months_seq) - len(x)),
        missing_values=('DATE', lambda x: ', '.join([date.strftime("%Y-%m-%d") for date in months_seq[~months_seq.isin(x)]]))
    ).reset_index()

summ_acc

summ_acc[summ_acc['missing_count'] == 0]

import pandas as pd
df = pd.DataFrame({
    'Year': [2010, 2011, 2012, 2013, 2014],
    'Sales': [50, 70, 65, 85, 90],
    'Expenses': [40, 60, 55, 75, 80]
})
year_list = list(df['Year'].unique())[::-1]
year_list

grouped = df.groupby(['MARKET', 'ACCOUNT_ID', 'CHANNEL_ID', 'MPG_ID']).filter(lambda x: x['MPG_ID'].count()>35)
grouped

grouped.query("ACCOUNT_ID == 35624 and MARKET == 8252 and CHANNEL_ID == 9653	and MPG_ID == 380306")

grouped.query("ACCOUNT_ID == 35624 and MARKET == 8252 and CHANNEL_ID == 9772 and MPG_ID == 379182")['DATE']















months_seq[~months_seq.isin(['2020-10-01'])]



str(df['DATE'][1])

for date in months_seq.strftime("%Y-%m-%d"):
  if date not in ['2020-08-01', '2020-09-01', '2020-10-01', '2020-11-01',
               '2020-12-01', '2021-01-01', '2021-02-01', '2021-03-01',
               '2021-04-01', '2021-05-01', '2021-06-01', '2021-07-01',
               '2021-08-01']:
    print(date)

summ_acc.to_csv('Summary_statistics_EU.csv')

