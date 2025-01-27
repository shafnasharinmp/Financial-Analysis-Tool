#nixtla>=0.5.1
import pandas as pd
from nixtla import NixtlaClient

df = pd.read_csv('/content/Updated_date_EU.csv')
print(df.shape)
df.head()

df.drop(columns=['PROD_ID','PERIOD_DATE','Unnamed: 0'], inplace=True)
df.rename(columns={'UPDATED_DATE': 'DATE'}, inplace=True)
data = df.query("MARKET == 8223 and ACCOUNT_ID == 35315 and CHANNEL_ID == 9813	and MPG_ID == 380360")
data.shape

df =data
df.head()

api_key= 'nixtla-tok-....'
nixtla_client = NixtlaClient(api_key)

nixtla_client.validate_api_key()

nixtla_client.plot(df, time_col='DATE', target_col='AMOUNT')

timegpt_fcst_df = nixtla_client.forecast(df=df, h=12, freq='MS', time_col='DATE', target_col='AMOUNT')
timegpt_fcst_df.head()

nixtla_client.plot(df, timegpt_fcst_df, time_col='DATE', target_col='AMOUNT')

"""Longer Forecast"""

timegpt_fcst_df = nixtla_client.forecast(df=df, h=36, time_col='DATE', target_col='AMOUNT', freq='MS', model='timegpt-1-long-horizon')
timegpt_fcst_df.head()

nixtla_client.plot(df, timegpt_fcst_df, time_col='DATE', target_col='AMOUNT')

"""Shorter Forecast"""

timegpt_fcst_df = nixtla_client.forecast(df=df, h=6, time_col='DATE', target_col='AMOUNT', freq='MS')
nixtla_client.plot(df, timegpt_fcst_df, time_col='DATE', target_col='AMOUNT')

train_size = int(len(df) * 0.8)
test = df[train_size:]
input_seq = df[:train_size]




"""Evaluation"""

# Load your time series data (replace with your own)
df = pd.read_csv('/content/Updated_date_EU.csv')
df

df.drop(columns=['PROD_ID','PERIOD_DATE','Unnamed: 0'], inplace=True)
df.rename(columns={'UPDATED_DATE': 'DATE'}, inplace=True)
data = df.query("MARKET == 8223 and ACCOUNT_ID == 35315 and CHANNEL_ID == 9813	and MPG_ID == 380360")
data.shape

df = data.set_index('DATE')
df = df.sort_index()
df.head()

df = df[['AMOUNT']]

def prepare_data(df, window_size):
    # Create rolling windows
    X, y = [], []
    for i in range(len(df) - window_size):
        X.append(df.iloc[i:i + window_size].values)
        y.append(df.iloc[i + window_size].values)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# Assume the data was scaled using MinMaxScaler
scaler = MinMaxScaler()

# Fit the scaler on the original data
scaler.fit(df[['AMOUNT']])

# Transform the data
df['AMOUNT_SCALED'] = scaler.transform(df[['AMOUNT']])

# Use the scaled data for preparing training data
window_size = 2  # Adjust as needed
X, y = prepare_data(df[['AMOUNT_SCALED']], window_size)

class TimesNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(TimesNet, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out[:, -1, :])  # Use the last time step's output
        return output

# Instantiate your TimesNet model
input_size = X.shape[2]  # Adjust based on your features
hidden_size = 128  # Increased hidden size
output_size = y.shape[1]  # Same as input size for forecasting
num_layers = 2  # Number of LSTM layers
model = TimesNet(input_size, hidden_size, output_size, num_layers)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train your model using your prepared data (X, y)
num_epochs = 200  # Increased number of epochs
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    # Forward pass
    outputs = model(X)
    loss = criterion(outputs, y)

    # Backward pass and optimization
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Save the trained model
torch.save(model.state_dict(), 'timesnet_model.pth')

# Define the prediction function with inverse scaling
def predict_future(model, input_data, future_steps, scaler):
    predictions = []
    current_input = input_data

    model.eval()
    with torch.no_grad():
        for _ in range(future_steps):
            future_pred = model(current_input)
            predictions.append(future_pred.squeeze().item())

            # Prepare the next input by appending the predicted value and removing the oldest value
            next_input = torch.cat((current_input[:, 1:, :], future_pred.unsqueeze(1)), dim=1)
            current_input = next_input

    # Inverse transform the predictions to original scale
    predictions_original_scale = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
    return predictions_original_scale

# Load the trained model
model.load_state_dict(torch.load('timesnet_model.pth'))

# Prepare input data for forecasting (e.g., last window_size data points)
input_data = X[-1].unsqueeze(0)  # Add a batch dimension to make it (1, sequence_length, input_size)

# Define the number of future steps to predict
future_steps = 24  # Adjust as needed

# Predict future values
future_predictions = predict_future(model, input_data, future_steps, scaler)
print(future_predictions)

import matplotlib.pyplot as plt

# Generate future dates for plotting
future_dates = pd.date_range(df.index[-1], periods=future_steps + 1, freq='MS')[1:]
future_dates

future_dates = future_dates.strftime('%Y-%m-%d')
print(future_dates)

df.index

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(df.index, df['AMOUNT'], label='Original Data', color='blue')
plt.plot(future_dates, future_predictions, label='Forecasted Data', marker='o', color='green')
plt.xlabel('Date')
plt.ylabel('Amount')
plt.xticks(rotation='vertical')
plt.title('Original and Forecasted Data')
plt.legend()
plt.tight_layout()
plt.show()



''' preparation'''


def prepare_data(df, window_size):
    # Create rolling windows
    X, y = [], []
    for i in range(len(df) - window_size):
        X.append(df.iloc[i:i + window_size].values)
        y.append(df.iloc[i + window_size].values)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

window_size = 8  # Adjust as needed
X, y = prepare_data(df[['AMOUNT']], window_size)  # Ensure df[['AMOUNT']] is used to keep 2D structure

class TimesNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TimesNet, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out[:, -1, :])  # Use the last time step's output
        return output

# Instantiate your TimesNet model
input_size = X.shape[2]  # Adjust based on your features
hidden_size = 64
output_size = y.shape[1]  # Same as input size for forecasting
model = TimesNet(input_size, hidden_size, output_size)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train your model using your prepared data (X, y)
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    # Forward pass
    outputs = model(X)
    loss = criterion(outputs, y)

    # Backward pass and optimization
    loss.backward()
    optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Save the trained model
torch.save(model.state_dict(), 'timesnet_model.pth')

import torch

def predict_future(model, input_data):
    with torch.no_grad():
        model.eval()
        future_pred = model(input_data)
    return future_pred

# Load the trained model
model.load_state_dict(torch.load('timesnet_model.pth'))

# Prepare input data for forecasting (e.g., last window_size data points)
input_data = X[-1].unsqueeze(0)  # Add a batch dimension to make it (1, sequence_length, input_size)

# Predict future values
future_predictions = predict_future(model, input_data)
print(future_predictions)

X[-1].unsqueeze(0)

import torch

def predict_future(model, input_data, future_steps):
    predictions = []
    current_input = input_data

    model.eval()
    with torch.no_grad():
        for _ in range(future_steps):
            future_pred = model(current_input)
            predictions.append(future_pred.squeeze().item())

            # Prepare the next input by appending the predicted value and removing the oldest value
            next_input = torch.cat((current_input[:, 1:, :], future_pred.unsqueeze(1)), dim=1)
            current_input = next_input

    return predictions

# Load the trained model
model.load_state_dict(torch.load('timesnet_model.pth'))

# Prepare input data for forecasting (e.g., last window_size data points)
input_data = X[-1].unsqueeze(0)  # Add a batch dimension to make it (1, sequence_length, input_size)

# Define the number of future steps to predict
future_steps = 24  # Adjust as needed

# Predict future values
future_predictions = predict_future(model, input_data, future_steps)
print(future_predictions)

''' Fit'''

from sklearn.preprocessing import MinMaxScaler
import torch
import numpy as np

# Assume the data was scaled using MinMaxScaler
scaler = MinMaxScaler()

# Fit the scaler on the original data
scaler.fit(df[['AMOUNT']])

# Transform the data
df['AMOUNT_SCALED'] = scaler.transform(df[['AMOUNT']])

# Use the scaled data for preparing training data
window_size = 8
X, y = prepare_data(df[['AMOUNT_SCALED']], window_size)

# Define and train the model as before...

# Define the prediction function with inverse scaling
def predict_future(model, input_data, future_steps, scaler):
    predictions = []
    current_input = input_data

    model.eval()
    with torch.no_grad():
        for _ in range(future_steps):
            future_pred = model(current_input)
            predictions.append(future_pred.squeeze().item())

            # Prepare the next input by appending the predicted value and removing the oldest value
            next_input = torch.cat((current_input[:, 1:, :], future_pred.unsqueeze(1)), dim=1)
            current_input = next_input

    # Inverse transform the predictions to original scale
    predictions_original_scale = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
    return predictions_original_scale

# Load the trained model
model.load_state_dict(torch.load('timesnet_model.pth'))

# Prepare input data for forecasting (e.g., last window_size data points)
input_data = X[-1].unsqueeze(0)  # Add a batch dimension to make it (1, sequence_length, input_size)

# Define the number of future steps to predict
future_steps = 24  # Adjust as needed

# Predict future values
future_predictions = predict_future(model, input_data, future_steps, scaler)
print(future_predictions)



"""Final"""

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
import numpy as np

df

def prepare_data(df, window_size):
    # Create rolling windows
    X, y = [], []
    for i in range(len(df) - window_size):
        X.append(df.iloc[i:i + window_size].values)
        y.append(df.iloc[i + window_size].values)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# Assume the data was scaled using MinMaxScaler
scaler = MinMaxScaler()

# Fit the scaler on the original data
scaler.fit(df[['AMOUNT']])

# Transform the data
df['AMOUNT_SCALED'] = scaler.transform(df[['AMOUNT']])

# Use the scaled data for preparing training data
window_size = 8
X, y = prepare_data(df[['AMOUNT_SCALED']], window_size)

class TimesNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TimesNet, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out[:, -1, :])  # Use the last time step's output
        return output

# Instantiate your TimesNet model
input_size = X.shape[2]  # Adjust based on your features
hidden_size = 64
output_size = y.shape[1]  # Same as input size for forecasting
model = TimesNet(input_size, hidden_size, output_size)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train your model using your prepared data (X, y)
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    # Forward pass
    outputs = model(X)
    loss = criterion(outputs, y)

    # Backward pass and optimization
    loss.backward()
    optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Save the trained model
torch.save(model.state_dict(), 'timesnet_model.pth')

# Define the prediction function with inverse scaling
def predict_future(model, input_data, future_steps, scaler):
    predictions = []
    current_input = input_data

    model.eval()
    with torch.no_grad():
        for _ in range(future_steps):
            future_pred = model(current_input)
            predictions.append(future_pred.squeeze().item())

            # Prepare the next input by appending the predicted value and removing the oldest value
            next_input = torch.cat((current_input[:, 1:, :], future_pred.unsqueeze(1)), dim=1)
            current_input = next_input

    # Inverse transform the predictions to original scale
    predictions_original_scale = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
    return predictions_original_scale

# Load the trained model
model.load_state_dict(torch.load('timesnet_model.pth'))

# Prepare input data for forecasting (e.g., last window_size data points)
input_data = X[-1].unsqueeze(0)  # Add a batch dimension to make it (1, sequence_length, input_size)

# Define the number of future steps to predict
future_steps = 24  # Adjust as needed

# Predict future values
future_predictions = predict_future(model, input_data, future_steps, scaler)
print(future_predictions)

