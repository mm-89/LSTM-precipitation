import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

from LSTM_model import LSTM_regression
from LSTMJOIN_model import LSTM_MultiTask

import torch
import torch.nn as nn
import torch.optim as optim

df1 = pd.read_csv("ogd-smn_gve_h_historical_2000-2009.csv", sep=";")
df2 = pd.read_csv("ogd-smn_gve_h_historical_2010-2019.csv", sep=";")

df = pd.concat([df1, df2], axis=0)

conversion = {
    #'tre200h0' : '2m air T hourly mean',
    'tre005h0' : 'air T ground hourly mean',
    'ure200h0' : '2m rel hum hourly mean',
    'tde200h0' : 'dew point',
    'prestah0' : 'atmospheric pressure (QFE)',
    'dkl010h0' : 'wind direction hourly mean',
    'fkl010h0' : 'wind speed (ms) hourly mean',
    'rre150h0' : 'cumulative precipitation',
    'gre000h0' : 'global irradiance',
    'sre000h0' : 'total sunshine'
 }

fets_sel = list(conversion.keys())
fets_sel = ["reference_timestamp"] + fets_sel

df = df[fets_sel]
df = df.rename(columns=conversion)

df["reference_timestamp"] = pd.to_datetime(df["reference_timestamp"], format="%d.%m.%Y %H:%M")

# wind conversion
df['wind NS component'] = df['wind speed (ms) hourly mean'] * np.degrees(np.cos(np.radians(df['wind direction hourly mean'])))
df['wind WE component'] = df['wind speed (ms) hourly mean'] * np.degrees(np.sin(np.radians(df['wind direction hourly mean'])))

df = df.drop(columns=['wind speed (ms) hourly mean', 'wind direction hourly mean'])
# df['average temperature'] = df[['2m air T hourly mean', 'air T ground hourly mean']].mean(axis=1)
# df = df.drop(columns=['2m air T hourly mean', 'air T ground hourly mean'])

df['cumulative precipitation'] = np.log1p(df['cumulative precipitation'])

df.hist('cumulative precipitation', bins=20)
plt.ylim(0, 20)
plt.show()

# --- LOG TRANSF ---

print(f"Max value for prec: {df['cumulative precipitation'].max()}, Min value for prec: {df['cumulative precipitation'].min()}")
print(f"Mean: {df['cumulative precipitation'].mean()}, std: {df['cumulative precipitation'].std()}")


df = df.dropna()

torch.manual_seed(121)

print(f"Cuda avaibility: {torch.cuda.is_available()}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device: {device}")

seq_len = 6 # in ore / tra 10 e 12 cambia poco

# selezione feature e target
features = df.drop(columns=['reference_timestamp']).values
targets = df["cumulative precipitation"].values.reshape(-1,1)

# scaler
feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

features_scaled = feature_scaler.fit_transform(features)
targets_scaled = target_scaler.fit_transform(targets)

# target binario non scalato
targets_bin = (targets > 0.1)
y_bin = np.array([targets_bin[i+seq_len] for i in range(len(targets_bin)-seq_len)], dtype=np.float32).reshape(-1,1)

# creazione sequenze
X = np.array([features_scaled[i:i+seq_len] for rain, i in zip(y_bin, range(len(features_scaled)-seq_len)) if rain], dtype=np.float32)

y = np.array([targets_scaled[i+seq_len] for rain, i in zip(y_bin, range(len(targets_scaled)-seq_len)) if rain], dtype=np.float32).reshape(-1,1)

# Dividiamo prima in train+val e test (80% train+val, 20% test)
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.25, random_state=42
)

# conversione in tensori PyTorch
X_train_torch = torch.from_numpy(X_train).to(device)
y_train_torch = torch.from_numpy(y_train).to(device)

X_test_torch = torch.from_numpy(X_test).to(device)
y_test_torch = torch.from_numpy(y_test).to(device)

# --------------------------------------------------------------------

model = LSTM_regression(in_feat=9, hidden_size=256, num_layers=3, out_feat=1).to(device)

#criterion = nn.MSELoss()
criterion = nn.SmoothL1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.005)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.9)

epochs = 1000

train_loss_tot = []
test_loss_tot = []

train_rmse_tot = []
test_rmse_tot = []

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    out = model(X_train_torch)
    loss = criterion(out, y_train_torch)
    loss.backward()

    optimizer.step()

    train_loss_tot.append(loss.item())

    model.eval()
    with torch.no_grad():

        val_out = model(X_test_torch)
        val_loss = criterion(val_out, y_test_torch).item()
        test_loss_tot.append(val_loss)

        y_pred_test = val_out.cpu().numpy()
        y_pred_test = target_scaler.inverse_transform(y_pred_test)
        y_true_test = target_scaler.inverse_transform(y_test)

        y_pred_train = out.cpu().numpy()
        y_pred_train = target_scaler.inverse_transform(y_pred_train)
        y_true_train = target_scaler.inverse_transform(y_train)

        mse_test = np.mean((np.expm1(y_pred_test) - np.expm1(y_true_test))** 2)
        rmse_test = mse_test ** 0.5
        test_rmse_tot.append(rmse_test)

        mse_train =  np.mean((np.expm1(y_pred_train) - np.expm1(y_true_train))** 2)
        rmse_train = mse_train ** 0.5
        train_rmse_tot.append(rmse_train)
        
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss train: {loss.item():.4f}, Loss test: {val_loss:.4f}. RMSE train: {rmse_train:.4f}, RMSE test: {rmse_test:.4f}")

        # if last_loss <= val_loss: break
        # else: last_loss = val_loss

    scheduler.step()

n = 2000
residuals = (y_test_torch - val_out).detach().cpu().numpy()

residuals_np = residuals[:n].flatten()

y_true = y_test_torch.detach().cpu().numpy().flatten()
y_pred = val_out.detach().cpu().numpy().flatten()

plt.figure(figsize=(12,5))
plt.bar(np.arange(n), y_true[:n], color='skyblue', alpha=0.5, label="TRUE")
plt.bar(np.arange(n), y_pred[:n], color='red', alpha=0.5, label="PRED")
plt.xlabel("Campioni")
plt.ylabel("Residuo")
plt.legend()
plt.title(f"Residui (primi {len(residuals_np)} campioni)")
plt.show()

plt.plot(np.arange(len(train_loss_tot)), train_loss_tot, label='LOSS train')
plt.plot(np.arange(len(test_loss_tot)), test_loss_tot, label='LOSS test')
# plt.plot(np.arange(len(train_rmse_tot)), train_rmse_tot, label='RMSE train')
# plt.plot(np.arange(len(test_rmse_tot)), test_rmse_tot, label='RMSE test')
plt.legend()
plt.show()