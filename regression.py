import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer

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
# df['wind NS component'] = df['wind speed (ms) hourly mean'] * np.degrees(np.cos(np.radians(df['wind direction hourly mean'])))
# df['wind WE component'] = df['wind speed (ms) hourly mean'] * np.degrees(np.sin(np.radians(df['wind direction hourly mean'])))

# df = df.drop(columns=['wind speed (ms) hourly mean', 'wind direction hourly mean'])
# df['average temperature'] = df[['2m air T hourly mean', 'air T ground hourly mean']].mean(axis=1)
# df = df.drop(columns=['2m air T hourly mean', 'air T ground hourly mean'])

# con quantitle transfromation non c'è bisogno di log transform
# df['cumulative precipitation'] = np.log1p(df['cumulative precipitation'])

print(f"Max value for prec: {df['cumulative precipitation'].max()}, Min value for prec: {df['cumulative precipitation'].min()}")
print(f"Mean: {df['cumulative precipitation'].mean()}, std: {df['cumulative precipitation'].std()}")

df = df.dropna()

torch.manual_seed(121)

print(f"Cuda avaibility: {torch.cuda.is_available()}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device: {device}")

seq_len = 6 # in ore / tra 10 e 12 cambia poco

# aggiungo rumore gaussiano alla precipitazione ----------------
df['cumulative precipitation'] += np.random.normal(0, 1e-3, size=(df['cumulative precipitation'].shape))
# --------------------------------------------------------------

# prendo il subset dove precipitation è > 0
df = df[df['cumulative precipitation'] > 0]

# selezione feature e target
X = df.drop(columns=['reference_timestamp']).values
y = df["cumulative precipitation"].values.reshape(-1,1)

# Dividiamo prima in train e test (80% train, 20% test) per semplicità
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

X_scaler = MinMaxScaler()
y_scaler = QuantileTransformer(output_distribution='normal', n_quantiles=1000)

X_train_scaled = X_scaler.fit_transform(X_train)
y_train_scaled = y_scaler.fit_transform(y_train)

# trasformo i rispettivi TEST set
X_test_scaled = X_scaler.transform(X_test)
y_test_scaled = y_scaler.transform(y_test)

# print(y_test_scaled.min(), y_test_scaled.max())
# print(y_train_scaled.min(), y_train_scaled.max())

def create_sequences(X, y, seq_len):
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i:i+seq_len])
        ys.append(y[i+seq_len])
    return np.array(Xs), np.array(ys)

X_train_t, y_train_t = create_sequences(X_train_scaled, y_train_scaled, seq_len)
X_test_t, y_test_t = create_sequences(X_test_scaled, y_test_scaled, seq_len)

X_train_torch = torch.from_numpy(X_train_t).float().to(device)
y_train_torch = torch.from_numpy(y_train_t).float().to(device)

X_test_torch = torch.from_numpy(X_test_t).float().to(device)
y_test_torch = torch.from_numpy(y_test_t).float().to(device)

# check visivo gaussianità --------------------------------------
plt.figure(figsize=(10,5))
plt.hist(y_train_scaled, bins=100, color='skyblue', edgecolor='black')
plt.title("Distribuzione della pioggia dopo QuantileTransformer")
plt.xlabel("Valore trasformato")
plt.ylabel("Frequenza")
plt.show()
# # ----------------------------------------------------------------

# --------------------------------------------------------------------

model = LSTM_regression(in_feat=9, hidden_size=64, num_layers=3, out_feat=1).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.99)

epochs = 50

train_loss_tot, test_loss_tot = [], []
train_rmse_tot, test_rmse_tot = [], []

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    train_out = model(X_train_torch)
    loss = criterion(train_out, y_train_torch)
    loss.backward()

    optimizer.step()

    train_loss_tot.append(loss.item())

    model.eval()
    with torch.no_grad():
        test_out = model(X_test_torch)
        test_loss = criterion(test_out, y_test_torch)
        test_loss_tot.append(test_loss.item())

        # Converti in NumPy
        y_pred_train = train_out.cpu().numpy()
        y_true_train = y_train_torch.cpu().numpy()
        y_pred_test = test_out.cpu().numpy()
        y_true_test = y_test_torch.cpu().numpy()

        # Inverse transform
        y_pred_train = y_scaler.inverse_transform(y_pred_train)
        y_true_train = y_scaler.inverse_transform(y_true_train)
        y_pred_test = y_scaler.inverse_transform(y_pred_test)
        y_true_test = y_scaler.inverse_transform(y_true_test)

        # Calcolo RMSE
        rmse_train = np.sqrt(np.mean((y_pred_train - y_true_train)**2))
        rmse_test = np.sqrt(np.mean((y_pred_test - y_true_test)**2))
        train_rmse_tot.append(rmse_train)
        test_rmse_tot.append(rmse_test)
        
    #if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss train: {loss.item():.4f}, Loss test: {test_loss:.4f}. RMSE train: {rmse_train:.4f}, RMSE test: {rmse_test:.4f}")

        # if last_loss <= val_loss: break
        # else: last_loss = val_loss

    #scheduler.step()

n = 2000

y_pred = y_scaler.inverse_transform(test_out.detach().cpu().numpy()).flatten()
y_true = y_scaler.inverse_transform(y_test_torch.detach().cpu().numpy()).flatten()

print(y_true[:n])

plt.figure(figsize=(12,5))
plt.bar(np.arange(n), y_true[:n], color='skyblue', alpha=0.5, label="TRUE")
plt.bar(np.arange(n), y_pred[:n], color='red', alpha=0.5, label="PRED")
plt.xlabel("Campioni")
plt.ylabel("Residuo")
plt.legend()
plt.show()

plt.figure(figsize=(12,5))
plt.bar(np.arange(n),
        y_scaler.transform(y_true_test[:n].reshape(-1, 1)).ravel(),
        color='skyblue', alpha=0.5, label="TRUE")
plt.bar(np.arange(n),
        y_scaler.transform(y_pred_test[:n].reshape(-1, 1)).ravel(),
        color='red', alpha=0.5, label="PRED")
plt.xlabel("Campioni")
plt.ylabel("Residuo (scalato)")
plt.legend()
plt.show()


plt.plot(np.arange(len(train_loss_tot)), train_loss_tot, label='LOSS train')
plt.plot(np.arange(len(test_loss_tot)), test_loss_tot, label='LOSS test')
plt.legend()
plt.show()
plt.plot(np.arange(len(train_rmse_tot)), train_rmse_tot, label='RMSE train')
plt.plot(np.arange(len(test_rmse_tot)), test_rmse_tot, label='RMSE test')
plt.legend()
plt.show()