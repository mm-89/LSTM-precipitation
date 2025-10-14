import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

from LSTM_model import LSTM_classification, FocalLoss
from LSTMJOIN_model import LSTM_MultiTask

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

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

# se classi sbilanciate
neg = df[df['cumulative precipitation'] == 0].shape[0]
pos = df[df['cumulative precipitation'] != 0].shape[0]
weight = neg / pos

df['precipitation'] = (df['cumulative precipitation'] != 0).astype(int)

df = df.drop(columns='cumulative precipitation')

df = df.dropna()

torch.manual_seed(121)

print(f"Cuda avaibility: {torch.cuda.is_available()}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device: {device}")

seq_len = 10 # in ore / tra 10 e 12 cambia poco

# selezione feature e target
features = df.drop(columns=['reference_timestamp']).values
targets = df["precipitation"].values.reshape(-1,1)

# scaler
feature_scaler = MinMaxScaler()

features_scaled = feature_scaler.fit_transform(features)

# creazione sequenze
X = np.array([features_scaled[i:i+seq_len] for i in range(len(features_scaled)-seq_len)], dtype=np.float32)
y = np.array([targets[i+seq_len] for i in range(len(targets)-seq_len)], dtype=np.float32).reshape(-1,1)

X_ori = X
y_ori = y

# ---- PREPARAZIONE RESAMPLIG ----

idx_rain = np.where(y.flatten() == 1)[0]
idx_no_rain = np.where(y.flatten() == 0)[0]

# Numero di esempi di pioggia
n_rain = len(idx_rain)

# Resample casuale dagli esempi no-pioggia per avere 50-50
np.random.seed(42)
idx_no_rain_resampled = np.random.choice(idx_no_rain, size=n_rain, replace=False)

# Combina gli indici e mescola
idx_balanced = np.concatenate([idx_rain, idx_no_rain_resampled])
np.random.shuffle(idx_balanced)

# Sequenze e target binari bilanciati
X = X[idx_balanced]
y = y[idx_balanced]

# -----------------------------------------------------------
X_train_val, X_test, y_train_val, y_test = train_test_split(
     X, y, test_size=0.2, random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X_train_val, y_train_val, test_size=0.25, random_state=42
)

# -----------------------------------------------------------
X_orig_train_val, X_orig_test_val, y_orig_train_val, y_orig_test_val = train_test_split(
    X_ori, y_ori, test_size=0.2, random_state=42
)

X_orig_train, X_orig_test, y_orig_train, y_orig_test = train_test_split(
    X_orig_train_val, y_orig_train_val, test_size=0.25, random_state=42
)
# -----------------------------------------------------------

# conversione in tensori PyTorch CLASSIFICAZIONE
X_train_torch = torch.from_numpy(X_train).to(device)
X_test_torch = torch.from_numpy(X_test).to(device)

X_orig_train_torch = torch.from_numpy(X_orig_train).to(device)
X_orig_test_torch = torch.from_numpy(X_orig_test).to(device)

y_train_torch = torch.from_numpy(y_train).to(device)
y_test_torch = torch.from_numpy(y_test).to(device)

y_orig_train_torch = torch.from_numpy(y_orig_train).to(device)
y_orig_test_torch = torch.from_numpy(y_orig_test).to(device)

# BCE vuole float...
y_train_torch = y_train_torch.float()
y_test_torch  = y_test_torch.float()
y_orig_train_torch = y_orig_train_torch.float()
y_orig_test_torch = y_orig_test_torch.float()

# -----------------------------------------------------------------------

model = LSTM_classification(in_feat=9, hidden_size=256, num_layers=3, out_feat=1).to(device)

#pos_weight = torch.tensor()
criterion = nn.BCEWithLogitsLoss()
#criterion = FocalLoss()

optimizer = optim.Adam(model.parameters(), lr=0.00005)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.65)

epochs = 1000

thres = 0.5

# backup
# optimizer = optim.Adam(model.parameters(), lr=0.00005, weight_decay=3e-5)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.8)
# epochs = 10000
# thres = 0.7

loss_tot = []
val_loss_tot = []
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    out = model(X_train_torch)
    loss = criterion(out, y_train_torch)
    loss.backward()

    optimizer.step()

    loss_tot.append(loss.item())

    model.eval()
    with torch.no_grad():

        val_out = model(X_orig_test_torch) # uso tutto il dataset originale
        val_loss = criterion(val_out, y_orig_test_torch).item()
        val_loss_tot.append(val_loss)

        train_preds = torch.sigmoid(out).squeeze()
        train_acc = ((train_preds > thres).float() == y_train_torch.squeeze()).float().mean().item()
        
        val_preds = torch.sigmoid(val_out).squeeze()
        val_acc = ((val_preds > thres).float() == y_orig_test_torch.squeeze()).float().mean().item()

        probs = torch.sigmoid(val_out).squeeze()         # (N,)
        preds_label = (probs > thres).int()
        
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss train: {loss.item():.4f}, loss test: {val_loss}; Train Acc: {train_acc:.4f}. Val Acc: {val_acc:.4f}")

    #scheduler.step()
plt.plot(np.arange(len(loss_tot)), loss_tot, label='train')
plt.plot(np.arange(len(val_loss_tot)), val_loss_tot, label='test')
plt.legend()
plt.show()

# --- Calcolo metriche ---
preds_np  = preds_label.cpu().numpy()
probs_np  = probs.cpu().numpy()

precision = precision_score(y_orig_test, preds_np)
recall    = recall_score(y_orig_test, preds_np)
f1        = f1_score(y_orig_test, preds_np)
roc_auc   = roc_auc_score(y_orig_test, preds_np)
cm        = confusion_matrix(y_orig_test, preds_np)

print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")
print("Confusion Matrix:")
print(cm)

n = 1500
y_true = df['precipitation'] # solo per vedere valori realistici
y_rain = (y_true > 0)
y_pred = val_preds.detach().cpu().numpy().flatten()
y_pred = (y_pred > thres)

plt.figure(figsize=(12,5))
plt.bar(np.arange(n), y_rain[:n], color='skyblue', alpha=0.5, label="TRUE")
plt.bar(np.arange(n), y_pred[:n], color='red', alpha=0.2, label="PRED")
plt.xlabel("Campioni")
plt.ylabel("Residuo")
plt.legend()
plt.show()