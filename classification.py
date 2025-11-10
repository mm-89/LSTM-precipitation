import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

from LSTM_model import LSTM_classification, FocalLoss
from LSTMJOIN_model import LSTM_MultiTask

from sklearn.utils import resample
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

#df1 = pd.read_csv("ogd-smn_gve_h_historical_2000-2009.csv", sep=";")
df = pd.read_csv("ogd-smn_gve_h_historical_2010-2019.csv", sep=";")

#df = pd.concat([df1, df2], axis=0)

conversion = {
    'tre005h0' : 'T', # temperatura ground
    'ure200h0' : '2mRH',
    'pva200h0' : '2mVP',
    'tde200h0' : 'DP',
    'prestah0' : 'QFE',
    #'dkl010h0' : 'wind speed', 
    #'fkl010h0' : 'wind direction',
    'rre150h0' : 'precipitation',
    'gre000h0' : 'irradiance',
    #'sre000h0' : 'sunshine'
 }

# conversion = {
#     'tre005s0' : 'T', # temperatura ground
#     'ure200s0' : '2mRH',
#     'pva200s0' : '2mVP',
#     'tde200s0' : 'DP',
#     'prestas0' : 'QFE',
#     #'dkl010z0' : 'wind speed', 
#     #'fkl010z0' : 'wind direction',
#     'rre150z0' : 'precipitation',
#     'gre000z0' : 'irradiance',
#     'sre000z0' : 'sunshine'
#  }

fets_sel = list(conversion.keys())
fets_sel = ["reference_timestamp"] + fets_sel

df = df[fets_sel]
df = df.rename(columns=conversion)

# df['T_3h'] = df['T'].rolling(3).mean()
df['precipitation_3h'] = df['precipitation'].rolling(6).min()

# df['dT'] = df['T'].diff(1)
df['dP'] = df['QFE'].diff(1)
df['dU'] = df['2mRH'].diff(1)
# df['dTd'] = df['dP'].diff(1)

#df['precipitation'] += np.random.normal(0, 1e-3, size=(df['precipitation'].shape))

df['T'].plot()
plt.show()

corr = df.corr(numeric_only=True)

# Crea la heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True, cbar_kws={"shrink": 0.8})
plt.title("Matrice di correlazione - df_graeua")
plt.tight_layout()
plt.show()

# se classi sbilanciate
neg = df[df['precipitation'] == 0].shape[0]
pos = df[df['precipitation'] != 0].shape[0]
weight = neg / pos

df = df.dropna()

torch.manual_seed(121)

print(f"Cuda avaibility: {torch.cuda.is_available()}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device: {device}")

seq_len = 5 # in ore / tra 10 e 12 cambia poco

df['precipitation'] = (df['precipitation'] > 0).astype(int)

# selezione feature e target
X = df.drop(columns=['reference_timestamp']).values
y = df["precipitation"].values.reshape(-1,1)

# -------------------------------------------------

# Dividiamo prima in train e test (80% train, 20% test) per semplicità
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

X_scaler = StandardScaler()
X_train_scaled = X_scaler.fit_transform(X_train)
X_test_scaled = X_scaler.transform(X_test)

# creazione sequenze
def create_sequences(X, y, seq_len):
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i:i+seq_len])
        ys.append(y[i+seq_len])
    return np.array(Xs), np.array(ys)

X_train_t, y_train_t = create_sequences(X_train_scaled, y_train, seq_len)
X_test_t, y_test_t = create_sequences(X_test_scaled, y_test, seq_len)

X_train_torch = torch.from_numpy(X_train_t).float().to(device)
y_train_torch = torch.from_numpy(y_train_t).float().to(device)

X_test_torch = torch.from_numpy(X_test_t).float().to(device)
y_test_torch = torch.from_numpy(y_test_t).float().to(device)

# -----------------------------------------------------------------------
save_model = False
charge_model = False

model = LSTM_classification(in_feat=10, hidden_size=32, num_layers=1, out_feat=1).to(device)

weight *= 1
pos_weight = torch.tensor([weight], device=device)
#criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
criterion = FocalLoss()

optimizer = optim.Adam(model.parameters(), lr=0.01)
#scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

epochs = 100

thres = 0.5

if charge_model:
    checkpoint = torch.load('checkpoint.pth', weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    loss = checkpoint['loss']
    scheduler = checkpoint['scheduler']
    print(f"Model charged, starts at epoch: {start_epoch}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load('checkpoint.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
else:
    start_epoch = 0

loss_tot = []
val_loss_tot = []
for epoch in range(start_epoch, epochs):
    model.train()
    optimizer.zero_grad()
    out = model(X_train_torch)
    loss = criterion(out, y_train_torch)
    loss.backward()

    optimizer.step()

    loss_tot.append(loss.item())

    model.eval()
    with torch.no_grad():

        val_out = model(X_test_torch)
        val_loss = criterion(val_out, y_test_torch).item()
        val_loss_tot.append(val_loss)

        train_probs = torch.sigmoid(out).squeeze(-1)
        train_preds = (train_probs > thres).int()
        train_acc = (train_preds == y_train_torch.squeeze(-1)).float().mean().item()

        val_probs = torch.sigmoid(val_out).squeeze(-1)
        val_preds = (val_probs > thres).int()
        val_acc = (val_preds == y_test_torch.squeeze(-1)).float().mean().item()

    #if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss train: {loss.item():.4f}, loss test: {val_loss}; Train Acc: {train_acc:.4f}. Val Acc: {val_acc:.4f}")

    #scheduler.step()
    
plt.plot(np.arange(len(loss_tot)), loss_tot, label='train')
plt.plot(np.arange(len(val_loss_tot)), val_loss_tot, label='test')
plt.legend()
plt.show()

# --- Calcolo metriche ---
preds_np = val_preds.cpu().numpy()  # labels binarie
probs_np = val_probs.cpu().numpy()  # probabilità continue

# metriche
precision = precision_score(y_test_t.ravel(), preds_np)
recall    = recall_score(y_test_t.ravel(), preds_np)
f1        = f1_score(y_test_t.ravel(), preds_np)
roc_auc   = roc_auc_score(y_test_t.ravel(), probs_np)
cm        = confusion_matrix(y_test_t.ravel(), preds_np)

print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")
print("Confusion Matrix:")
print(cm)

n = 5000
y_true = df['precipitation']
y_true = y_true[int(len(X) * 0.8):]
y_pred = val_preds.detach().cpu().numpy().flatten()

plt.figure(figsize=(12,5))
plt.bar(np.arange(n), y_true[:n], color='skyblue', alpha=0.5, label="TRUE")
plt.bar(np.arange(n), y_pred[:n], color='red', alpha=0.2, label="PRED")
plt.xlabel("Campioni")
plt.ylabel("Residuo")
plt.legend()
plt.show()

import numpy as np
from sklearn.metrics import precision_recall_curve, f1_score

# y_true = etichette reali (0/1)
# y_scores = probabilità previste dal modello (output di predict_proba[:, 1] o decision_function)
# esempio:
# y_true = np.array([...])
# y_scores = np.array([...])

# Calcola precision, recall e soglie
precision, recall, thresholds = precision_recall_curve(y_test_t.ravel(), probs_np)

# Calcola F1 per ogni soglia
f1 = 2 * (precision * recall) / (precision + recall + 1e-10)

# Trova la soglia che massimizza l'F1-score
idx = np.argmax(f1)
best_threshold = thresholds[idx]
best_f1 = f1[idx]

print(f"Soglia ottimale: {best_threshold:.3f}")
print(f"F1 ottimale: {best_f1:.3f}")
print(f"Precisione a soglia ottimale: {precision[idx]:.3f}")
print(f"Recall a soglia ottimale: {recall[idx]:.3f}")

# SALVIAMO il modello
if save_model:
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'scheduler' : scheduler,
    }, 'checkpoint.pth')