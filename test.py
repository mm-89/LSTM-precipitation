import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import QuantileTransformer

df = pd.read_csv("ogd-smn_gve_h_historical_2010-2019.csv", sep=";")
print(df.columns)

#df = df.dropna()

#df['rre150h0'] = np.log1p(df['rre150h0'])

df.hist('rre150h0', bins=20)
plt.ylim(0, 10)
plt.show()

rain = np.array(df['rre150h0'].to_list())

mask = (rain > 0) & (rain < 2)
rain = rain[mask]
rain += np.random.normal(0, 1e-5, size=(rain.size))

rain = rain.reshape(-1, 1)

Q1 = np.percentile(rain, 25)
Q3 = np.percentile(rain, 75)
IQR = Q3 - Q1

# Outlier (limite superiore)
upper_limit = Q3 + 10*IQR

print(upper_limit)

qt = QuantileTransformer(output_distribution='normal', n_quantiles=1000)
rain_transformed = qt.fit_transform(rain)

plt.figure(figsize=(10,5))
plt.hist(rain_transformed, bins=20, color='skyblue', edgecolor='black')
plt.title("Distribuzione della pioggia dopo QuantileTransformer")
plt.xlabel("Valore trasformato")
plt.ylabel("Frequenza")
plt.show()

Q1 = np.percentile(rain_transformed, 25)
Q3 = np.percentile(rain_transformed, 75)
IQR = Q3 - Q1

# Outlier (limite superiore)
upper_limit = Q3 + 10*IQR

print(upper_limit)