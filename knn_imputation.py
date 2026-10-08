import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

# Örnek veri seti
df = pd.DataFrame({
    "Yas": [25, 30, np.nan, 40, 35],
    "Maas": [30000, 40000, 35000, np.nan, 50000],
    "Deneyim": [2, 5, 4, 10, np.nan]
})

print("Eksik veriler:")
print(df)

# KNN Imputer oluşturma
# n_neighbors=2 -> en yakın 2 komşu dikkate alınır
imputer = KNNImputer(n_neighbors=2)

# Eksik değerleri doldurma
doldurulmus_veri = imputer.fit_transform(df)

# Tekrar DataFrame'e çevirme
df_doldurulmus = pd.DataFrame(
    doldurulmus_veri,
    columns=df.columns
)

print("\nKNN imputasyonu sonrası:")
print(df_doldurulmus)