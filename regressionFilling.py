import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Örnek veri seti
df = pd.DataFrame({
    "Yas": [25, 30, 35, 40, 45],
    "Deneyim": [2, 5, 8, 12, 15],
    "Maas": [30000, 45000, np.nan, 70000, np.nan]
})

print("Eksik veriler doldurulmadan önce:")
print(df)

# Maaş değeri eksik olmayan gözlemler
train = df[df["Maas"].notna()]

# Maaş değeri eksik olan gözlemler
missing = df[df["Maas"].isna()]

# Bağımsız değişkenler ve hedef değişken
X_train = train[["Yas", "Deneyim"]]
y_train = train["Maas"]

# Regresyon modeli
model = LinearRegression()

# Modeli eğit
model.fit(X_train, y_train)

# Eksik maaşları tahmin et
X_missing = missing[["Yas", "Deneyim"]]
predictions = model.predict(X_missing)

# Tahmin edilen değerlerle eksikleri doldur
df.loc[df["Maas"].isna(), "Maas"] = predictions

print("\nEksik veriler doldurulduktan sonra:")
print(df)