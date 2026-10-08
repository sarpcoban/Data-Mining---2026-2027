import pandas as pd
import numpy as np

# Örnek zaman serisi
df = pd.DataFrame({
    "Saat": ["08:00", "09:00", "10:00", "11:00", "12:00", "13:00"],
    "Sicaklik": [12, 14, np.nan, np.nan, 20, 22]
})

# Forward Fill
df["Forward_Fill"] = df["Sicaklik"].ffill()

# Backward Fill
df["Backward_Fill"] = df["Sicaklik"].bfill()

# Doğrusal interpolasyon
df["Linear_Interpolation"] = df["Sicaklik"].interpolate(method="linear")

print(df)