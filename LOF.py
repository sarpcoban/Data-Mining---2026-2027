import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

# Örnek veri
data = pd.DataFrame({
    "X1": [10, 11, 10, 12, 11, 10, 12, 50, 11, 10],
    "X2": [20, 21, 19, 20, 22, 21, 19, 80, 20, 21]
})

# Veriyi standartlaştırma
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data)

# LOF modeli
lof = LocalOutlierFactor(
    n_neighbors=3,
    contamination=0.1
)

# Tahmin
data["LOF_Sonuc"] = lof.fit_predict(X_scaled)

# LOF skorları
data["LOF_Skoru"] = lof.negative_outlier_factor_

# -1 = aykırı değer
#  1 = normal değer
print(data)

# Sadece aykırı değerler
aykiri = data[data["LOF_Sonuc"] == -1]

print("\nAykırı değerler:")
print(aykiri)

# Görselleştirme
normal = data[data["LOF_Sonuc"] == 1]
aykiri = data[data["LOF_Sonuc"] == -1]

plt.scatter(normal["X1"], normal["X2"], label="Normal")
plt.scatter(
    aykiri["X1"],
    aykiri["X2"],
    marker="x",
    s=100,
    label="Aykırı"
)

plt.xlabel("X1")
plt.ylabel("X2")
plt.title("LOF ile Aykırı Değer Tespiti")
plt.legend()
plt.show()