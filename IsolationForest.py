import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# Örnek veri
df = pd.DataFrame({
    "Maas": [30000, 32000, 31000, 29500, 30500, 31500, 30000, 29000, 32500, 31000],
    "Yas":  [25, 27, 26, 24, 28, 29, 28, 23, 30, 27]
})

# Isolation Forest modeli
model = IsolationForest(
    contamination=0.2,   # Verilerin yaklaşık %10'unun aykırı olduğu varsayılıyor
    random_state=42
)

# Modeli eğit ve tahmin yap
df["Aykiri"] = model.fit_predict(df[["Maas", "Yas"]])

# Isolation Forest:
#  1  -> Normal gözlem
# -1  -> Aykırı gözlem

print(df)

# Sadece aykırı değerleri göster
aykiri_degerler = df[df["Aykiri"] == -1]

print("\nAykırı Değerler:")
print(aykiri_degerler)

# Görselleştirme
normal = df[df["Aykiri"] == 1]
aykiri = df[df["Aykiri"] == -1]

plt.scatter(normal["Yas"], normal["Maas"], label="Normal")
plt.scatter(aykiri["Yas"], aykiri["Maas"], label="Aykırı")

plt.xlabel("Yaş")
plt.ylabel("Maaş")
plt.title("Isolation Forest ile Aykırı Değer Tespiti")
plt.legend()
plt.show()