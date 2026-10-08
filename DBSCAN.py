import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# Örnek veri
df = pd.DataFrame({
    "X": [1, 1.2, 1.1, 1.3, 5, 5.1, 5.2, 5.3, 10, 12],
    "Y": [1, 1.1, 1.3, 1.2, 5, 5.2, 5.1, 5.3, 1, 10]
})

# 1. Veriyi standartlaştır
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[["X", "Y"]])

# 2. DBSCAN modelini oluştur
dbscan = DBSCAN(
    eps=0.6,       # Komşuluk yarıçapı
    min_samples=3  # Bir bölgenin yoğun sayılması için minimum nokta
)

# 3. Küme etiketlerini belirle
df["Kume"] = dbscan.fit_predict(X_scaled)

# 4. -1 olan noktalar aykırı değer
aykiri_degerler = df[df["Kume"] == -1]

print("Tüm veriler:")
print(df)

print("\nAykırı değerler:")
print(aykiri_degerler)

# 5. Görselleştirme
normal = df[df["Kume"] != -1]
aykiri = df[df["Kume"] == -1]

plt.scatter(
    normal["X"],
    normal["Y"],
    label="Normal Veriler"
)

plt.scatter(
    aykiri["X"],
    aykiri["Y"],
    marker="x",
    s=100,
    label="Aykırı Değer"
)

plt.xlabel("X")
plt.ylabel("Y")
plt.title("DBSCAN ile Aykırı Değer Belirleme")
plt.legend()
plt.show()