import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Örnek veri
data = {
    "Boy": [150, 155, 160, 165, 170, 175, 180, 185, 150],
    "Kilo": [50, 52, 55, 60, 68, 72, 80, 85, 150]
}

df = pd.DataFrame(data)

print(df)

# K-Means modeli
# Verileri 2 kümeye ayırıyoruz
kmeans = KMeans(n_clusters=2, random_state=42)

# Modeli eğit ve küme numaralarını al
df["Kume"] = kmeans.fit_predict(df[["Boy", "Kilo"]])

print("\nKümelenmiş veri:")
print(df)

# Küme merkezleri
print("\nKüme merkezleri:")
print(kmeans.cluster_centers_)

# Grafik
plt.scatter(
    df["Boy"],
    df["Kilo"],
    c=df["Kume"]
)

# Küme merkezlerini göster
plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    marker="X",
    s=200
)

plt.xlabel("Boy")
plt.ylabel("Kilo")
plt.title("K-Means Kümeleme")
plt.show()