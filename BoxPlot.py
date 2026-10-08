import pandas as pd
import matplotlib.pyplot as plt

# Örnek veri
df = pd.DataFrame({
    "Maas": [20, 22, 21, 23, 24, 25, 26, 22, 21, 50, 60]
})

# Boxplot
plt.boxplot(df["Maas"])
plt.ylabel("Maaş")
plt.title("Maaş Değişkeni Boxplot")
plt.show()

Q1 = df["Maas"].quantile(0.25)
Q3 = df["Maas"].quantile(0.75)

IQR = Q3 - Q1

alt_sinir = Q1 - 1.5 * IQR
ust_sinir = Q3 + 1.5 * IQR

aykiri_degerler = df[
    (df["Maas"] < alt_sinir) |
    (df["Maas"] > ust_sinir)
]

print("Q1:", Q1)
print("Q3:", Q3)
print("IQR:", IQR)
print("Alt sınır:", alt_sinir)
print("Üst sınır:", ust_sinir)

print("\nAykırı değerler:")
print(aykiri_degerler)