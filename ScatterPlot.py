import pandas as pd
import matplotlib.pyplot as plt

# Örnek veri
df = pd.DataFrame({
    "Boy":  [160, 165, 168, 170, 172, 175, 178, 180, 182, 185, 190],
    "Kilo": [55,  60,  63,  65,  68,  72,  75,  78,  80,  85,  150]
})

# Scatter Plot
plt.scatter(df["Boy"], df["Kilo"])

plt.xlabel("Boy (cm)")
plt.ylabel("Kilo (kg)")
plt.title("Boy - Kilo Scatter Plot")

plt.show()

plt.scatter(df["Boy"], df["Kilo"])

# Aykırı gözlemi işaretleme
plt.scatter(190, 150, s=150, marker="x")

plt.xlabel("Boy (cm)")
plt.ylabel("Kilo (kg)")
plt.title("Scatter Plot ile Aykırı Değer Belirleme")

plt.show()