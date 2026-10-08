import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# 1. Pozitif korelasyon
x1 = np.arange(1, 21)
y1 = 2 * x1 + np.random.normal(0, 4, 20)

# 2. Negatif korelasyon
x2 = np.arange(1, 21)
y2 = 50 - 2 * x2 + np.random.normal(0, 4, 20)

# 3. Korelasyon yok
x3 = np.random.randint(1, 20, 30)
y3 = np.random.randint(1, 20, 30)

# 4. Yoğun kümeler / alt gruplar
x4 = np.concatenate([
    np.random.normal(5, 1, 20),
    np.random.normal(15, 1, 20)
])

y4 = np.concatenate([
    np.random.normal(5, 1, 20),
    np.random.normal(15, 1, 20)
])

# Grafikler
fig, ax = plt.subplots(2, 2, figsize=(10, 8))

# Pozitif korelasyon
ax[0, 0].scatter(x1, y1)
ax[0, 0].set_title("Pozitif Korelasyon")

# Negatif korelasyon
ax[0, 1].scatter(x2, y2)
ax[0, 1].set_title("Negatif Korelasyon")

# Korelasyon yok
ax[1, 0].scatter(x3, y3)
ax[1, 0].set_title("Korelasyon Yok")

# Kümeler
ax[1, 1].scatter(x4, y4)
ax[1, 1].set_title("Alt Gruplar / Kümeler")

plt.tight_layout()
plt.show()