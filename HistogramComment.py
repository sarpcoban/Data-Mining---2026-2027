import numpy as np
import matplotlib.pyplot as plt

# Sonuçların her çalıştırmada aynı olması için
np.random.seed(42)

# 1. Simetrik (normal) dağılım
normal = np.random.normal(loc=50, scale=10, size=1000)

# 2. Sağa çarpık dağılım
saga_carpik = np.random.exponential(scale=10, size=1000)

# 3. Sola çarpık dağılım
sola_carpik = 50 - np.random.exponential(scale=10, size=1000)

# Grafik alanı
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Normal dağılım
axes[0].hist(normal, bins=30, edgecolor="black")
axes[0].axvline(np.mean(normal), linestyle="--", label="Ortalama")
axes[0].set_title("Simetrik / Normal Dağılım")
axes[0].set_xlabel("Değer")
axes[0].set_ylabel("Frekans")
axes[0].legend()

# Sağa çarpık
axes[1].hist(saga_carpik, bins=30, edgecolor="black")
axes[1].axvline(np.mean(saga_carpik), linestyle="--", label="Ortalama")
axes[1].axvline(np.median(saga_carpik), linestyle=":", label="Medyan")
axes[1].set_title("Sağa Çarpık Dağılım")
axes[1].set_xlabel("Değer")
axes[1].legend()

# Sola çarpık
axes[2].hist(sola_carpik, bins=30, edgecolor="black")
axes[2].axvline(np.mean(sola_carpik), linestyle="--", label="Ortalama")
axes[2].axvline(np.median(sola_carpik), linestyle=":", label="Medyan")
axes[2].set_title("Sola Çarpık Dağılım")
axes[2].set_xlabel("Değer")
axes[2].legend()

plt.tight_layout()
plt.show()