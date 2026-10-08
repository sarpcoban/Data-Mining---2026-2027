import pandas as pd
import numpy as np

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

df = pd.DataFrame({
    "Yas": [25, 30, 35, 40, 45, 50],
    "Gelir": [30000, 42000, np.nan, 60000, 68000, 75000],
    "Harcama": [12000, 18000, 22000, 28000, np.nan, 36000]
})

print("Orijinal veri:")
print(df)

# 5 farklı imputasyonlu veri seti üretelim
imputed_datasets = []

for i in range(5):

    imputer = IterativeImputer(
        random_state=i,
        sample_posterior=True
    )

    imputed = imputer.fit_transform(df)

    imputed_df = pd.DataFrame(
        imputed,
        columns=df.columns
    )

    imputed_datasets.append(imputed_df)

    print(f"\n{i+1}. İmputasyon:")
    print(imputed_df.round(2))

from sklearn.linear_model import LinearRegression

coefficients = []

for dataset in imputed_datasets:

    X = dataset[["Yas", "Gelir"]]
    y = dataset["Harcama"]

    model = LinearRegression()
    model.fit(X, y)

    coefficients.append(model.coef_)

print(coefficients)