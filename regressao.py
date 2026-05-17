import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error


# =========================
# CARREGAR DATASET
# =========================

df = pd.read_csv("house_price_regression_dataset.csv")

print("5 primeiras linhas:")
print(df.head())


# =========================
# ANÁLISE INICIAL
# =========================

print("\nQuantidade de linhas e colunas:")
print(df.shape)

print("\nColunas:")
print(df.columns)

print("\nValores nulos:")
print(df.isnull().sum())

print("\nEstatísticas:")
print(df.describe())


# =========================
# SELEÇÃO DAS COLUNAS
# =========================

df2 = df[[
    'Square_Footage',
    'Num_Bedrooms',
    'Neighborhood_Quality',
    'House_Price'
]]

print("\nDescribe das colunas selecionadas:")
print(df2.describe())


# =========================
# GRÁFICO
# =========================oro

plt.scatter(df['Square_Footage'], df['House_Price'])

plt.xlabel("Metragem")
plt.ylabel("Preço")

plt.title("Metragem x Preço")

plt.show()


# =========================
# X e y
# =========================

X = df[['Square_Footage', 'Num_Bedrooms', 'Neighborhood_Quality']]

y = df['House_Price']


# =========================
# HOLDOUT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# =========================
# MODELO
# =========================

modelo = LinearRegression()

modelo.fit(X_train, y_train)


# =========================
# PREVISÕES
# =========================

y_pred = modelo.predict(X_test)


# =========================
# MÉTRICAS
# =========================

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

print("\nMAE:")
print(mae)

print("\nMSE:")
print(mse)


# =========================
# INTERPRETAÇÃO
# =========================

print("""
O MAE representa o erro médio absoluto.
Ele mostra quanto o modelo erra em média.

O MSE representa o erro quadrático médio.
Ele penaliza erros maiores.

Se os erros forem baixos,
o modelo conseguiu prever bem os preços.
""")