 ## Random Forest ##

## Importar las bibliotecas necesarias

import pandas as pd
import numpy as np
# Visualización
import matplotlib.pyplot as plt
import seaborn as sns
# Modelado
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

## Cargar el dataset

df = pd.read_csv('databases/disuguaglianza-economica-globale-e-povert-1980-2024.csv')

df.head()

## Exploración de datos

df.info()
df.describe()

## Verificar valores nulos

df.isnull().sum()

## Limpiar los datos

# Eliminar filas sin target

df = df.dropna(subset = ['gini_index'])

# Imputar valores faltantes en features numéricas con la mediana

for col in ['poverty_rate', 'gdp_per_capita']:
    df[col] = df[col].fillna(df[col].median())

## Analisis exploratorio

plt.figure(figsize=(10, 6))
sns.histplot(df['gini_index'], bins=30, kde=True)
plt.title('Distribución del Índice de Gini')
plt.show()

# Matriz de correlación

plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title('Matriz de Correlación')
plt.show()

## Feature Engineering

df_model = df.drop(columns=['country', 'iso_code'])


## Definir variables predictoras y variable objetivo

X = df_model.drop('gini_index', axis=1)
y = df_model['gini_index']

## Dividir el dataset en conjunto de entrenamiento y prueba

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

## Entrenar el modelo de Random Forest

model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)

model.fit(X_train, y_train)

## Predicciones

y_pred = model.predict(X_test)

## Evaluación del modelo

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f'MAE: {mae:.4f}')
print(f'RMSE: {rmse:.4f}')
print(f'R²: {r2:.4f}')

## Importancia de las características

importances = pd.DataFrame({
    'Feature': X.columns, 'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)

print(importances)    

# Visualización de la importancia de las características

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importances)
plt.title('Importancia de las Variables')
plt.show()
