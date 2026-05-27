## Caso de Estudio: Segmentación de Transacciones con GMM ##

## Importar las bibliotecas necesarias

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
from matplotlib.gridspec import GridSpec
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import matplotlib.cm as cm

## Cargar el dataset

df = pd.read_csv('databases/ventas-por-factura.csv')

df_raw = df.copy()

# Parseo del campo Monto (usa coma como separador decimal)
df_raw['Monto'] = (df_raw['Monto'].astype(str)
                   .str.replace('.', '', regex=False)
                   .str.replace(',', '.', regex=False)
                   .astype(float))
df_raw['Fecha de factura'] = pd.to_datetime(
    df_raw['Fecha de factura'], dayfirst=False, errors='coerce')

print(f"\n✓ Archivo cargado: {filename}")
print(f"  Filas: {df_raw.shape[0]:,}  |  Columnas: {df_raw.shape[1]}")

## Análisis Exploratorio de Datos (EDA)

df_raw.info()

df_raw.head(8)

df_raw[['Cantidad', 'Monto']].describe().round(2)

df_raw.isnull().sum()

## Limpieza y preparación de los datos

df = df_raw.copy()

# 1. Eliminar devoluciones: facturas que empiezan con 'C'

df = df[~df['N° de factura'].str.startswith('C', na=False)]

# 2. Eliminar cantidades y montos negativos o cero

df = df[(df['Cantidad'] > 0) & (df['Monto'] > 0)]

# 3. Eliminar registros sin cliente identificado

df = df.dropna(subset=['ID Cliente']) 
df['ID Cliente'] = df['ID Cliente'].astype(int)

# 4. Seleccionar features para el modelo

X_raw = df[['Cantidad', 'Monto']].copy()

# 5. Remover outliers extremos (percentil 99) — mejora visualización

q99_c = X_raw['Cantidad'].quantile(0.99)
q99_m = X_raw['Monto'].quantile(0.99)
X_raw = X_raw[(X_raw['Cantidad'] <= q99_c) & (X_raw['Monto'] <= q99_m)]

# 6. Escalado estándar — OBLIGATORIO para GMM
#    (media=0, std=1) para que ambas variables tengan igual peso

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

## Exploración visual

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
fig.suptitle('Análisis Exploratorio de Datos', fontsize=16)

# Scatter: Cantidad vs Monto

ax = axes[0]
ax.scatter(X_raw['Cantidad'], X_raw['Monto'], alpha=0.5, color='blue')
ax.set_title('Cantidad vs Monto')
ax.set_xlabel('Cantidad')
ax.set_ylabel('Monto')

# Histograma: Cantidad

ax = axes[1]
ax.hist(X_raw['Cantidad'], bins = 60, color='orange', edgecolor='black')
ax.set_title('Distribución de Cantidad')
ax.set_xlabel('Cantidad')
ax.set_ylabel('Frecuencia')
ax.avline(X_raw['Cantidad'].mean(), color=ACCENT, lw=1.8,
           label=f'Media = {X_raw["Cantidad"].mean():.0f}')
ax.legend()
ax.grid(True, alpha=0.3) 

# Histograma: Monto

ax = axes[2]
ax.hist(X_raw['Monto'], bins=60, color='green', edgecolor='black')
ax.set_title('Distribución de Monto')
ax.set_xlabel('Monto')
ax.set_ylabel('Frecuencia')
ax.avline(X_raw['Monto'].mean(), color=ACCENT, lw=1.8,
           label=f'Media = {X_raw["Monto"].mean():.0f}')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()           

##  Selección de K con BIC y AIC

K_range = 