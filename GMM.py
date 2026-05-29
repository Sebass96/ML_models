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

K_range = range(1, 9)
bics, aics = [], []

for k in K_range:
    gmm_k = GaussianMixture(n_components=k, covariance_type = 'full',
                            n_init = 5, random_state=42) # n_init=5 entrena el modelo 5 veces con diferentes inicializaciones y elige la mejor
    gmm_k.fit(X_scaled)
    bics.append(gmm_k.bic(X_scaled))
    aics.append(gmm_k.aic(X_scaled))
    delta = bics[-1] - bics[-2] if len(bics) > 1 else 0
    print(f"  K={k}  BIC={gmm_k.bic(X_scaled):>10,.1f}  "
          f"AIC={gmm_k.aic(X_scaled):>10,.1f}  ΔBIC={delta:>+10,.1f}")  

K_FINAL = 4                         

## Curvas BIC y AIC

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
fig.suptitle('Selección de K — Criterio del Codo en BIC/AIC',
             fontsize=14, fontweight='bold', y=1.02)

K_list = list(K_range)
for ax, vals, nombre, clr in zip(
        axes,
        [bics, aics],
        ['BIC — Bayesian Information Criterion',
         'AIC — Akaike Information Criterion'],
        ['#E63946', '#457B9D']):

    ax.plot(K_list, vals, 'o-', color=clr, lw=2.5, ms=8,
            markerfacecolor='white', markeredgewidth=2)
    ax.axvline(K_FINAL, color='#E9C46A', lw=2, ls='--',
               label=f'Codo K = {K_FINAL}')
    ax.scatter([K_FINAL], [vals[K_FINAL - 1]],
               color='#E9C46A', s=160, zorder=5)
    ax.annotate('Codo\n(punto de\ninflexión)',
                xy=(K_FINAL, vals[K_FINAL - 1]),
                xytext=(K_FINAL + 1.3,
                        vals[K_FINAL - 1] + (max(vals) - min(vals)) * 0.15),
                arrowprops=dict(arrowstyle='->', color='#E9C46A', lw=1.5),
                color='#E9C46A', fontsize=8)
    ax.set_xlabel('Número de componentes K')
    ax.set_ylabel('Criterio de información')
    ax.set_title(nombre)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.4)
    ax.set_xticks(K_list)

plt.tight_layout()
plt.show()

## Entrenar el modelo final con K=4

gmm = GaussianMixture(
    n_components = K_FINAL,
    covariance_type = 'full',
    n_init = 10, # n_init=10 entrena el modelo 10 veces con diferentes inicializaciones y elige la mejor
    max_iter = 300, # Aumentar el número máximo de iteraciones para asegurar convergencia
    random_state = 42
)

gmm.fit(X_scaled)

# Desescalar medias para interpretarlas en unidades originales
medias_orig = scaler.inverse_transform(gmm.means_)

print(f'\n  ── Parámetros aprendidos (espacio original) ──')
for k in range(K_FINAL):
    print(f'\n  Componente {k+1}:')
    print(f'    π (peso)         : {gmm.weights_[k]:.4f}'
          f'  ({gmm.weights_[k]*100:.1f}% de las transacciones)')
    print(f'    μ Cantidad       : {medias_orig[k, 0]:,.1f} unidades')
    print(f'    μ Monto          : £{medias_orig[k, 1]:,.2f}')

# ── Asignaciones y responsabilidades ──
resps  = gmm.predict_proba(X_scaled)   # (N, K)  — SOFT assignment
labels = gmm.predict(X_scaled)         # (N,)    — HARD assignment
log_p  = gmm.score_samples(X_scaled)   # log p(x_i)

X_model = X_raw.copy().reset_index(drop=True)
X_model['Segmento'] = labels + 1
X_model['log_prob'] = log_p
for k in range(K_FINAL):
    X_model[f'r_{k+1}'] = resps[:, k]

print(f'\n  ── Distribución por segmento ──')
for k in range(K_FINAL):
    n = (labels == k).sum()
    print(f'    Segmento {k+1}: {n:,} transacciones ({n/len(labels)*100:.1f}%)')

##  Soft vs Hard assignment

PALETTE  = ['#E63946', '#457B9D', '#2A9D8F', '#E9C46A',
            '#6A4C93', '#F4A261', '#264653', '#A8DADC']

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
fig.suptitle('Responsabilidades del EM: Asignación Suave vs. Dura',
             fontsize=14, fontweight='bold', y=1.02)

# ── SOFT: color = r_1 (prob. de pertenecer al Seg. 1) ──
sc = axes[0].scatter(X_scaled[:, 0], X_scaled[:, 1],
                     c=resps[:, 0], cmap='RdYlBu_r',
                     s=8, alpha=0.6, linewidths=0)
plt.colorbar(sc, ax=axes[0],
             label='r₁ — responsabilidad componente 1')
axes[0].set_title('SOFT assignment\n'
                  'Color = probabilidad de pertenecer al Segmento 1\n'
                  '(azul ≈ 0%, rojo ≈ 100%)')
axes[0].set_xlabel('Cantidad (escalada)')
axes[0].set_ylabel('Monto (escalado)')
axes[0].grid(True, alpha=0.3)

# ── HARD: etiqueta del segmento de mayor responsabilidad ──
for k in range(K_FINAL):
    mask = labels == k
    axes[1].scatter(X_scaled[mask, 0], X_scaled[mask, 1],
                    s=8, alpha=0.5, color=PALETTE[k],
                    linewidths=0, label=f'Segmento {k+1}')
axes[1].set_title('HARD assignment\n'
                  'Cada punto → segmento de mayor r_k\n'
                  '(arg máx de las responsabilidades)')
axes[1].set_xlabel('Cantidad (escalada)')
axes[1].set_ylabel('Monto (escalado)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("""
  CONCEPTO CLAVE:
  El GMM provee asignación SUAVE: cada transacción pertenece a
  todos los segmentos con cierta probabilidad.
  Ejemplo: r₁=0.85 → el punto pertenece principalmente al seg. 1
           pero hay un 15% de probabilidad de que sea del seg. 2.

  La asignación DURA es simplemente: ẑ = argmáx_k r_ik
  Esto es equivalente a lo que hace K-means, pero basado en
  probabilidades en lugar de distancias.
""")


