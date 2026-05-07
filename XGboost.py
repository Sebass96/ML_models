## XGBoost Aplicado: Predicción de Precio de Smartphones ##

## Importar las bibliotecas necesarias

### Librerías para manipulación de datos

from numpy.testing import verbose
import pandas as pd
import numpy as np
import re
import warnings
warnings.filterwarnings('ignore')

### Librerías para visualización
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

### Librerías para modelado
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print('✅ Todas las librerías cargadas correctamente')
print(f'   XGBoost versión: {xgb.__version__}')

## Cargar el dataset

df_raw = pd.read_csv('databases/Mobiles_Dataset.csv', encoding = 'latin-1')

print(f'📐 Dimensiones del dataset: {df_raw.shape[0]} filas × {df_raw.shape[1]} columnas')
print(f'\n📋 Variables disponibles:')
for col in df_raw.columns:
    print(f'   • {col} ({df_raw[col].dtype})')

df_raw.head(8)    

## Distribución por marca

brand_counts = df_raw['Company Name'].value_counts()

fig, ax = plt.subplots(figsize = (10, 6))
bars = ax.bar(brand_counts.index, brand_counts.values,
                color = sns.color_palette('muted', len(brand_counts)))

## Etiquetas
for bar, val in zip(bars, brand_counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
            str(val), ha = 'center', va = 'bottom', fontsize = 10, fontweight = 'bold')

ax.set_title('Distribución de Smartphones por Marca', fontsize = 16, fontweight = 'bold')
ax.set_xlabel('Marca', fontsize = 12)
ax.set_ylabel('Cantidad de Modelos', fontsize = 12)
plt.xticks(rotation = 45, ha = 'right')
plt.tight_layout()
plt.show()

print(f'\n✅ Total de marcas: {df_raw["Company Name"].nunique()}')

## Valores faltantes por variable

print('📊 Valores faltantes por variable:')
nulls = df_raw.isnull().sum()
print(nulls[nulls > 0] if nulls.sum() > 0 else '   No hay valores faltantes.')

## Preprocesamiento de datos

# ─── Función auxiliar: extraer primer número de un string
def extract_number(series: pd.Series) -> pd.Series:
    """
    Extrae el primer número (entero o decimal) de cada celda de texto.
    Elimina comas de separación de miles antes de parsear.
    Ejemplo: '3,600mAh' → 3600.0 | 'USD 799' de→ 799.0
    """
    return (
        series
        .astype(str)
        .str.replace(',', '', regex = False)  # Eliminar comas
        .str.extract(r'(\d+\.?\d*)')  # Extraer el primer número
        .astype(float)  # Convertir a float
    )

# ─── Construcción del dataframe limpio
df = pd.DataFrame()

df['brand'] = df_raw['Company Name']
df['ram_gb'] = extract_number(df_raw['RAM'])
df['battery_mah'] = extract_number(df_raw['Battery Capacity'])
df['screen_inches'] = extract_number(df_raw['Screen Size'])
df['weight_g'] = extract_number(df_raw['Mobile Weight'])
df['front_cam_mp'] = extract_number(df_raw['Front Camera'])
df['back_cam_mp'] = extract_number(df_raw['Back Camera'])
df['launched_year'] = extract_number(df_raw['Launched Year'])
df['price_usd'] = extract_number(df_raw['Launched Price (USA)'])

#─── Codificación de la variable 'brand'
# Label Encoding: asignar un número único a cada marca

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['brand_encoded'] = le.fit_transform(df['brand'])

# Tabla de referencia para marcas codificadas
brand_map = dict(zip(le.classes_, le.transform(le.classes_)))
print('✅ Codificación de marcas:')
for brand, code in brand_map.items():
    print(f'   {code:2d} → {brand}')

## Resumen estadístico
print('📊 Estadísticas descriptivas del dataset procesado:\n')
df.drop(columns = 'brand').describe().round(2).T    

## Distribución de la variable objetivo 'price_usd'

fig, axes = plt.subplots(1, 2, figsize = (14, 5))

# Histograma
axes[0].hist(df['price_usd'], bins = 40, color = 'skyblue', edgecolor = 'white', alpha = 0.85)
axes[0].axvline(df['price_usd'].mean(), color = 'tomato', linestyle = '--', lw = 2, 
                label = f'Media: ${df["price_usd"].mean():.2f}')
axes[0].axvline(df['price_usd'].median(), color = 'gold', linestyle = '--', lw = 2, 
                label = f'Mediana: ${df["price_usd"].median():.2f}')
axes[0].set_title('Distribución del Precio de Smartphones', fontsize = 14, fontweight = 'bold')
axes[0].set_xlabel('Precio en USD', fontsize = 12)
axes[0].set_ylabel('Frecuencia', fontsize = 12)
axes[0].legend()                              

# Boxplot
top_brands = df['brand'].value_counts().head(8).index
df_top = df[df['brand'].isin(top_brands)]
brand_order = df_top.groupby('brand')['price_usd'].median().sort_values(ascending = False).index

sns.boxplot(data = df_top, x = 'brand', y = 'price_usd', order = brand_order, 
            ax = axes[1], palette = 'muted')
axes[1].set_title('Precio de Smartphones por Marca (Top 8)', fontsize = 14, fontweight = 'bold')
axes[1].set_xlabel('Marca', fontsize = 12)
axes[1].set_ylabel('Precio en USD', fontsize = 12)
axes[1].tick_params(axis = 'x', rotation = 45)

plt.tight_layout()
plt.show()

# Correlaciones con el precio

features = ['ram_gb', 'battery_mah', 'screen_inches', 'weight_g',
            'front_cam_mp', 'back_cam_mp', 'launched_year', 'brand_encoded']

corr = df[features + ['price_usd']].corr()['price_usd'].drop('price_usd').sort_values()

fig, ax = plt.subplots(figsize = (10, 6))
colors = ['tomato' if v < 0 else 'seagreen' for v in corr]
bars = ax.barh(corr.index, corr.values, color = colors, edgecolor = 'white')
ax.axvline(0, color = 'black', lw = 0.8)

for bar, val in zip(bars, corr.values):
    xpos = val + 0.01 if val > 0 else val - 0.01
    ha = 'left' if val > 0 else 'right'
    ax.text(xpos, bar.get_y() + bar.get_height() / 2, 
            f'{val:.2f}', ha = ha, va = 'center', fontsize = 10)

ax.set_title('Correlación de Características con el Precio', fontsize = 14, fontweight = 'bold')
ax.set_xlabel('Correlación de Pearson', fontsize = 12)
ax.set_xlim(-0.5, 0.75)
plt.tight_layout()
plt.show()            

# ─── Demostración numérica: cálculo de Gain para un nodo ─────────────────────
# Supongamos que tenemos 6 smartphones en un nodo y evaluamos dividir por RAM ≤ 6GB

# Residuos simulados (precio_real - prediccion_actual)
# Grupo izquierdo: RAM ≤ 6GB  → 3 teléfonos baratos
# Grupo derecho:  RAM > 6GB   → 3 teléfonos premium
g_L = np.array([-200, -150, -180])  # gradientes izquierda
g_R = np.array([300, 400, 350])     # gradientes derecha

# Para pérdida cuadrática h_i = 1 siempre
h_L = np.ones(3)
h_R = np.ones(3)

# Parámetros de regularización
lam = 1.0   # lambda (reg_lambda)
gam = 0.1   # gamma

# Sumas de gradientes y hessianos por hoja (Sección 3.4 del material)
G_L, H_L = g_L.sum(), h_L.sum()
G_R, H_R = g_R.sum(), h_R.sum()
G_P, H_P = G_L + G_R, H_L + H_R  # padre (antes de dividir)

# Cálculo del Gain (Ecuación 11)
term_L = G_L**2 / (H_L + lam)
term_R = G_R**2 / (H_R + lam)
term_P = G_P**2 / (H_P + lam)
gain   = 0.5 * (term_L + term_R - term_P) - gam

# Pesos óptimos de cada hoja (Ecuación 9)
w_L = -G_L / (H_L + lam)
w_R = -G_R / (H_R + lam)

print('=' * 55)
print('  DEMOSTRACIÓN: Criterio de división de un nodo')
print('=' * 55)
print(f'  División: RAM ≤ 6GB (izquierda) | RAM > 6GB (derecha)')
print(f'  λ = {lam}  |  γ = {gam}')
print()
print(f'  G_L = {G_L:.1f}   H_L = {H_L:.1f}')
print(f'  G_R = {G_R:.1f}   H_R = {H_R:.1f}')
print()
print(f'  Término L  = G_L² / (H_L + λ) = {term_L:.2f}')
print(f'  Término R  = G_R² / (H_R + λ) = {term_R:.2f}')
print(f'  Término P  = G_P² / (H_P + λ) = {term_P:.2f}')
print()
print(f'  Gain = 0.5 × ({term_L:.2f} + {term_R:.2f} − {term_P:.2f}) − {gam}')
print(f'  Gain = {gain:.4f}  → {"✅ SE DIVIDE" if gain > 0 else "❌ NO SE DIVIDE"}')
print()
print(f'  Peso óptimo hoja izquierda  w*_L = {w_L:.2f}  (precio ajustado: ${w_L:.0f})')
print(f'  Peso óptimo hoja derecha    w*_R = {w_R:.2f}  (precio ajustado: +${w_R:.0f})')
print('=' * 55)

## Entrenamiento del modelo XGBoost

# ─── Separación features / target ─────────────────────────────────────────────
FEATURES = ['ram_gb', 'battery_mah', 'screen_inches', 'weight_g',
            'front_cam_mp', 'back_cam_mp', 'launched_year', 'brand_encoded']
TARGET   = 'price_usd'

X = df[FEATURES]
y = df[TARGET]

print(f'✅ Features: {FEATURES}')
print(f'✅ Target:   {TARGET}')
print(f'✅ Shape X: {X.shape}  |  Shape y: {y.shape}')

# ─── División entrenamiento / prueba (80% / 20%) ──────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size= 0.20, random_state= 42
)

print(f'✅ Conjunto de entrenamiento: {X_train.shape[0]} muestras')
print(f'✅ Conjunto de prueba: {X_test.shape[0]} muestras')

# ─── Definición del modelo XGBoost ────────────────────────────────────────────
#
# Referencia de hiperparámetros → Tabla 1 del material (Sección 4)
#
#   n_estimators  : número de árboles secuenciales (M en la Ecuación 1)
#   learning_rate : η — escala la contribución de cada árbol
#   max_depth     : profundidad máxima de cada árbol base
#   reg_lambda    : λ — regularización L2 sobre los pesos de las hojas (Ec. 7)
#   gamma         : γ — mínima ganancia para dividir un nodo (Ec. 11)
#   subsample     : fracción de muestras usadas por árbol (reduce varianza)
#   colsample_bytree: fracción de features usadas por árbol
#   random_state  : semilla para reproducibilidad

modelo_xgb = xgb.XGBRegressor(
    n_estimators      = 300,
    learning_rate     = 0.05,   # η pequeño → convergencia más suave
    max_depth         = 5,
    reg_lambda        = 1.5,    # λ: regularización L2
    gamma             = 0.2,    # γ: penalización por división
    subsample         = 0.8,    # 80% de muestras por árbol
    colsample_bytree  = 0.8,    # 80% de features por árbol
    random_state      = 42,
    verbosity         = 0       # sin logs durante entrenamiento
)       

print('⚙️  Hiperparámetros configurados:')
params = modelo_xgb.get_params()
for key in ['n_estimators', 'learning_rate', 'max_depth', 'reg_lambda', 'gamma', 'subsample', 'colsample_bytree']:
    print(f'   {key:20s} = {params[key]}')

# ─── Entrenamiento con monitoreo de la curva de aprendizaje ──────────────────
# eval_set permite ver cómo evolucionan el error de train y test en cada árbol

modelo_xgb.fit(
    X_train, y_train,
    eval_set = [(X_train, y_train), (X_test, y_test)],
    verbose = False  # No imprimir logs de cada iteración
)    

print('✅ Modelo entrenado correctamente')
print(f'   Árboles construidos: {modelo_xgb.n_estimators}')

## Evaluación y métricas

# ─── Predicciones ─────────────────────────────────────────────────────────────
y_pred_train = modelo_xgb.predict(X_train)
y_pred_test  = modelo_xgb.predict(X_test)

# ─── Cálculo de métricas ──────────────────────────────────────────────────────
def evaluar(y_real, y_pred, nombre):
    mae = mean_absolute_error(y_real, y_pred)
    rmse = np.sqrt(mean_squared_error(y_real, y_pred))
    r2 = r2_score(y_real, y_pred)
    mape = np.mean(np.abs((y_real - y_pred) / (y_real + 1e-9))) * 100
    print(f'  ── {nombre} ──')
    print(f'  MAE  (Error Absoluto Medio)  : ${mae:,.2f}')
    print(f'  RMSE (Raíz Error Cuadrático) : ${rmse:,.2f}')
    print(f'  MAPE (Error % Medio)         : {mape:.2f}%')
    print(f'  R²   (Coef. Determinación)   : {r2:.4f}')
    return mae, rmse, r2

print('\n📊 RESULTADOS DEL MODELO\n')
mae_tr, rmse_tr, r2_tr = evaluar(y_train, y_pred_train, '🔵 ENTRENAMIENTO')
print()
mae_te, rmse_te, r2_te = evaluar(y_test,  y_pred_test,  '🟢 PRUEBA')

print()
overfit = r2_tr - r2_te
print(f'  Diferencia R² (train - test): {overfit:.4f}  →  ', end='')
print('✅ Bien generalizado' if overfit < 0.1 else '⚠️  Posible sobreajuste') 

# ─── Curva de aprendizaje (train vs test por árbol) ───────────────────────────
# Muestra cómo cada árbol nuevo reduce el error (concepto central de boosting

from matplotlib import ticker as mticker

results = modelo_xgb.evals_result()
epochs = len(results['validation_0']['rmse'])
x_axis = range(epochs)

fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(x_axis, results['validation_0']['rmse'], color='steelblue',
        alpha=0.7, label='Error Entrenamiento (RMSE)', lw=1.5)
ax.plot(x_axis, results['validation_1']['rmse'], color='tomato',
        alpha=0.9, label='Error Prueba (RMSE)', lw=2)

# Mínimo de test
best_iter = np.argmin(results['validation_1']['rmse'])
best_rmse = results['validation_1']['rmse'][best_iter]
ax.axvline(best_iter, color='gray', linestyle='--', lw=1)
ax.scatter(best_iter, best_rmse, color='tomato', s=80, zorder=5)
ax.annotate(f'Mínimo: ${best_rmse:,.0f}\n(árbol {best_iter})',
            xy=(best_iter, best_rmse),
            xytext=(best_iter + 20, best_rmse + 30),
            fontsize=9, arrowprops=dict(arrowstyle='->', color='gray'))

ax.set_title('📉 Curva de aprendizaje: reducción del error por árbol\n'
             '(Sección 5.2 del material — Ejemplo de Regresión)',
             fontweight='bold')
ax.set_xlabel('Número de árboles construidos')
ax.set_ylabel('RMSE (USD)')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))
ax.legend()
plt.tight_layout()
plt.show()

print(f'\n📌 Cada punto en la curva = un árbol nuevo que corrige los residuos del anterior')
print(f'   Esto es exactamente el proceso de boosting descrito en la Sección 2 del material.')

# ─── Importancia de características ──────────────────────────────────────────
# Cuánto contribuyó cada variable a las decisiones de los árboles

importances = pd.Series(modelo_xgb.feature_importances_, index=FEATURES)
importances = importances.sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(10, 6))
colors = sns.color_palette('muted', len(importances))
bars = ax.barh(importances.index, importances.values, color=colors, edgecolor='white')

for bar, val in zip(bars, importances.values):
    ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
            f'{val:.3f}', ha='left', va='center', fontsize=10)

ax.set_title('Importancia de Características según XGBoost', fontsize=14, fontweight='bold')
ax.set_xlabel('Importancia Normalizada', fontsize=12)
ax.set_xlim(0, importances.max() * 1.18)
plt.tight_layout()
plt.show()

# ─── Predichos vs Reales ──────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

lim = (0, max(y_test.max(), y_pred_test.max()) * 1.05)

for ax, y_real, y_pred, title, color in [
    (axes[0], y_train, y_pred_train, '🔵 Entrenamiento', 'steelblue'),
    (axes[1], y_test,  y_pred_test,  '🟢 Prueba',        'seagreen')
]:
    ax.scatter(y_real, y_pred, alpha=0.5, color=color, s=25, edgecolors='white', lw=0.3)
    ax.plot(lim, lim, 'k--', lw=1.5, label='Predicción perfecta')
    r2 = r2_score(y_real, y_pred)
    ax.set_title(f'{title}   R² = {r2:.4f}', fontweight='bold')
    ax.set_xlabel('Precio real (USD)')
    ax.set_ylabel('Precio predicho (USD)')
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))
    ax.legend(fontsize=8)

plt.suptitle('Precio Real vs. Precio Predicho por XGBoost', fontweight='bold', y=1.01)
plt.tight_layout()
plt.show()

# ─── Distribución de residuos ─────────────────────────────────────────────────
residuos = y_test.values - y_pred_test

fig, axes = plt.subplots(1, 2, figsize=(13, 4))

# Histograma de residuos
axes[0].hist(residuos, bins=35, color='mediumpurple', edgecolor='white', alpha=0.85)
axes[0].axvline(0, color='black', lw=1.5, linestyle='--')
axes[0].axvline(residuos.mean(), color='tomato', lw=2,
                label=f'Media: ${residuos.mean():.0f}')
axes[0].set_title('Distribución de Residuos (y_real − ŷ)', fontweight='bold')
axes[0].set_xlabel('Error (USD)')
axes[0].set_ylabel('Frecuencia')
axes[0].legend()

# Residuos vs predichos
axes[1].scatter(y_pred_test, residuos, alpha=0.5, color='mediumpurple',
                s=25, edgecolors='white', lw=0.3)
axes[1].axhline(0, color='black', lw=1.5, linestyle='--')
axes[1].set_title('Residuos vs. Valores Predichos', fontweight='bold')
axes[1].set_xlabel('Precio predicho (USD)')
axes[1].set_ylabel('Residuo (USD)')
axes[1].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))

plt.suptitle('Análisis de Residuos del Modelo', fontweight='bold', y=1.01)
plt.tight_layout()
plt.show()

# ─── Grid Search con validación cruzada ─────────────────────────────────────
# NOTA: Reducimos el grid para que sea rápido en clase.
# En producción se exploraría un espacio más amplio o se usaría Bayesian Search.

param_grid = {
    'n_estimators': [100, 300],
    'learning_rate': [0.05, 0.1],
    'max_depth': [4, 5, 6],
    'gamma': [0.1, 0.3]
}

gs = GridSearchCV(
    estimator  = xgb.XGBRegressor(random_state=42, verbosity=0,
                                   subsample=0.8, colsample_bytree=0.8),
    param_grid = param_grid,
    cv         = 5,               # 5-fold cross validation
    scoring    = 'neg_root_mean_squared_error',
    n_jobs     = -1,              # usa todos los núcleos disponibles
    verbose    = 1
)

gs.fit(X_train, y_train)

print('\n🏆 Mejores hiperparámetros encontrados:')
for k, v in gs.best_params_.items():
    print(f'   {k:20s} = {v}')
print(f'\n   RMSE (CV) = ${-gs.best_score_:,.2f}')

# ─── Modelo final con los mejores hiperparámetros ─────────────────────────────
modelo_final = gs.best_estimator_
y_pred_final = modelo_final.predict(X_test)

mae_f  = mean_absolute_error(y_test, y_pred_final)
rmse_f = np.sqrt(mean_squared_error(y_test, y_pred_final))
r2_f   = r2_score(y_test, y_pred_final)

print('\n📊 COMPARATIVA: Modelo base vs Modelo optimizado (sobre conjunto de prueba)')
print(f'{"Métrica":30s} {"Base":>15s} {"Optimizado":>15s} {"Mejora":>10s}')
print('-' * 72)
print(f'{"MAE  ($)":30s} {mae_te:>15,.2f} {mae_f:>15,.2f} {mae_te - mae_f:>+10,.2f}')
print(f'{"RMSE ($)":30s} {rmse_te:>15,.2f} {rmse_f:>15,.2f} {rmse_te - rmse_f:>+10,.2f}')
print(f'{"R²":30s} {r2_te:>15.4f} {r2_f:>15.4f} {r2_f - r2_te:>+10.4f}')

# ─── Efecto de gamma sobre la complejidad (Ecuación 11) ────────────────────────
# Visualiza cómo γ controla el número de hojas y el error
gammas = [0, 0.1, 0.5, 1.0, 2.0, 5.0]
rmse_vals = []

for g in gammas:
    m = xgb.XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=5,
                          gamma=g, subsample=0.8, random_state=42, verbosity=0)
    m.fit(X_train, y_train)
    preds = m.predict(X_test)
    rmse_vals.append(np.sqrt(mean_squared_error(y_test, preds)))

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(gammas, rmse_vals, 'o-', color='darkorange', lw=2, markersize=8)
ax.fill_between(gammas, rmse_vals, min(rmse_vals), alpha=0.1, color='darkorange')

best_g = gammas[np.argmin(rmse_vals)]
best_r = min(rmse_vals)
ax.scatter(best_g, best_r, s=120, color='red', zorder=5,
           label=f'Mejor γ = {best_g}  (RMSE = ${best_r:,.0f})')

ax.set_title('Efecto del parámetro γ sobre el RMSE de prueba\n'
             '(γ controla la poda de nodos — Ecuación 11 del material)',
             fontweight='bold')
ax.set_xlabel('γ (gamma)')
ax.set_ylabel('RMSE (USD)')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))
ax.legend()
plt.tight_layout()
plt.show()

# ─── Predicción sobre ejemplos nuevos ─────────────────────────────────────────
# Aquí los alumnos pueden cambiar los valores y ver cómo varía la predicción

nuevos_phones = pd.DataFrame({
    'ram_gb'        : [4,    8,    12,   16],
    'battery_mah'   : [4000, 4500, 5000, 5000],
    'screen_inches' : [6.1,  6.5,  6.7,  6.8],
    'weight_g'      : [170,  190,  205,  220],
    'front_cam_mp'  : [8,    12,   32,   50],
    'back_cam_mp'   : [48,   108,  200,  200],
    'launched_year' : [2023, 2024, 2024, 2025],
    'brand_encoded' : [brand_map.get('Samsung', 0),
                       brand_map.get('Samsung', 0),
                       brand_map.get('Apple',   0),
                       brand_map.get('Apple',   0)]
})

predicciones = modelo_final.predict(nuevos_phones)

print('📱 Predicciones de precio para smartphones nuevos:\n')
etiquetas = ['Samsung (gama media)', 'Samsung (gama alta)',
             'Apple (gama alta)',    'Apple (flagship)']
for label, pred in zip(etiquetas, predicciones):
    print(f'   {label:25s} → Precio estimado: ${pred:,.2f} USD')
