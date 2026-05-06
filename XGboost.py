## XGBoost Aplicado: Predicción de Precio de Smartphones ##

## Importar las bibliotecas necesarias

### Librerías para manipulación de datos

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
form sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print('✅ Todas las librerías cargadas correctamente')
print(f'   XGBoost versión: {xgb.__version__}')

## Cargar el dataset

df_raw = pd.read_csv('databases/Mobiles_Dataset.csv', encoding = 'latin-1')

print(f'📐 Dimensiones del dataset: {df_raw.shape[0]} filas × {df_raw.shape[1]} columnas')
print(f'\n📋 Variables disponibles:')
for col in df_raw.columns:
    print(f'   • {col} ({df_raw[col].dtype})')

df_raw.head(8)    

