## Caso de Estudio: DBSCAN aplicado a Mercados Financieros Globalet ##

## Importar las bibliotecas necesarias

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as pe
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.collections as mcoll
import seaborn as sns
import warnings
from scipy.stats import gaussian_kde
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score, davies_bouldin_score, silhouette_samples
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

warnings.filterwarnings('ignore')

# ── Paleta premium (fondo blanco, colores ricos) ────────────
PALETTE = {
    'Stock Index':    '#1A237E',   # azul índigo profundo
    'Cryptocurrency': '#B71C1C',   # rojo oscuro
    'Commodity':      '#1B5E20',   # verde bosque
    'Currency':       '#4A148C',   # morado ciruela
}
CLUSTER_COLORS = {
    0:  '#0D47A1',
    1:  '#1B5E20',
    2:  '#E65100',
    3:  '#4A148C',
    4:  '#006064',
    -1: '#C62828',
}
ACCENT   = '#263238'
SOFT_BG  = '#F8F9FA'
GRID_COL = '#E0E0E0'
TEXT_MAIN= '#212121'

## Cargar el dataset

df_raw = pd.read_csv('databases/global_financial_markets_2000_Now.csv')
df_raw['date'] = pd.to_datetime(df_raw['date'])
df_raw = df_raw.sort_values(['date', 'asset_name'])

print(f"\n  Registros:   {len(df_raw):,}")
print(f"  Activos:     {df_raw['asset_name'].nunique()}")
print(f"  Período:     {df_raw['date'].min().date()} → {df_raw['date'].max().date()}")
print(f"  Columnas:    {list(df_raw.columns)}")
print(f"  Nulos:       {df_raw.isnull().sum().sum()}")

# Retornos logarítmicos
df_raw['ret'] = df_raw.groupby('asset_name')['close'].transform(
    lambda x: np.log(x / x.shift(1))
)

## Feature Engineering

records = []
for activo, g in df_raw.groupby('asset_name'):
    ret   = g['ret'].dropna()
    if len(ret) < 100:
        continue
    rend  = ret.mean() * 252
    vol   = ret.std()  * np.sqrt(252)
    cum   = (1 + ret).cumprod()
    dd    = ((cum - cum.cummax()) / cum.cummax()).min()
    rf    = 0.02 / 252
    sharpe = (ret.mean() - rf) / ret.std() * np.sqrt(252) if ret.std() > 0 else 0
    calmar = rend / abs(dd) if dd != 0 else 0
    skew  = float(ret.skew())
    kurt  = float(ret.kurtosis())
    var95 = float(np.percentile(ret, 5))
    records.append(dict(
        activo=activo, tipo=g['asset_type'].iloc[0],
        rend=rend, vol=vol, dd=dd,
        sharpe=sharpe, calmar=calmar,
        skew=skew, kurt=kurt, var95=var95
    ))

df_f = pd.DataFrame(records).set_index('activo')
print(f"  ✅  {len(df_f)} activos con características listas.\n")
print(df_f[['tipo','rend','vol','dd','sharpe','calmar']].round(3).to_string())

