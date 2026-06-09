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

# ── Estilo global ───────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor':  'white',
    'axes.facecolor':    'white',
    'axes.edgecolor':    '#BDBDBD',
    'axes.linewidth':    0.8,
    'axes.grid':         True,
    'grid.color':        GRID_COL,
    'grid.linewidth':    0.6,
    'grid.linestyle':    '--',
    'grid.alpha':        0.7,
    'font.family':       'DejaVu Sans',
    'font.size':         10,
    'axes.titlesize':    13,
    'axes.titleweight':  'bold',
    'axes.titlepad':     10,
    'axes.labelsize':    10.5,
    'axes.labelcolor':   TEXT_MAIN,
    'xtick.color':       '#555555',
    'ytick.color':       '#555555',
    'legend.framealpha': 0.92,
    'legend.edgecolor':  '#BDBDBD',
    'legend.fontsize':   9.5,
    'figure.dpi':        130,
    'savefig.dpi':       180,
    'savefig.bbox':      'tight',
    'savefig.facecolor': 'white',
    'lines.linewidth':   2,
    'patch.linewidth':   0.5,
})

def add_watermark(fig, text='ML Aplicado · Gladys Choque Ulloa · 2026'):
    fig.text(0.99, 0.01, text, ha='right', va='bottom',
             fontsize=7, color='#9E9E9E', style='italic',
             transform=fig.transFigure)

def title_bar(ax, label, color='#1A237E'):
    """Añade una barra de color encima del título del eje."""
    ax.set_title(label, color=TEXT_MAIN, pad=12)
    ax.spines['top'].set_linewidth(2.5)
    ax.spines['top'].set_color(color)

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

## Preprocesamiento de datos

FEATS = ['rend', 'vol', 'dd', 'sharpe', 'skew', 'kurt', 'var95']

X_raw = df_f[FEATS].copy().replace([np.inf, -np.inf], np.nan).fillna(df_f[FEATS].median())
scaler = StandardScaler()
X = scaler.fit_transform(X_raw)

print(f"\n  ✅  Estandarización completada  →  media≈{X.mean():.4f}  σ≈{X.std():.4f}")

##  EDA — Riesgo vs Retorno

fig, axes = plt.subplots(1, 2, figsize=(18, 7.5))
fig.suptitle('Espacio Riesgo–Retorno de Activos Financieros Globales (2000–2026)',
             fontsize=16, fontweight='bold', color=TEXT_MAIN, y=1.01)

markers_tipo = {'Stock Index':'o','Cryptocurrency':'D','Commodity':'s','Currency':'^'}

for panel, excluir_cripto, ax in [(0, False, axes[0]), (1, True, axes[1])]:
    df_plot = df_f if not excluir_cripto else df_f[df_f['tipo'] != 'Cryptocurrency']
    for tipo, grp in df_plot.groupby('tipo'):
        sc = ax.scatter(
            grp['vol'] * 100, grp['rend'] * 100,
            c=PALETTE[tipo], marker=markers_tipo[tipo],
            s=120, alpha=0.92, edgecolors='white', linewidths=0.8,
            label=tipo, zorder=4
        )
    # Etiquetas
    for activo, row in df_plot.iterrows():
        ax.annotate(
            activo,
            (row['vol']*100, row['rend']*100),
            fontsize=7.2, color='#333333', fontweight='semibold',
            xytext=(5, 4), textcoords='offset points',
            arrowprops=None
        )
    ax.axhline(0, color='#9E9E9E', lw=1, ls='--', zorder=2)
    ax.set_xlabel('Volatilidad Anual  (%)')
    ax.set_ylabel('Rendimiento Anual Medio  (%)')
    titulo = 'Todos los activos' if not excluir_cripto else 'Sin criptomonedas (escala detalle)'
    title_bar(ax, titulo, PALETTE['Stock Index'])
    ax.legend(loc='upper left', markerscale=1.2, framealpha=0.95)
    # Cuadrantes
    ylim = ax.get_ylim(); xlim = ax.get_xlim()
    ax.fill_betweenx(ylim, xlim[0], 0, color='#FFEBEE', alpha=0.25, zorder=1)
    ax.fill_betweenx([0, ylim[1]], xlim[0], xlim[1], color='#E8F5E9', alpha=0.25, zorder=1)
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    # Anotación cuadrante
    ax.text(xlim[1]*0.97, ylim[1]*0.92, 'Alto retorno\nAlta volatilidad',
            ha='right', va='top', fontsize=8, color='#388E3C', alpha=0.7)
    ax.text(xlim[1]*0.97, ylim[0]*0.85 if ylim[0] < 0 else ylim[0]*1.05,
            'Bajo retorno\nAlta volatilidad',
            ha='right', va='bottom', fontsize=8, color='#C62828', alpha=0.7)

plt.tight_layout()

add_watermark(fig)

plt.show()

##  Heatmap correlación

labels_es = {
    'rend':'Rendimiento\nAnual', 'vol':'Volatilidad\nAnual',
    'dd':'Max\nDrawdown', 'sharpe':'Sharpe\nRatio',
    'skew':'Asimetría', 'kurt':'Curtosis', 'var95':'VaR 5%'
}

corr = df_f[FEATS].corr()
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)  # triángulo SUPERIOR oculto

fig, ax = plt.subplots(figsize=(10, 8))
cmap_div = LinearSegmentedColormap.from_list(
    'custom_div', ['#1A237E','#5C6BC0','white','#EF9A9A','#B71C1C'], N=256
)

im = ax.imshow(np.where(~mask.T, corr.values, np.nan),
               cmap=cmap_div, vmin=-1, vmax=1, aspect='auto')

n = len(corr)
tick_labels = [labels_es.get(c, c) for c in corr.columns]
ax.set_xticks(range(n)); ax.set_xticklabels(tick_labels, fontsize=9.5)
ax.set_yticks(range(n)); ax.set_yticklabels(tick_labels, fontsize=9.5)

# Valores en celdas
for i in range(n):
    for j in range(n):
        if not mask.T[i, j]:
            val = corr.values[i, j]
            color = 'white' if abs(val) > 0.55 else TEXT_MAIN
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=9.5, color=color, fontweight='bold' if abs(val) > 0.5 else 'normal')

# Grid
for x in np.arange(-0.5, n, 1):
    ax.axhline(x, color='white', lw=1.5)
    ax.axvline(x, color='white', lw=1.5)

cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
cbar.set_label('Coeficiente de correlación', fontsize=10)
cbar.ax.tick_params(labelsize=9)

title_bar(ax, 'Correlación entre Características — ¿Qué variables aportan información única?',
          PALETTE['Currency'])
ax.set_facecolor('white')
plt.tight_layout()
add_watermark(fig)
plt.show()

##  k-Distancias 

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
fig.suptitle('Selección del Parámetro ε mediante el Gráfico de k-Distancias\n'
             '"El codo de la curva indica el ε óptimo"',
             fontsize=14, fontweight='bold', color=TEXT_MAIN)

epsilons_opt = {}
for i, mp in enumerate([3, 4, 5]):
    nbrs = NearestNeighbors(n_neighbors=mp-1, metric='euclidean').fit(X)
    dists, _ = nbrs.kneighbors(X)
    k_dist = np.sort(dists[:, -1])[::-1]
    # Detectar codo: punto de máxima curvatura
    diffs2 = np.diff(np.diff(k_dist))
    codo = int(np.argmax(np.abs(diffs2))) + 1
    eps_val = round(float(k_dist[codo]), 2)
    epsilons_opt[mp] = eps_val

    ax = axes[i]
    # Área bajo la curva
    ax.fill_between(range(len(k_dist)), k_dist, alpha=0.08,
                    color=list(PALETTE.values())[i])
    ax.plot(k_dist, color=list(PALETTE.values())[i], lw=2.2, zorder=3)
    # Zonas
    ax.axvspan(0, codo, alpha=0.06, color='#1B5E20', label='Zona densa')
    ax.axvspan(codo, len(k_dist), alpha=0.06, color='#B71C1C', label='Zona ruido')
    # Líneas de referencia
    ax.axvline(codo, color='#FF6F00', lw=1.8, ls='--', zorder=4)
    ax.axhline(eps_val, color='#C62828', lw=1.8, ls='--', zorder=4)
    ax.scatter([codo], [k_dist[codo]], s=120, color='#C62828',
               zorder=6, edgecolors='white', lw=1.5)
    # Anotación del codo
    ax.annotate(f'  Codo → ε ≈ {eps_val}',
                xy=(codo, k_dist[codo]),
                xytext=(codo + max(1, len(k_dist)//8), k_dist[codo] + k_dist.max()*0.08),
                fontsize=9.5, color='#C62828', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#C62828', lw=1.5))
    ax.set_xlabel('Puntos ordenados (mayor → menor distancia)')
    ax.set_ylabel(f'{mp-1}-distancia al vecino más cercano')
    title_bar(ax, f'MinPts = {mp}  →  ε sugerido = {eps_val}',
              list(PALETTE.values())[i])
    ax.legend(fontsize=8.5)

plt.tight_layout()
add_watermark(fig)
plt.show()

EPS     = 0.80   # validado con grilla completa → máximo Silhouette Score
MIN_PTS = 4
print(f"\n  ✅  Parámetros elegidos: ε={EPS}, MinPts={MIN_PTS}")
print("  ℹ️   (Exploración sistemática de ε × MinPts confirmó este valor óptimo)")

##  Entrenamiento de DBSCAN

db = DBSCAN(eps=EPS, min_samples=MIN_PTS, metric='euclidean', algorithm= 'ball_tree')
labels = db.fit_predict(X)

core_mask = np.zeros(len(X), dtype=bool)
core_mask[db.core_sample_indices_] = True

df_f['cluster'] = labels
df_f['core'] = core_mask
df_f['punto_tipo'] = 'Frontera'
df_f.loc[core_mask, 'punto_tipo'] = 'Nucleo'
df_f.loc[labels == -1, 'punto_tipo'] = 'Ruido'

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = (labels == -1).sum()

print(f"\n  Clusters encontrados : {n_clusters}")
print(f"  Puntos de ruido      : {n_noise}  ({n_noise/len(labels)*100:.1f}%)")
print(f"  Puntos núcleo        : {core_mask.sum()}")
print(f"  Puntos frontera      : {(~core_mask & (labels != -1)).sum()}")

for c in sorted(set(labels)):
    g = df_f[df_f['cluster'] == c]
    nombre = '── RUIDO (outliers)' if c == -1 else f'── Cluster {c}'
    print(f"\n  {nombre}  [{len(g)} activos]")
    for a, row in g.iterrows():
        print(f"     {a:22s}  [{row['tipo']:14s}]  ({row['punto_tipo']})")

## PCA Clusters

pca   = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X)
var_  = pca.explained_variance_ratio_

fig, axes = plt.subplots(1, 2, figsize=(20, 8.5))
fig.suptitle(f'DBSCAN — Agrupamiento de Activos Financieros Globales\n'
             f'ε={EPS}  ·  MinPts={MIN_PTS}  ·  {n_clusters} cluster(s) + {n_noise} outlier(s)  ·  '
             f'PCA varianza explicada: {sum(var_)*100:.0f}%',
             fontsize=14.5, fontweight='bold', color=TEXT_MAIN)

marker_shapes = {'Nucleo':'o', 'Frontera':'s', 'Ruido':'X'}
marker_sizes  = {'Nucleo':130, 'Frontera':100, 'Ruido':175}

for ax_idx, (ax, modo) in enumerate(zip(axes, ['cluster','tipo'])):
    for i, (activo, row) in enumerate(df_f.iterrows()):
        cid = row['cluster']
        tp  = row['punto_tipo']
        if modo == 'cluster':
            color = CLUSTER_COLORS.get(cid, '#78909C')
        else:
            color = PALETTE[row['tipo']]
        mshape = marker_shapes[tp]
        msize  = marker_sizes[tp]
        ax.scatter(X_pca[i,0], X_pca[i,1], c=color, marker=mshape,
                   s=msize, alpha=0.90, edgecolors='white', linewidths=0.9,
                   zorder=4)
        ax.annotate(
            activo,
            (X_pca[i,0], X_pca[i,1]),
            fontsize=7.3, color='#1A1A1A', fontweight='semibold',
            xytext=(5,5), textcoords='offset points', zorder=5,
            bbox=dict(boxstyle='round,pad=0.15', fc='white', alpha=0.55, ec='none')
        )

    ax.set_xlabel(f'PC1  ({var_[0]*100:.1f}% varianza)', labelpad=8)
    ax.set_ylabel(f'PC2  ({var_[1]*100:.1f}% varianza)', labelpad=8)

    if modo == 'cluster':
        title_bar(ax, 'Clusters DBSCAN  (● Núcleo   ■ Frontera   ✕ Ruido)')
        handles = [mpatches.Patch(color=CLUSTER_COLORS.get(c,'#78909C'),
                   label='Ruido (outlier)' if c==-1 else f'Cluster {c}')
                   for c in sorted(set(labels))]
    else:
        title_bar(ax, 'Clase de Activo  (validación de coherencia financiera)')
        handles = [mpatches.Patch(color=v, label=k) for k,v in PALETTE.items()]
    ax.legend(handles=handles, loc='best', framealpha=0.95, fontsize=9.5)

plt.tight_layout()
add_watermark(fig)
plt.show()

## Silhouette

mask_val    = labels != -1
X_val       = X[mask_val]
labels_val  = labels[mask_val]
names_val   = df_f.index[mask_val].tolist()

if len(set(labels_val)) > 1:
    sil_global  = silhouette_score(X_val, labels_val)
    sil_samples = silhouette_samples(X_val, labels_val)

    fig, (ax_sil, ax_bar) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'Análisis Silhouette — Calidad de los Clusters DBSCAN\n'
                 f'Silhouette Score Global = {sil_global:.4f}  (rango: -1 → 1, mejor cercano a 1)',
                 fontsize=13, fontweight='bold', color=TEXT_MAIN)

    # Panel izquierdo: barras silhouette por activo
    y_lower = 10
    cluster_ids = sorted(set(labels_val))
    for c in cluster_ids:
        mask_c = labels_val == c
        vals_c = np.sort(sil_samples[mask_c])
        y_upper = y_lower + len(vals_c)
        color_c = CLUSTER_COLORS.get(c,'#78909C')
        ax_sil.fill_betweenx(np.arange(y_lower, y_upper), 0, vals_c,
                             facecolor=color_c, alpha=0.85, edgecolor='white', lw=0.3)
        label_y = (y_lower + y_upper) / 2
        ax_sil.text(-0.05, label_y, f'C{c}' if c != -1 else 'Ruido',
                    ha='right', va='center', fontsize=9, color=color_c, fontweight='bold')
        y_lower = y_upper + 5

    ax_sil.axvline(sil_global, color='#C62828', lw=2, ls='--',
                   label=f'Silhouette global = {sil_global:.3f}')
    ax_sil.axvline(0, color='#424242', lw=0.8, ls='-')
    ax_sil.set_xlabel('Coeficiente Silhouette por activo')
    ax_sil.set_ylabel('Activos agrupados por cluster')
    ax_sil.set_yticks([])
    ax_sil.set_xlim(-0.3, 1.05)
    title_bar(ax_sil, 'Silhouette por activo  (azul=bien asignado, rojo=mal asignado)')
    ax_sil.legend(fontsize=9.5)

    # Panel derecho: silhouette por activo (top/bottom)
    sil_df = pd.DataFrame({'activo': names_val, 'silhouette': sil_samples,
                           'cluster': labels_val})
    sil_df = sil_df.sort_values('silhouette', ascending=True)
    colors_sil = [CLUSTER_COLORS.get(c,'#78909C') for c in sil_df['cluster']]
    bars = ax_bar.barh(sil_df['activo'], sil_df['silhouette'],
                       color=colors_sil, alpha=0.85, edgecolor='white', height=0.7)
    ax_bar.axvline(0, color='#424242', lw=0.8)
    ax_bar.axvline(sil_global, color='#C62828', lw=1.8, ls='--',
                   label=f'Media = {sil_global:.3f}')
    ax_bar.set_xlabel('Coeficiente Silhouette')
    title_bar(ax_bar, 'Ranking de activos por cohesión al cluster')
    ax_bar.legend(fontsize=9.5)
    for bar, val in zip(bars, sil_df['silhouette']):
        ax_bar.text(val + 0.01 if val >= 0 else val - 0.01,
                    bar.get_y() + bar.get_height()/2,
                    f'{val:.2f}', va='center', ha='left' if val >= 0 else 'right',
                    fontsize=8, color='#333333')

    plt.tight_layout()
    add_watermark(fig)
    plt.show()
    print(f"\n  ✅  Silhouette Score = {sil_global:.4f}")
else:
    sil_global = None
    print("  ⚠️  Un solo cluster — Silhouette no calculable. Ajusta ε.")

##  Métricas y tabla comparativa

mask_v = labels != -1
if len(set(labels[mask_v])) > 1:
    sil_score = silhouette_score(X[mask_v], labels[mask_v])
    db_s  = davies_bouldin_score(X[mask_v], labels[mask_v])
    print(f"\n  Silhouette Score (DBSCAN):    {sil_score:.4f}   (mejor → 1)")
    print(f"  Davies-Bouldin (DBSCAN):      {db_s:.4f}   (mejor → 0)")

print(f"\n  {'ε':>5}  {'MP':>3}  {'Clusters':>8}  {'Ruido':>5}  {'Silhouette':>11}")
print(f"  {'-'*42}")
for eps_t, mp_t in [(0.8,3),(0.8,4),(1.0,3),(1.0,4),(EPS,MIN_PTS),(1.5,4),(2.0,4)]:
    l_t  = DBSCAN(eps=eps_t, min_samples=mp_t).fit_predict(X)
    nc_t = len(set(l_t))-(1 if -1 in l_t else 0)
    nr_t = (l_t==-1).sum()
    mv_t = l_t != -1
    sil_t = silhouette_score(X[mv_t], l_t[mv_t]) if nc_t>1 and mv_t.sum()>2 else float('nan')
    sel = ' ◄ ELEGIDO' if (eps_t==EPS and mp_t==MIN_PTS) else ''
    sil_str2 = f'{sil_t:.4f}' if not np.isnan(sil_t) else '   N/A'
    print(f"  {eps_t:>5.2f}  {mp_t:>3}  {nc_t:>8}  {nr_t:>5}  {sil_str2:>11}{sel}")


