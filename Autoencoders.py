## Usando Autoencoders como modelo de Anomaly Detection ##

# Importando as bibliotecas necessárias

# ── Core ────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# ── Visualización ────────────────────────────────────────────────────────────
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Machine Learning ─────────────────────────────────────────────────────────
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, roc_curve, precision_recall_curve,
                              average_precision_score, f1_score)

# ── Deep Learning ────────────────────────────────────────────────────────────
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks

# ── Configuración estética global ────────────────────────────────────────────
PALETTE = {
    'fraud'  : '#e74c3c',   # rojo
    'legit'  : '#2ecc71',   # verde
    'primary': '#2980b9',   # azul
    'dark'   : '#1a252f',   # fondo oscuro
    'accent' : '#f39c12',   # amarillo/naranja
    'purple' : '#8e44ad',
}

plt.rcParams.update({
    'figure.facecolor'  : '#1a252f',
    'axes.facecolor'    : '#212f3d',
    'axes.edgecolor'    : '#34495e',
    'axes.labelcolor'   : '#ecf0f1',
    'axes.titlecolor'   : '#ecf0f1',
    'xtick.color'       : '#bdc3c7',
    'ytick.color'       : '#bdc3c7',
    'text.color'        : '#ecf0f1',
    'grid.color'        : '#2c3e50',
    'grid.linestyle'    : '--',
    'grid.alpha'        : 0.5,
    'font.family'       : 'DejaVu Sans',
    'axes.titlesize'    : 14,
    'axes.labelsize'    : 12,
})

np.random.seed(42)
tf.random.set_seed(42)

print('✅ Librerías importadas correctamente')
print(f'   TensorFlow: {tf.__version__}')
print(f'   Pandas:     {pd.__version__}')
