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

# Cargando el dataset

df = pd.read_csv('databases/healthcare_fraud_detection.csv')

print('=' * 55)
print('       RESUMEN DEL DATASET')
print('=' * 55)
print(f'  Registros totales : {len(df):,}')
print(f'  Variables         : {df.shape[1]}')
print(f'  Fraudes           : {df["Is_Fraud"].sum():,} ({df["Is_Fraud"].mean()*100:.1f}%)')
print(f'  Legítimas         : {(df["Is_Fraud"]==0).sum():,} ({(df["Is_Fraud"]==0).mean()*100:.1f}%)')
print(f'  Valores faltantes : {df.isnull().sum().sum()}')
print('=' * 55)

df.head(3)

# Preprocesamiento de los datos

# ── 1. Copiar y manejar nulos 

df_proc = df.copy()
df_proc['Insurance_Type'].fillna('Unknown', inplace=True)
df_proc['Provider_Specialty'].fillna('Unknown', inplace=True)
df_proc['Prior_Visits_12m'].fillna(df_proc['Prior_Visits_12m'].median(), inplace=True)
print(f'  ✅ Valores nulos imputados: {df.isnull().sum().sum()} → {df_proc.isnull().sum().sum()}')

# ── 2. Codificación de variables categóricas

df_proc['Claim_Date']        = pd.to_datetime(df_proc['Claim_Submission_Date'])
df_proc['Claim_Month']       = df_proc['Claim_Date'].dt.month
df_proc['Claim_DayOfWeek']   = df_proc['Claim_Date'].dt.dayofweek
df_proc['Approval_Ratio']    = df_proc['Approved_Amount'] / (df_proc['Claim_Amount'] + 1e-6)
df_proc['Claim_Per_Stay']    = df_proc['Claim_Amount'] / (df_proc['Length_of_Stay'] + 1)
print('  ✅ Nuevas variables creadas: Claim_Month, Approval_Ratio, Claim_Per_Stay')

# ── 3. Codificación de variables categóricas

cat_cols = ['Patient_Gender', 'Insurance_Type', 'Provider_Specialty',
            'Claim_Status', 'Visit_Type']

le = LabelEncoder()
for col in cat_cols:
    df_proc[col + '_enc'] = le.fit_transform(df_proc[col].astype(str))
print(f'  ✅ Variables categóricas codificadas: {cat_cols}')

# ── 4. Selección de features

FEATURES = [
    'Patient_Age', 'Claim_Amount', 'Approved_Amount',
    'Days_Between_Service_and_Claim', 'Number_of_Claims_Per_Provider_Monthly',
    'Length_of_Stay', 'Prior_Visits_12m', 'Chronic_Condition_Flag',
    'Claim_Month', 'Claim_DayOfWeek', 'Approval_Ratio', 'Claim_Per_Stay',
    'Patient_Gender_enc', 'Insurance_Type_enc', 'Provider_Specialty_enc',
    'Claim_Status_enc', 'Visit_Type_enc'
]

X = df_proc[FEATURES].values
y = df_proc['Is_Fraud'].values
print(f'  ✅ Features seleccionados: {len(FEATURES)} variables')

# ── 5. Escalado de features

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

## ── 6. ESTRATEGIA CLAVE: entrenar SOLO con legítimas ─────────────────────
# Separar fraude y legítimo

X_legit = X_scaled[y == 0]
X_fraud = X_scaled[y == 1]

# Dividir en train/test solo con legítimos

X_train, X_val = train_test_split(X_legit, test_size=0.2, random_state=42)

# Test set: mezcla de legítimas (val) + TODAS las fraudulentas

X_test = np.vstack([X_val, X_fraud])
y_test = np.hstack([np.zeros(len(X_val)), np.ones(len(X_fraud))])

print(f'\n  📐 Dimensiones finales:')
print(f'     Train (solo legítimas) : {X_train.shape}')
print(f'     Validación (legítimas) : {X_val.shape}')
print(f'     Test (legítimas+fraude): {X_test.shape}')
print(f'     Fraudes en test        : {int(y_test.sum())} ({y_test.mean()*100:.1f}%)')
print(f'     Dimensión de entrada   : {X_train.shape[1]} features')

# Construcción del Autoencoder

INPUT_DIM   = X_train.shape[1]   # 17 features
LATENT_DIM  = 8                  # espacio latente

def build_autoencoder(input_dim, latent_dim):
    
    # Encoder
    inputs = keras.Input(shape=(input_dim,), name = 'input')

    x = layers.Dense(64, activation='relu', name='enc_dense_1')(inputs)
    x = layers.BatchNormalization(name='enc_bn_1')(x)
    x = layers.Dropout(0.15, name='enc_dropout_1')(x)

    x = layers.Dense(32, activation='relu', name='enc_dense_2')(x)
    x = layers.BatchNormalization(name='enc_bn_2')(x)
    x = layers.Dropout(0.15, name='enc_dropout_2')(x)

    # Espacio latente (cuello de botella)

    latent = layers.Dense(latent_dim, activation='relu', name='latent')(x)

    # Decoder
    x = layers.Dense(32, activation='relu', name='dec_dense_1')(latent)
    x = layers.BatchNormalization(name='dec_bn_1')(x)
    x = layers.Dropout(0.15, name='dec_dropout_1')(x)

    x = layers.Dense(64, activation='relu', name='dec_dense_2')(x)
    x = layers.BatchNormalization(name='dec_bn_2')(x)

    # Salida: misma dimensión que entrada, activación lineal
    outputs = layers.Dense(input_dim, activation='linear', name='output')(x)

    # Modelo Autoencoder
    model = keras.Model(inputs=inputs, outputs=outputs, name='Autoencoder_FraudDetection')
    return model

    
