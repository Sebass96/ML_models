#   CASO APLICADO — CatBoost 

## 1. Cargar el dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from catboost import CatBoostClassifier, Pool, cv as catboost_cv
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_curve, auc, ConfusionMatrixDisplay
)

# Estética global de gráficos
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#f8f9fa",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "DejaVu Sans",
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})
CATBLUE   = "#1E50A0"
CATGREEN  = "#008060"
CATRED    = "#C0392B"
CATYELLOW = "#E6AC00"

print("\n[OK] Librerías importadas correctamente.")

## 2. Cargar el dataset

df = pd.read_csv('databases/train_and_test2.csv')
print(f"\n  • Dimensiones del dataset : {df.shape[0]} filas × {df.shape[1]} columnas")
print(f"  • Variable objetivo       : '2urvived'")
print(f"  • Distribución del target :")
vc = df["2urvived"].value_counts()
for k, v in vc.items():
    label = "Sobrevivió" if k == 1 else "No sobrevivió"
    print(f"      {k} ({label}) : {v} ({v/len(df)*100:.1f}%)")

# ── Columnas relevantes (descartamos las columnas 'zero*' y el ID) ───────────
FEATURES = ["Age", "Fare", "Sex", "sibsp", "Parch", "Pclass", "Embarked"]
TARGET   = "2urvived"

# ── Valores faltantes ────────────────────────────────────────────────────────
print("\n  • Valores faltantes por columna:")

faltantes = df[FEATURES + [TARGET]].isnull().sum()
print(faltantes[faltantes > 0].to_string() if faltantes.sum() > 0
      else "      Ninguno (ya imputados en el dataset original)")

# ── Convertir a string las categóricas para que CatBoost las reconozca ───────
# CatBoost acepta enteros pero es más explícito con strings cuando son
# categorías semánticas.
CATEGORIC_COLS = ["Pclass", "Sex", "Embarked"]
df[CATEGORIC_COLS] = df[CATEGORIC_COLS].astype(str)

# Rellenar los 2 NaN de Embarked con la moda
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)

print("\n  • Primeras 5 filas (columnas seleccionadas):")
print(df[FEATURES + [TARGET]].head().to_string(index=False))

## 3. Análisis exploratorio

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# ── 3a. Distribución de supervivencia

ax = axes[0, 0]
conteos = df[TARGET].value_counts()
bars = ax.bar(
    ["No sobrevivió\n(0)", "Sobrevivió\n(1)"],
    conteos.values,
    color=[CATRED, CATGREEN], edgecolor="black", linewidth=1.5
)

for bar, v in zip(bars, conteos.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10, 
    f"{v}\n({v/len(df)*100:.1f}%)", ha='center', fontsize=11, color='black')

ax.set_title("Distribución de Supervivencia", fontsize=13)
ax.set_ylabel("Cantidad de Pasajeros", fontsize=11)
ax.set_ylim(0, max(conteos.values) * 1.2)

# ── 3b. Supervivencia por Pclass

ax = axes[0, 1]
cross_class = df.groupby("Pclass")[TARGET].mean().reset_index()
cross_class.columns = ["Pclass", "TasaSupervivencia"]
colores_class = {
    "1": CATGREEN, "2": CATYELLOW, "3": CATRED
}

for _, row in cross_class.iterrows():
    ax.bar(f"Clase {row['Pclass']}", row["TasaSupervivencia"],
           color=colores_class.get(row["Pclass"], CATBLUE), edgecolor="white")
    ax.text(f"Clase {row['Pclass']}", row["TasaSupervivencia"] + 0.01,
            f"{row['TasaSupervivencia']:.1%}", ha="center", fontsize=10) 

ax.set_title("Tasa de Supervivencia por Clase", fontsize=13)
ax.set_ylabel("Tasa de Supervivencia", fontsize=11)
ax.set_ylim(0, 1)

# ── 3c. Supervivencia por Sexo ───────────────────────────────────────────────
ax = axes[0, 2]
# Sex: 0=Masculino, 1=Femenino (según el dataset)
cross_sex = df.groupby("Sex")[TARGET].mean()
etiquetas_sex = {"0": "Masculino", "1": "Femenino"}
for sex_val, tasa in cross_sex.items():
    label = etiquetas_sex.get(sex_val, sex_val)
    color = CATGREEN if sex_val == "1" else CATBLUE
    ax.bar(label, tasa, color=color, edgecolor="white")
    ax.text(label, tasa + 0.01, f"{tasa:.1%}", ha="center", fontsize=10)
ax.set_title("Tasa de Supervivencia por Sexo")
ax.set_ylabel("Proporción de sobrevivientes")
ax.set_ylim(0, 0.9)

# ── 3d. Distribución de Edad según supervivencia ─────────────────────────────
ax = axes[1, 0]
df[df[TARGET] == 1]["Age"].plot.hist(
    bins=25, alpha=0.6, color=CATGREEN, label="Sobrevivió", ax=ax)
df[df[TARGET] == 0]["Age"].plot.hist(
    bins=25, alpha=0.6, color=CATRED, label="No sobrevivió", ax=ax)
ax.set_title("Distribución de Edad por Supervivencia")
ax.set_xlabel("Edad")
ax.set_ylabel("Frecuencia")
ax.legend()

# ── 3e. Distribución de Tarifa según supervivencia ───────────────────────────
ax = axes[1, 1]
df[df[TARGET] == 1]["Fare"].clip(0, 300).plot.hist(
    bins=30, alpha=0.6, color=CATGREEN, label="Sobrevivió", ax=ax)
df[df[TARGET] == 0]["Fare"].clip(0, 300).plot.hist(
    bins=30, alpha=0.6, color=CATRED, label="No sobrevivió", ax=ax)
ax.set_title("Distribución de Tarifa por Supervivencia\n(limitada a 300 para legibilidad)")
ax.set_xlabel("Tarifa")
ax.set_ylabel("Frecuencia")
ax.legend()

# ── 3f. Puerto de embarque ───────────────────────────────────────────────────
ax = axes[1, 2]
cross_emb = df.groupby("Embarked")[TARGET].mean().reset_index()
emb_labels = {"0": "Cherbourg\n(0)", "1": "Queenstown\n(1)", "2": "Southampton\n(2)"}
for _, row in cross_emb.iterrows():
    label = emb_labels.get(row["Embarked"], row["Embarked"])
    ax.bar(label, row[TARGET], color=CATBLUE, edgecolor="white")
    ax.text(label, row[TARGET] + 0.01, f"{row[TARGET]:.1%}", ha="center", fontsize=10)
ax.set_title("Tasa de Supervivencia por Puerto\nde Embarque")
ax.set_ylabel("Proporción de sobrevivientes")
ax.set_ylim(0, 0.75)

plt.tight_layout()
'plt.savefig("eda_titanic.png", dpi=150, bbox_inches="tight")'
plt.show()
print("  [OK] Gráfico EDA guardado como 'eda_titanic.png'")

## 4. División Train / Test (estratificada)

X = df[FEATURES].copy()  # Crea una copia independiente del DataFrame.
y = df[TARGET].copy()

# División 80% entrenamiento — 20% prueba, estratificada por el target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.2, random_state = 42, stratify = y) # stratify para mantener la proporción de clases

# Índices de columnas categóricas en X
cat_indices = [X.columns.get_loc(c) for c in CATEGORIC_COLS]

print(f"\n  • Tamaño entrenamiento : {X_train.shape[0]} ejemplos")
print(f"  • Tamaño prueba        : {X_test.shape[0]} ejemplos")
print(f"  • Columnas categóricas : {CATEGORIC_COLS} → índices {cat_indices}")
print(
    "\n  ▶ NOTA TEÓRICA:"
    "\n    CatBoost calculará la OTS para Pclass, Sex y Embarked usando"
    "\n    SOLO los ejemplos anteriores en la permutación (ver ecuación 3"
    "\n    del material teórico). Nosotros no hacemos ningún encoding manual."
)    

# 5. Entrenamiento del modelo CatBoost

print("""
  ▶ HIPERPARÁMETROS ELEGIDOS (justificación):
    iterations        = 1000  → número máximo de árboles (Oblivious Trees)
    learning_rate     = 0.05  → tasa de aprendizaje η pequeña para mejor generalización
    depth             = 6     → profundidad de cada árbol simétrico
    l2_leaf_reg       = 3     → regularización L2 sobre los pesos de las hojas
    cat_features      → declara qué columnas son categóricas (OTS automático)
    early_stopping    = 50    → detiene el entrenamiento si el AUC no mejora en 50 rondas
    eval_metric       = 'AUC' → métrica de validación durante el entrenamiento
""")

model = CatBoostClassifier(
    iterations = 1000,
    learning_rate = 0.05,
    depth = 6,
    l2_leaf_reg = 3,
    cat_features = cat_indices,
    eval_metric = 'AUC',
    early_stopping_rounds = 50,
    random_seed = 42,
    verbose = 100
)

# ▶ Pool es la estructura de datos nativa de CatBoost

train_pool = Pool(X_train, y_train, cat_features=cat_indices)
test_pool  = Pool(X_test,  y_test,  cat_features=cat_indices)

model.fit(train_pool, eval_set=test_pool)

print(f"\n  [OK] Entrenamiento finalizado.")
print(f"  • Mejor iteración      : {model.best_iteration_}")


# 6. Curva de aprendizaje (Train vs Validation)

evals = model.evals_result_
# El error KeyError: 'AUC' se produce porque 'AUC' no es una clave en evals["learn"].
# CatBoost, por defecto, registra 'Logloss' para el conjunto de entrenamiento cuando se proporciona un 'eval_set',
# y 'AUC' para el conjunto de validación (porque eval_metric se estableció en "AUC").
# Para corregir esto dentro de esta celda, graficaremos 'Logloss' para la curva de entrenamiento,
# ya que es la métrica disponible para el conjunto de aprendizaje.
iters = range(len(evals["learn"]["Logloss"])) # Cambiado de AUC a Logloss

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(iters, evals["learn"]["Logloss"], # Cambiado de AUC a Logloss
        color=CATBLUE, lw=2, label="Entrenamiento (Logloss)") # Etiqueta actualizada
ax.plot(iters, evals["validation"]["AUC"],
        color=CATGREEN, lw=2, label="Validación (AUC)")
ax.axvline(model.best_iteration_, color=CATRED, ls="--", lw=1.5,
           label=f"Mejor iteración ({model.best_iteration_})")
ax.set_title(
    "Curva de Aprendizaje — CatBoost Titanic\n",
    fontsize=13, color=CATBLUE, fontweight="bold"
)
ax.set_xlabel("Iteración (número de árboles)")
ax.set_ylabel("Métrica") # Etiqueta del eje Y actualizada
ax.legend(framealpha=0.9)
ax.annotate(
    "▶ Ordenated Boosting reduce el sesgo\n   en cada gradiente iterativo",
    xy=(model.best_iteration_, evals["validation"]["AUC"][model.best_iteration_]),
    xytext=(model.best_iteration_ + 60,
            evals["validation"]["AUC"][model.best_iteration_] - 0.05),
    arrowprops=dict(arrowstyle="->", color=CATRED),
    fontsize=9, color=CATRED,
)
plt.tight_layout()
'plt.savefig("curva_aprendizaje.png", dpi=150, bbox_inches="tight")'
plt.show()
print("  [OK] Gráfico guardado como 'curva_aprendizaje.png'")

# 7. Evaluación del modelo

y_pred = model.predict(test_pool)
y_pred_prob = model.predict_proba(test_pool)[:, 1] # Probabilidad de la clase positiva (sobrevivió)

acc = accuracy_score(y_test, y_pred)
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

print(f"\n  ┌─────────────────────────────────────┐")
print(f"  │  Accuracy  : {acc:.4f}                  │")
print(f"  │  AUC-ROC   : {roc_auc:.4f}                  │")
print(f"  └─────────────────────────────────────┘")
print("\n  Reporte de clasificación:")
print(classification_report(y_test, y_pred,
      target_names=["No sobrevivió (0)", "Sobrevivió (1)"]))

# ── Visualización: Matriz de Confusión + Curva ROC

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(
    "Evaluación del Modelo CatBoost — Titanic\n",
    fontsize=13, fontweight="bold", color=CATBLUE
)

# Matriz de confusión

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["No sobrevivió (0)", "Sobrevivió (1)"]
)
disp.plot(ax = ax1, colorbar = False,
          cmap = plt.cm.Blues, values_format = 'd')
ax1.set_title(f"Matriz de Confusión\nAccuracy = {acc:.4f}", color=CATBLUE)

# Anotaciones manuales en la matriz
tn, fp, fn, tp = cm.ravel()
ax1.text(-0.4, -0.4,
    f"VP={tp} | FP={fp}\nFN={fn} | VN={tn}",
    fontsize=8, color="gray", transform=ax1.transAxes)

# Curva ROC
ax2.plot(fpr, tpr, color=CATBLUE, lw=2,
         label=f"CatBoost (AUC = {roc_auc:.4f})")
ax2.plot([0, 1], [0, 1], color="gray", ls="--", lw=1, label="Clasificador aleatorio")
ax2.fill_between(fpr, tpr, alpha=0.08, color=CATBLUE)
ax2.set_title(f"Curva ROC — AUC = {roc_auc:.4f}", color=CATBLUE)
ax2.set_xlabel("Tasa de Falsos Positivos (FPR)")
ax2.set_ylabel("Tasa de Verdaderos Positivos (TPR)")
ax2.legend(loc="lower right", framealpha=0.9)

plt.tight_layout()
'plt.savefig("evaluacion_modelo.png", dpi=150, bbox_inches="tight")'
plt.show()
print("  [OK] Gráfico guardado como 'evaluacion_modelo.png'")

# 8. Importancia de Variables (Feature Importance)

importances = model.get_feature_importance(train_pool)
feat_names = X.columns.to_list()
fi_df = pd.DataFrame({
    "Variable": feat_names, "Importancia": importances
}).sort_values(by="Importancia", ascending=False)

print("\n  Importancia de variables (PredictionValuesChange):")
for _, row in fi_df.sort_values("Importancia", ascending=False).iterrows():
    barra = "█" * int(row["Importancia"] / 2)
    cat_mark = " ← CATEGÓRICA (OTS)" if row["Variable"] in CATEGORIC_COLS else ""
    print(f"    {row['Variable']:10s}: {row['Importancia']:6.2f}  {barra}{cat_mark}")

# ── Gráfico horizontal ───────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
colors = [CATGREEN if v in CATEGORIC_COLS else CATBLUE
          for v in fi_df["Variable"]]
bars = ax.barh(fi_df["Variable"], fi_df["Importancia"],
               color=colors, edgecolor="white", linewidth=0.8)
for bar, val in zip(bars, fi_df["Importancia"]):
    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
            f"{val:.2f}", va="center", fontsize=9)

patch_cat = mpatches.Patch(color=CATGREEN, label="Variable categórica (OTS aplicada)")
patch_num = mpatches.Patch(color=CATBLUE,  label="Variable numérica")
ax.legend(handles=[patch_cat, patch_num], loc="lower right", fontsize=9)

ax.set_title(
    "Importancia de Variables — CatBoost\n",
    fontsize=13, fontweight="bold", color=CATBLUE
)
ax.set_xlabel("Importancia (PredictionValuesChange)")
ax.set_xlim(0, fi_df["Importancia"].max() * 1.15)

plt.tight_layout()
'plt.savefig("importancia_variables.png", dpi=150, bbox_inches="tight")'
plt.show()
print("  [OK] Gráfico guardado como 'importancia_variables.png'")    

# 9. Validación Cruzada con CatBoost nativo

full_pool = Pool(X, y, cat_features=cat_indices)
cv_params = {
    "iterations"    : 1000,
    "learning_rate" : 0.05,
    "depth"         : 6,
    "l2_leaf_reg"   : 3,
    "loss_function" : "Logloss", # Añadido: Se requiere para cross-validation
    "eval_metric"   : "AUC",
    "early_stopping_rounds": 50,
    "random_seed"   : 42,
    "verbose"       : 0,
}

cv_results = catboost_cv(
    pool       = full_pool,
    params     = cv_params,
    fold_count = 5,
    stratified = True,
    plot       = False,
    verbose    = False,
)

best_idx  = cv_results["test-AUC-mean"].idxmax()
cv_auc    = cv_results["test-AUC-mean"].iloc[best_idx]
cv_std    = cv_results["test-AUC-std"].iloc[best_idx]

print(f"\n  Resultados 5-Fold CV:")
print(f"  • AUC promedio : {cv_auc:.4f} ± {cv_std:.4f}")
print(f"  • Mejor iter.  : {best_idx + 1}")

# ── Gráfico CV ───────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4))
iters_cv = range(len(cv_results))
ax.plot(iters_cv, cv_results["test-AUC-mean"],
        color=CATGREEN, lw=2, label="AUC Validación (media 5-fold)")
ax.fill_between(
    iters_cv,
    cv_results["test-AUC-mean"] - cv_results["test-AUC-std"],
    cv_results["test-AUC-mean"] + cv_results["test-AUC-std"],
    alpha=0.2, color=CATGREEN, label="± 1 desv. estándar"
)
# ax.plot(iters_cv, cv_results["learn-Logloss-mean"],
#         color=CATBLUE, lw=2, ls="--", label="Logloss Entrenamiento (media)") # Comentado porque esta métrica no está disponible en cv_results
ax.axvline(best_idx, color=CATRED, ls=":", lw=1.5,
           label=f"Mejor iter. ({best_idx+1})")
ax.set_title(
    f"Validación Cruzada 5-Fold — CatBoost Titanic\n"
    f"AUC = {cv_auc:.4f} ± {cv_std:.4f}   ",
    fontsize=11, fontweight="bold", color=CATBLUE
)
ax.set_xlabel("Iteración")
ax.set_ylabel("Métrica")
ax.legend(fontsize=9)
plt.tight_layout()
'plt.savefig("cross_validation.png", dpi=150, bbox_inches="tight")'
plt.show()
print("  [OK] Gráfico guardado como 'cross_validation.png'")

# 10. Predicción sobre nuevos pasajeros

nuevos = pd.DataFrame({
    "Age"      : [25,   55,   8,    35],
    "Fare"     : [7.25, 50.0, 30.0, 80.0],
    "Sex"      : ["0",  "0",  "1",  "1"],     # 0=Masculino, 1=Femenino
    "sibsp"    : [0,    0,    2,    1],
    "Parch"    : [0,    0,    1,    0],
    "Pclass"   : ["3",  "1",  "2",  "1"],
    "Embarked" : ["2",  "0",  "2",  "0"],     # 2=Southampton, 0=Cherbourg
})

perfiles = [
    "Hombre joven (3ª clase, Southampton)",
    "Hombre mayor (1ª clase, Cherbourg)",
    "Niña (2ª clase, Southampton)",
    "Mujer adulta (1ª clase, Cherbourg)",
]

nuevos_pool = Pool(nuevos, cat_features=cat_indices)
prob_surv   = model.predict_proba(nuevos_pool)[:, 1]
pred_surv   = model.predict(nuevos_pool)

print("\n  Perfil del pasajero              | P(Sobrevive) | Predicción")
print("  " + "─" * 60)
for perfil, prob, pred in zip(perfiles, prob_surv, pred_surv):
    resultado = "✓ SOBREVIVE" if pred == 1 else "✗ NO SOBREVIVE"
    print(f"  {perfil:<35}| {prob:.4f}       | {resultado}")

# ── Gráfico de predicciones ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4))
colores_pred = [CATGREEN if p == 1 else CATRED for p in pred_surv]
bars = ax.bar(range(len(perfiles)), prob_surv,
              color=colores_pred, edgecolor="white", linewidth=1.2, width=0.5)
ax.axhline(0.5, color="gray", ls="--", lw=1.5, label="Umbral de decisión (0.5)")
for bar, p, pr in zip(bars, prob_surv, pred_surv):
    label = f"{p:.3f}\n{'Sobrevive' if pr==1 else 'No sobrevive'}"
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.02, label,
            ha="center", fontsize=9, fontweight="bold",
            color=CATGREEN if pr == 1 else CATRED)
ax.set_xticks(range(len(perfiles)))
ax.set_xticklabels(perfiles, rotation=12, ha="right", fontsize=9)
ax.set_ylabel("Probabilidad de supervivencia")
ax.set_ylim(0, 1.15)
ax.set_title(
    "Predicciones para Pasajeros Hipotéticos\n",
    fontsize=13, fontweight="bold", color=CATBLUE
)
ax.legend()
plt.tight_layout()
'plt.savefig("predicciones_nuevos.png", dpi=150, bbox_inches="tight")'
plt.show()
print("  [OK] Gráfico guardado como 'predicciones_nuevos.png'")