# Explicacion End-to-End del Proyecto de Clasificacion Binaria de Parkinson

## 1) Objetivo general del sistema

Este proyecto implementa un pipeline completo de Machine Learning para clasificacion binaria:

- Clase positiva: Parkinson
- Clase negativa: No Parkinson (Healthy + otros diagnosticos)

La implementacion es reproducible y cubre:

1. Carga de datos preprocesados
2. Construccion de variables objetivo binarias
3. Ingenieria de caracteristicas (movimiento + cuestionario + demografia)
4. Entrenamiento y validacion con multiples algoritmos
5. Seleccion del mejor modelo
6. Guardado de artefactos para reproduccion e inferencia
7. Generacion automatica de graficos, explainability y reportes

## 2) Vista de datos y fuentes que usa el pipeline

El pipeline trabaja sobre la carpeta preprocesada del dataset PADS:

- Data/pads-parkinsons-disease-smartwatch-dataset-1.0.0/preprocessed

Archivos usados:

- file_list.csv: metadata por sujeto y label original
- movement/NNN_ml.bin: senales de movimiento por sujeto
- questionnaire/NNN_ml.bin: respuestas NMS por sujeto

### 2.1) Volumen de datos

En esta version del dataset:

- Sujetos totales: 469
- Labels originales:
  - 0: 79
  - 1: 276
  - 2: 114

Transformacion a binario:

- y = 1 si label original == 1 (Parkinson)
- y = 0 si label original en {0, 2}

Distribucion binaria final:

- Positivos (Parkinson): 276
- Negativos (No Parkinson): 193

### 2.2) Estructura de datos de movimiento

Cada archivo movement/NNN_ml.bin se reconstruye con:

- 132 canales
- 976 timesteps por canal
- fs = 100 Hz

Estos parametros estan definidos en DatasetConfig:

- movement_channels = 132
- movement_timesteps = 976
- fs_hz = 100.0

### 2.3) Estructura de cuestionario

Cada archivo questionnaire/NNN_ml.bin contiene:

- 30 items (NMS)

Parametro en DatasetConfig:

- questionnaire_items = 30

## 3) Flujo de datos exacto en el codigo

La ruta funcional principal es:

1. scripts/train_model.py
2. src/pd_binary_classifier/training.py -> train_binary_pipeline
3. src/pd_binary_classifier/data.py + src/pd_binary_classifier/features.py

### 3.1) Carga de metadata

load_metadata:

- Lee file_list.csv
- Fuerza id como string y relleno a 3 digitos (001, 002, ...)

### 3.2) Construccion del target

build_binary_target:

- target binario con regla estricta:
  - 1 para Parkinson
  - 0 para el resto

### 3.3) Carga de binarios por sujeto

Por cada subject_id:

- load_movement_array valida tamano esperado y hace reshape a (132, 976)
- load_questionnaire_array valida tamano esperado (30)

Si un archivo no coincide con la dimension esperada, se lanza ValueError para evitar entrenamiento con datos corruptos.

### 3.4) Demografia y variables clinicas

build_demographic_features produce:

- Numericas: age, height, weight, age_at_diagnosis
- Booleanas convertidas a float: appearance_in_kinship, appearance_in_first_grade_kinship
- Categoricas codificadas con one-hot:
  - gender
  - handedness
  - effect_of_alcohol_on_tremor

Manejo de vacios:

- strings vacios o nan string se reemplazan por Unknown en categoricas

## 4) Ingenieria de caracteristicas (feature engineering)

La extraccion de features de movimiento esta en features.py.

### 4.1) Nombres de canales de movimiento

Se construyen 132 canales por combinatoria:

- 11 tasks:
  - Relaxed1, Relaxed2
  - RelaxedTask1, RelaxedTask2
  - StretchHold, HoldWeight, DrinkGlas, CrossArms, TouchNose
  - Entrainment1, Entrainment2
- 2 wrists: LeftWrist, RightWrist
- 6 senales por task/wrist:
  - Accelerometer X,Y,Z
  - Gyroscope X,Y,Z

Total canales = 11 x 2 x 6 = 132

### 4.2) Features por canal

Para cada canal (vector temporal de 976 puntos) se calculan 12 features:

1. mean
2. std
3. median
4. iqr
5. rms
6. energy
7. abs_mean
8. skew
9. kurtosis_excess
10. dom_freq
11. bandpower_3_7hz
12. spectral_entropy

Detalles espectrales:

- FFT real (rfft)
- dom_freq: frecuencia con max potencia excluyendo DC
- bandpower_3_7hz: suma de potencia entre 3 y 7 Hz
- spectral_entropy: entropia de distribucion de potencia normalizada

### 4.3) Features totales del modelo

- Movimiento: 132 x 12 = 1584
- Cuestionario: 30
- Demografia/clinica: numero variable segun one-hot (en esta corrida queda 14)

Total aproximado en esta implementacion: 1628 features.

Nota: el numero exacto de demograficas one-hot puede cambiar si cambian categorias presentes en file_list.csv.

### 4.4) Limpieza numerica

Antes de entrenar:

- np.nan_to_num sobre X completo
- En cada modelo, SimpleImputer(strategy='median')

Esto evita fallos por NaN/Inf y asegura pipeline robusto.

## 5) Algoritmos utilizados

El pipeline compara 3 modelos clasicos para datos tabulares:

### 5.1) Logistic Regression

Pipeline:

- SimpleImputer(median)
- StandardScaler
- LogisticRegression(max_iter=5000, class_weight='balanced', random_state=seed)

Ventajas:

- Baseline interpretable
- Rapido
- Bueno cuando la separacion es casi lineal

### 5.2) Random Forest

Pipeline:

- SimpleImputer(median)
- RandomForestClassifier(
  - n_estimators=600
  - class_weight='balanced_subsample'
  - random_state=seed
  - n_jobs=-1
)

Ventajas:

- Captura no linealidades
- Robusto a escala de variables

### 5.3) HistGradientBoosting

Pipeline:

- SimpleImputer(median)
- HistGradientBoostingClassifier(
  - max_depth=6
  - learning_rate=0.05
  - max_iter=400
  - random_state=seed
)

Ventajas:

- Muy competitivo en tabular
- Buen balance entre precision y costo

## 6) Metodo de entrenamiento y validacion

### 6.1) Split de validacion

Se usa StratifiedKFold:

- n_splits configurable (default 5)
- shuffle=True
- random_state fijo

Esto preserva proporcion de clases en cada fold.

### 6.2) Entrenamiento por modelo

Para cada algoritmo:

1. Entrena en train fold
2. Predice probabilidades en test fold
3. Guarda metrica por fold
4. Guarda predicciones OOF (out-of-fold) por sujeto

### 6.3) Metricas calculadas

En _compute_metrics:

- balanced_accuracy
- roc_auc
- f1
- sensitivity
- specificity

Definiciones practicas:

- sensitivity = TP / (TP + FN)
- specificity = TN / (TN + FP)
- umbral de decision = 0.5

### 6.4) Seleccion del mejor modelo

Criterio de seleccion:

- maximo balanced_accuracy_mean (promedio de folds)

Luego:

- se reentrena ese mejor modelo con TODO el dataset
- se guarda artifact final en model_best.joblib

## 7) Artefactos de salida y para que sirve cada uno

Archivos principales:

- model_best.joblib
  - pipeline entrenado
  - nombre de mejor modelo
  - feature_names
  - config usada
  - definicion de labels

- cv_fold_metrics.csv
  - metrica por fold por modelo

- cv_summary_metrics.csv
  - resumen agregado por modelo

- oof_predictions_<model>.csv
  - probabilidad out-of-fold por sujeto
  - prediccion con threshold 0.5

- X_features.npy
- y_binary.npy
- feature_names.json
- subjects_binary_labels.csv

Estos permiten auditoria, trazabilidad y reproducibilidad completa.

## 8) Analisis automatico posterior al entrenamiento

Si generate_analysis=True (default), se crean:

### 8.1) Graficos de evaluacion

- plot_roc_curve.png
- plot_pr_curve.png
- plot_confusion_matrix.png

Estos se construyen con OOF del mejor modelo.

### 8.2) Explainability por permutation importance

- explainability_permutation_importance.csv
- plot_permutation_top20.png

Implementacion optimizada:

- no permuta todas las features (seria muy costoso)
- primero rankea por importancia base del modelo (coef_ o feature_importances_)
- evalua top 120 candidatos
- 5 repeticiones por feature
- metrica objetivo: caida de balanced accuracy

Esto hace que explainability sea ejecutable en tiempos practicos.

### 8.3) SHAP opcional

- Si shap esta instalado:
  - explainability_shap_top.csv
  - plot_shap_top20.png
- Si no esta instalado:
  - explainability_shap_status.txt con diagnostico

## 9) Reporteria automatica

Se generan dos reportes:

### 9.1) report_model.md

Incluye:

- definicion del problema
- modelo ganador
- metricas OOF del mejor modelo
- tabla comparativa de modelos
- lista de artefactos generados

### 9.2) report_model.pdf

Incluye:

- portada tecnica con metricas
- tabla resumen
- paginas con imagenes ROC/PR/CM/permutation/SHAP(si existe)

## 10) Inferencia en produccion (prediccion por sujeto)

Ruta:

- scripts/predict_subject.py
- src/pd_binary_classifier/inference.py

Flujo exacto:

1. Carga model_best.joblib
2. Recupera config guardada dentro del artifact
3. Reconstruye feature vector del subject_id:
   - movimiento + cuestionario + demografia
4. model.predict_proba(x)[0,1]
5. Aplica threshold 0.5

Salida:

- subject_id
- predicted_label
- probability_parkinson

## 11) Reproducibilidad: como se garantiza

Puntos clave:

- seed fijo en split y modelos
- pipeline serializado
- feature_names guardados
- y y X finales guardados
- OOF guardado por modelo
- reportes auto-generados

Comando base reproducible:

- C:/Python313/python.exe scripts/train_model.py --folds 5 --output-dir outputs

Comando rapido de chequeo:

- C:/Python313/python.exe scripts/train_model.py --folds 2 --output-dir outputs_quick

Comando de inferencia:

- C:/Python313/python.exe scripts/predict_subject.py --subject-id 001 --model-path outputs/model_best.joblib

## 12) Decisiones metodologicas importantes

### 12.1) Por que balanced accuracy como criterio principal

Porque hay desbalance de clase (276 vs 193) y balanced accuracy da el mismo peso relativo a ambas clases.

### 12.2) Por que OOF para evaluar

Permite estimacion mas honesta de generalizacion en CV:

- cada sujeto es evaluado en fold no visto por el modelo
- se pueden generar curvas globales sobre predicciones OOF

### 12.3) Por que mezclar 0 y 2 como negativos

Es la definicion del problema binario solicitado:

- detectar Parkinson vs no Parkinson

Ventaja:

- escenario mas realista clinicamente (negativo heterogeneo)

Trade-off:

- la clase negativa es mas compleja y puede reducir especificidad

## 13) Limitaciones actuales del pipeline

1. No hay calibracion de probabilidad (Platt/Isotonic)
2. Threshold fijo 0.5 (no optimizado por objetivo clinico)
3. Permutation importance optimizada sobre top candidatos (no full scan)
4. SHAP depende de compatibilidad de libreria opcional
5. No hay split externo hold-out separado de CV

## 14) Mejoras recomendadas

1. Optimizar threshold por Youden o por recall objetivo
2. Agregar calibracion de probabilidad
3. Evaluar nested CV para tuning hiperparametrico
4. Comparar escenarios:
   - Parkinson vs Healthy
   - Parkinson vs Otros
   - Parkinson vs (Healthy+Otros)
5. Agregar fairness slices por edad/genero

## 15) Mapa de archivos del sistema (implementacion)

Nucleo del modelo:

- src/pd_binary_classifier/config.py
- src/pd_binary_classifier/data.py
- src/pd_binary_classifier/features.py
- src/pd_binary_classifier/training.py
- src/pd_binary_classifier/inference.py

Scripts de ejecucion:

- scripts/train_model.py
- scripts/predict_subject.py
- scripts/show_results.py

Documentacion:

- README.md
- docs/MODEL_GUIDE.md
- docs/EXPLICACION_END_TO_END.md

## 16) Resumen ejecutivo final

Este sistema implementa una solucion tabular multimodal robusta y reproducible para clasificar Parkinson vs No Parkinson usando:

- biomarcadores digitales de movimiento de smartwatch
- cuestionario NMS
- variables demograficas/clinicas

El entrenamiento compara tres familias de modelos, selecciona automaticamente el mejor por balanced accuracy, guarda artefactos de reproduccion, genera evaluacion visual, explainability y reportes listos para presentar.

Con esto tienes un pipeline completo de punta a punta, auditable y ejecutable por comando, sin pasos manuales ocultos.
