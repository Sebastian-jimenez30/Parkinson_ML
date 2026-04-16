# Entrega: Problema, Datos, EDA, Modelo y Resultados

## 1. El problema: definicion clara del reto a resolver

El reto de este proyecto es construir un clasificador binario que, a partir de senales de smartwatch, cuestionarios de sintomas no motores y variables clinicas/demograficas, determine si un sujeto pertenece a la clase:

- Parkinson (clase positiva)
- No Parkinson (clase negativa)

La formulacion binaria adoptada fue:

- Positivo: sujetos con diagnostico Parkinson (label original = 1)
- Negativo: sujetos Healthy + otros diagnosticos de movimiento (labels originales = 0 y 2)

Esta definicion responde a un escenario clinico realista de tamizaje inicial, donde el sistema debe separar Parkinson de un conjunto heterogeneo de no-Parkinson, en lugar de compararlo unicamente contra controles sanos.

Objetivo tecnico del modelo:

1. Maximizar la capacidad discriminativa entre ambas clases.
2. Mantener equilibrio entre sensibilidad y especificidad.
3. Ser reproducible y auditable (pipeline completo, artefactos y reportes automaticos).

## 2. Datos: la base de datos seleccionada

La base utilizada fue PADS (Parkinsons Disease Smartwatch dataset), disponible localmente en:

- [Data/pads-parkinsons-disease-smartwatch-dataset-1.0.0](Data/pads-parkinsons-disease-smartwatch-dataset-1.0.0)

Para entrenamiento del modelo se uso la version preprocesada oficial del dataset en:

- [Data/pads-parkinsons-disease-smartwatch-dataset-1.0.0/preprocessed](Data/pads-parkinsons-disease-smartwatch-dataset-1.0.0/preprocessed)

Archivos principales consumidos por el pipeline:

1. [Data/pads-parkinsons-disease-smartwatch-dataset-1.0.0/preprocessed/file_list.csv](Data/pads-parkinsons-disease-smartwatch-dataset-1.0.0/preprocessed/file_list.csv)
2. [Data/pads-parkinsons-disease-smartwatch-dataset-1.0.0/preprocessed/movement](Data/pads-parkinsons-disease-smartwatch-dataset-1.0.0/preprocessed/movement)
3. [Data/pads-parkinsons-disease-smartwatch-dataset-1.0.0/preprocessed/questionnaire](Data/pads-parkinsons-disease-smartwatch-dataset-1.0.0/preprocessed/questionnaire)

Dimension del conjunto:

- Sujetos totales: 469
- Distribucion label original:
  - 0 (Healthy): 79
  - 1 (Parkinson): 276
  - 2 (Otros diagnosticos): 114

Distribucion binaria final:

- Parkinson (1): 276
- No Parkinson (0): 193

Modalidades usadas por sujeto:

1. Movimiento (binario): contiene series temporales ya estandarizadas por canal.
2. Cuestionario NMS: 30 respuestas binarias.
3. Demografia/clinica: edad, estatura, peso, edad de diagnostico, genero, lateralidad, antecedentes, etc.

## 3. EDA: resumen del analisis exploratorio realizado

El EDA (Analisis Exploratorio de Datos) se enfoco en verificar calidad estructural, consistencia y riesgo metodologico antes de entrenar.

### 3.1 Verificacion de integridad y consistencia

Se verifico que:

1. Existe correspondencia 1:1 entre sujetos del CSV y archivos binarios de movimiento/cuestionario.
2. El tamano de cada binario coincide con la forma esperada del pipeline.
3. Los IDs de sujeto pueden normalizarse en formato de tres digitos (001, 002, ...).

En particular, para movimiento:

- Forma esperada por sujeto: 132 canales x 976 pasos temporales.

Para cuestionario:

- Forma esperada por sujeto: 30 items.

La validacion de formas se implementa en:

- [src/pd_binary_classifier/data.py](src/pd_binary_classifier/data.py)

mediante checks explicitos de tamano y excepciones de error en caso de inconsistencia.

### 3.2 Distribucion de clases y sesgo de muestreo

El hallazgo principal del EDA fue el desbalance de clases:

- Clase positiva (Parkinson) mayoritaria (276)
- Clase negativa combinada menor (193)

Implicacion:

- Accuracy simple puede ser enganosa.
- Se priorizo Balanced Accuracy como metrica de seleccion de modelo.

### 3.3 Composicion clinica de la clase negativa

La clase negativa no es homogenea (controles sanos + otros diagnosticos neurologicos). Esto aumenta la dificultad del problema, pero representa mejor la practica clinica de descarte diferencial.

### 3.4 Revision de metadata y codificacion

Se reviso el manejo de variables tabulares para evitar ruido:

1. Numericas: conversion robusta con coercion.
2. Booleanas: mapeo explicito True/False a 1.0/0.0.
3. Categoricas: one-hot encoding con categoria Unknown para vacios.

Esto reduce riesgo de errores por formato mixto en columnas de metadata.

## 4. Modelo: explicacion del modelo implementado y criterio de seleccion de parametros e hiperparametros

El sistema no usa un unico modelo fijo de entrada; implementa comparacion controlada de tres algoritmos, y seleccion automatica del mejor por desempeno en validacion cruzada.

### 4.1 Pipeline de features (ingenieria de caracteristicas)

El vector final por sujeto combina tres bloques:

1. Features de movimiento (dominante)
2. Features de cuestionario
3. Features demograficas/clinicas

#### 4.1.1 Features de movimiento

Por cada uno de 132 canales, se calculan 12 descriptores:

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

Total movimiento: 132 x 12 = 1584 features.

Implementado en:

- [src/pd_binary_classifier/features.py](src/pd_binary_classifier/features.py)

#### 4.1.2 Features de cuestionario

- 30 variables binarias (NMS) incluidas directamente.

#### 4.1.3 Features tabulares

- Variables numericas, booleanas y categoricas codificadas con one-hot.

### 4.2 Algoritmos comparados

En [src/pd_binary_classifier/training.py](src/pd_binary_classifier/training.py) se comparan:

1. Logistic Regression
2. Random Forest
3. HistGradientBoosting

Con pipelines de preprocesamiento consistentes.

#### 4.2.1 Logistic Regression

- Imputer: mediana
- Escalado: StandardScaler
- Hiperparametros:
  - max_iter = 5000
  - class_weight = balanced
  - random_state = seed

Criterio de estos hiperparametros:

- max_iter alto para asegurar convergencia en alta dimensionalidad.
- class_weight balanced para compensar desbalance.

#### 4.2.2 Random Forest

- Imputer: mediana
- Hiperparametros:
  - n_estimators = 600
  - class_weight = balanced_subsample
  - n_jobs = -1
  - random_state = seed

Criterio:

- n_estimators suficientemente alto para estabilidad del ensamble.
- balanced_subsample para mitigar sesgo por clases.

#### 4.2.3 HistGradientBoosting

- Imputer: mediana
- Hiperparametros:
  - max_depth = 6
  - learning_rate = 0.05
  - max_iter = 400
  - random_state = seed

Criterio:

- profundidad moderada para captar no linealidad sin sobreajuste extremo.
- learning rate conservador con numero razonable de iteraciones.

### 4.3 Metodo de entrenamiento

Se uso validacion cruzada estratificada:

- StratifiedKFold
- n_splits configurable (default 5)
- shuffle = True
- random_state fijo

Proceso por modelo:

1. Entrenar en folds de entrenamiento.
2. Predecir probabilidades en fold de validacion.
3. Guardar metricas por fold.
4. Construir predicciones OOF (out-of-fold).

Criterio de seleccion del mejor modelo:

- Mayor balanced_accuracy_mean en CV.

Luego se reentrena el mejor modelo sobre todo el conjunto y se serializa.

### 4.4 Metricas utilizadas

1. Balanced Accuracy (principal)
2. ROC AUC
3. F1
4. Sensitivity
5. Specificity

Razon del criterio principal:

- Balanced Accuracy es mas robusta al desbalance de clases que accuracy simple.

### 4.5 Explainability y reporteria

Tras seleccionar modelo ganador, el sistema genera:

1. Curvas ROC y PR
2. Matriz de confusion
3. Permutation importance (top candidatos para costo computacional razonable)
4. SHAP opcional (si dependencia instalada)
5. Reporte automatico en Markdown y PDF

Salidas de referencia (corrida completa):

- [outputs_fullstep/report_model.md](outputs_fullstep/report_model.md)
- [outputs_fullstep/report_model.pdf](outputs_fullstep/report_model.pdf)
- [outputs_fullstep/cv_summary_metrics.csv](outputs_fullstep/cv_summary_metrics.csv)

## 5. Resultados: conclusiones y hallazgos obtenidos

### 5.1 Desempeno comparativo de modelos

Resumen observado en la corrida validada:

- HistGradientBoosting: mejor Balanced Accuracy promedio.
- Logistic Regression: segundo mejor, menor capacidad no lineal.
- Random Forest: alta sensibilidad pero especificidad baja, indicando tendencia a sobre-predecir Parkinson en este escenario binario.

Valores registrados en:

- [outputs_fullstep/cv_summary_metrics.csv](outputs_fullstep/cv_summary_metrics.csv)

Hallazgo clave:

- El modelo de boosting ofrece el mejor equilibrio entre detectar Parkinson (sensibilidad) y no sobrediagnosticar (especificidad), bajo la definicion binaria adoptada.

### 5.2 Interpretacion clinico-tecnica

1. El problema no es trivial porque la clase negativa incluye trastornos de movimiento parecidos a Parkinson.
2. La fusion multimodal (movimiento + cuestionario + demografia) mejora riqueza de senal para clasificacion.
3. El criterio de Balanced Accuracy es apropiado y alineado con el riesgo de desbalance.

### 5.3 Limitaciones detectadas

1. SHAP depende de libreria opcional y compatibilidad del entorno.
2. Umbral fijo 0.5; no se optimizo por objetivo clinico.
3. No se uso holdout externo aparte de CV.

### 5.4 Conclusiones finales

1. El pipeline implementado resuelve de forma reproducible el reto binario planteado.
2. Existe evidencia de capacidad predictiva util, con mejor desempeno en HistGradientBoosting.
3. La arquitectura es auditable y extensible: permite reentrenar, comparar, explicar y reportar automaticamente.
4. El sistema esta listo como base de trabajo para iteraciones de investigacion (tuning, calibracion de umbral, validacion externa y analisis por subgrupos).

## 6. Referencias internas del proyecto

Codigo principal:

- [src/pd_binary_classifier/training.py](src/pd_binary_classifier/training.py)
- [src/pd_binary_classifier/features.py](src/pd_binary_classifier/features.py)
- [src/pd_binary_classifier/data.py](src/pd_binary_classifier/data.py)
- [src/pd_binary_classifier/inference.py](src/pd_binary_classifier/inference.py)

Scripts de ejecucion:

- [scripts/train_model.py](scripts/train_model.py)
- [scripts/show_results.py](scripts/show_results.py)
- [scripts/predict_subject.py](scripts/predict_subject.py)

Reportes y salidas:

- [outputs_fullstep/report_model.md](outputs_fullstep/report_model.md)
- [outputs_fullstep/report_model.pdf](outputs_fullstep/report_model.pdf)
- [outputs_fullstep/cv_summary_metrics.csv](outputs_fullstep/cv_summary_metrics.csv)
