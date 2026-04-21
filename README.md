# Clasificador multiclase neurologico

Este proyecto implementa un flujo reproducible de aprendizaje automatico para clasificar sujetos en tres grupos:

- Healthy (sano)
- Parkinson
- Other neurological diagnosis (otro diagnostico neurologico)

La idea es responder una pregunta clinicamente util: a partir de las senales del reloj inteligente, el cuestionario de sintomas y la metadata del sujeto, el sistema estima la probabilidad de pertenencia a cada una de las tres clases.

La salida principal del entrenamiento se guarda en una sola carpeta:

- [outputs](outputs)

## 1. Como se procesaron los datos

El modelo no trabaja directamente con archivos crudos. Primero se construyo una version procesada y estandarizada de los datos para que cada sujeto tenga una representacion homogenea y comparable. El objetivo de este paso fue eliminar ruido innecesario, alinear las senales y dejar una estructura apta para extraer caracteristicas.

### 1.1 Senales de movimiento

Las senales de movimiento se trataron como series temporales multicanal. Para cada sujeto se trabajo con una matriz de 132 canales y 976 muestras por canal, a 100 Hz.

Las estrategias usadas fueron:

1. Estandarizacion de la estructura temporal para que todos los sujetos tuvieran el mismo tamano.
2. Reordenamiento consistente de canales para mantener siempre la misma interpretacion por eje y sensor.
3. Eliminacion de canales o componentes poco utiles para la clasificacion, especialmente los asociados a tareas de bajo valor discriminativo.
4. Detrending de las senales de acelerometro para reducir sesgos de baja frecuencia y corregir deriva lenta del sensor.
5. Recorte inicial de una pequena fraccion temporal para disminuir el efecto de transitorios iniciales y dejar la parte mas estable de la actividad.

La idea de fondo fue que el modelo vea patrones de movimiento comparables entre sujetos, no diferencias accidentales de formato.

### 1.2 Cuestionario de sintomas

El cuestionario se incorporo como un vector de 30 variables binarias. Cada respuesta se uso como una senal clinica directa, sin transformaciones complejas, porque ya representa un atributo semiestructurado y facil de interpretar.

### 1.3 Metadata clinica y demografica

Tambien se incluyeron variables como edad, peso, estatura, genero, lateralidad y antecedentes relevantes. Estas variables se limpiaron y codificaron de forma consistente:

- variables numericas: conversion robusta y remplazo de valores invalidos por imputacion
- variables booleanas: transformacion a 0/1
- variables categoricas: codificacion one-hot

Esto permite que el modelo combine informacion de movimiento con contexto clinico del sujeto.

## 2. Estructura general del problema

La definicion multiclase usada en el proyecto es:

- clase 0: Healthy
- clase 1: Parkinson
- clase 2: Other neurological diagnosis

Este planteamiento es mas robusto porque no reduce el problema a Parkinson vs no Parkinson, sino que separa explicitamente sujetos sanos de otros casos neurologicos.

## 3. Modelos y entrenamiento

El pipeline compara tres modelos clasicos de aprendizaje supervisado sobre la misma matriz de caracteristicas:

1. Regresion logistica
2. Random Forest
3. HistGradientBoosting

### 3.1 Como se construye el vector de entrada

Para cada sujeto, el sistema junta tres bloques:

- caracteristicas de movimiento extraidas por canal
- variables del cuestionario
- variables clinicas y demograficas

En el bloque de movimiento se calculan descriptores temporales y espectrales. Esto convierte una senal larga en un conjunto de medidas resumidas que capturan amplitud, variabilidad, energia y contenido frecuencial.

### 3.2 Porque se eligieron estos modelos

- Regresion logistica: sirve como linea base fuerte, es interpretable y funciona bien cuando hay muchas variables y una frontera mas o menos lineal.
- Random Forest: captura interacciones no lineales entre variables y es robusto al ruido.
- HistGradientBoosting: suele rendir muy bien en datos tabulares complejos, porque modela relaciones no lineales con mayor flexibilidad.

### 3.3 Criterio para elegir hiperparametros

Los hiperparametros no se escogieron al azar. Se fijaron pensando en estabilidad, capacidad de generalizacion y costo computacional:

- Regresion logistica: `max_iter` alto para asegurar convergencia y `class_weight='balanced'` para compensar el desbalance de clases.
- Random Forest: numero alto de arboles para reducir varianza y `class_weight='balanced_subsample'` para evitar sesgo por la clase mayoritaria.
- HistGradientBoosting: profundidad moderada, tasa de aprendizaje conservadora y numero suficiente de iteraciones para aprender relaciones complejas sin sobreajustar demasiado.

### 3.4 Estrategia de entrenamiento

El entrenamiento se hace con validacion cruzada estratificada:

- se separan los datos en varios folds
- cada modelo se entrena en una parte y se valida en la otra
- se repite hasta cubrir todos los folds

Luego se promedian las metricas (balanced accuracy, F1 macro, precision/recall macro, ROC-AUC OVR y PR-AUC OVR) y se escoge el modelo con mejor balanced accuracy media.

Se usa balanced accuracy como criterio principal porque las clases no estan perfectamente balanceadas y una accuracy simple podria dar una impresion demasiado optimista en escenarios multiclase.

### 3.5 Que produce el entrenamiento

Todo queda guardado en [outputs](outputs):

- `model_best.joblib`: modelo final entrenado
- `cv_summary_metrics.csv`: resumen comparativo de modelos
- `cv_fold_metrics.csv`: metricas por fold
- `oof_predictions_<modelo>.csv`: probabilidades out-of-fold por clase
- `X_features.npy` y `y_multiclass.npy`: matriz final de entrenamiento
- `feature_names.json`: nombres de todas las caracteristicas
- `subjects_multiclass_labels.csv`: mapeo de sujeto y etiqueta multiclase
- graficas ROC OVR, PR OVR y matriz de confusion multiclase
- explainability por permutation importance
- reporte final en Markdown y PDF

## 4. Como ejecutar el proyecto

### 4.1 Instalar dependencias

```powershell
C:/Python313/python.exe -m pip install -r requirements.txt
```

### 4.2 Entrenar el modelo

```powershell
C:/Python313/python.exe scripts/train_model.py
```

Si quieres cambiar la cantidad de folds o la semilla:

```powershell
C:/Python313/python.exe scripts/train_model.py --seed 42 --folds 5 --output-dir outputs
```

### 4.3 Ver comparacion de modelos

```powershell
C:/Python313/python.exe scripts/show_results.py
```

### 4.4 Predecir un sujeto

```powershell
C:/Python313/python.exe scripts/predict_subject.py --subject-id 001
```

### 4.5 SHAP opcional

Si tu entorno soporta SHAP, instala el extra opcional:

```powershell
C:/Python313/python.exe -m pip install -r requirements-optional.txt
```

## 5. Documentacion adicional

Si quieres una explicacion mas tecnica, revisa:

- [docs/MODEL_GUIDE.md](docs/MODEL_GUIDE.md)
- [docs/ENTREGA_PROBLEMA_DATOS_EDA_MODELO_RESULTADOS.md](docs/ENTREGA_PROBLEMA_DATOS_EDA_MODELO_RESULTADOS.md)

## 6. Nota importante sobre la carpeta de salida

Este repositorio se organizo para trabajar con una sola carpeta de resultados:

- [outputs](outputs)

Las carpetas alternativas de ejecuciones previas fueron eliminadas para evitar confusion y mantener un unico lugar donde revisar artefactos, metricas y reportes.
