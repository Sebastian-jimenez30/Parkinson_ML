# Entrega: Problema, Datos, EDA, Modelo y Resultados

## 1. El problema

El objetivo de este proyecto es construir un sistema de clasificacion supervisada que permita distinguir entre tres estados clinicamente relevantes:

- Healthy: sujeto sano
- Parkinson: sujeto con diagnostico de enfermedad de Parkinson
- Other neurological diagnosis: sujeto con otro diagnostico neurologico relacionado con alteraciones del movimiento

Este planteamiento es mas robusto que una formulacion binaria porque obliga al modelo a resolver una tarea mas cercana a la practica clinica real. En lugar de separar solo Parkinson contra no Parkinson, el sistema debe reconocer si una persona es sana, si presenta Parkinson o si tiene otra condicion neurologica que puede generar patrones similares en las senales.

El problema se formula a partir de informacion multimodal recogida por smartwatch, cuestionarios de sintomas y variables demograficas y clinicas. La meta no es solo predecir una clase, sino hacerlo de forma reproducible, auditable y con un criterio de evaluacion que no favorezca artificialmente a la clase mayoritaria.

## 2. Datos

La base de datos seleccionada fue PADS, un conjunto de datos de smartwatch orientado al estudio de trastornos del movimiento. Para este proyecto se trabajo con la version ya estructurada y procesada del conjunto de datos, porque permite construir un pipeline uniforme para todos los sujetos y evita inconsistencias de formato.

Los insumos principales utilizados por el modelo fueron:

- La tabla de metadatos por sujeto, donde se encuentra la etiqueta de clase y variables demograficas o clinicas.
- Las senales de movimiento, almacenadas como series temporales multicanal por sujeto.
- El cuestionario de sintomas no motores, con 30 respuestas binarias por sujeto.

La estructura de trabajo mantiene una representacion por sujeto compuesta por tres bloques:

1. Movimiento: senales temporales multicanal.
2. Cuestionario: variables binarias del estado de sintomas.
3. Metadata: variables numericas, booleanas y categoricas.

En el conjunto utilizado se observan tres clases originales:

- 0: Healthy
- 1: Parkinson
- 2: Other neurological diagnosis

La distribucion no es perfectamente equilibrada, por lo que el analisis y la validacion debian centrarse en metricas robustas al desbalance.

## 3. EDA

El analisis exploratorio de datos se concentro en responder cuatro preguntas:

1. Los archivos tienen la forma esperada?
2. Las clases estan balanceadas?
3. La metadata es consistente y utilizable?
4. La representacion por sujeto es comparable entre casos?

### 3.1 Verificacion de integridad

Antes de entrenar, se reviso que cada sujeto tuviera correspondencia entre:

- su registro en la tabla principal,
- su archivo de movimiento,
- su archivo de cuestionario.

Tambien se valido que la carga de las senales produjera matrices con la forma esperada por sujeto. Esto es importante porque una clasificacion basada en series temporales falla rapidamente si hay archivos corrompidos, tamaños inesperados o desalineacion entre sujetos.

### 3.2 Distribucion de clases

Uno de los hallazgos mas importantes fue que la distribucion de clases no es uniforme. Esto significa que una medida como accuracy simple puede dar una lectura demasiado optimista. Por eso se decidio usar balanced accuracy como criterio principal de comparacion entre modelos.

La decision es metodologicamente correcta porque evita que el modelo parezca bueno solo por acertar mucho en la clase mayoritaria.

### 3.3 Calidad de la metadata

La metadata contiene variables numericas, booleanas y categoricas. Durante el EDA se verifico que estas variables pudieran limpiarse de manera consistente y convertirse en una matriz utilizable por modelos tabulares.

Se uso una estrategia simple y estable:

- variables numericas: conversion segura y tratamiento de valores faltantes con imputacion
- variables booleanas: mapeo a 0 y 1
- variables categoricas: codificacion one-hot

Esto evita que la informacion clinica se pierda y permite combinarla con las senales de movimiento.

### 3.4 Lectura exploratoria de la senal

La senal de smartwatch no se usa cruda. Primero se resume con estadisticas temporales y frecuenciales por canal. Esto reduce dimensionalidad y hace que el aprendizaje sea mas estable con un numero de sujetos relativamente pequeno.

En terminos practicos, el EDA permitio confirmar que el problema tenia sentido como clasificacion tabular multimodal, no como un simple clasificador sobre senales crudas.

## 4. Modelo

El sistema implementa un pipeline de clasificacion multiclase con tres modelos candidatos:

- Logistic Regression
- Random Forest
- HistGradientBoosting

La idea no es imponer un unico algoritmo desde el inicio, sino comparar varias familias de modelos que se comportan distinto frente a variables de alta dimensionalidad, relaciones no lineales y desbalance de clases.

### 4.1 Ingenieria de caracteristicas

Cada sujeto se transforma en un vector numerico con tres bloques:

1. Caracteristicas de movimiento
2. Caracteristicas del cuestionario
3. Caracteristicas demograficas y clinicas

#### 4.1.1 Movimiento

La senal se organiza en 132 canales y para cada canal se extraen 12 descriptores:

- media
- desviacion estandar
- mediana
- rango intercuartilico
- RMS
- energia
- media absoluta
- asimetria
- kurtosis excesiva
- frecuencia dominante
- potencia de banda 3 a 7 Hz
- entropia espectral

Esta estrategia convierte una senal larga en un conjunto de rasgos resumidos que capturan amplitud, variabilidad y contenido frecuencial. Es una forma practicamente util de resumir el movimiento sin depender de redes neuronales o de un entrenamiento demasiado pesado.

#### 4.1.2 Cuestionario

El cuestionario aporta 30 variables binarias. Su valor principal es que complementa la senal con informacion sintomatologica relevante. A nivel de interpretacion, ayuda a capturar manifestaciones que no siempre aparecen con claridad en el movimiento.

#### 4.1.3 Metadata

Se incorporan variables demograficas y clinicas como edad, estatura, peso, edad de diagnostico, genero, lateralidad y antecedentes. Esto permite que el clasificador vea el caso de forma mas completa.

### 4.2 Modelos comparados

Se comparan tres modelos porque cubren distintos compromisos entre interpretabilidad, capacidad de generalizacion y manejo de relaciones no lineales.

#### 4.2.1 Logistic Regression

Este modelo funciona como linea base fuerte. Es util cuando se quiere una referencia estable y relativamente interpretable.

Criterio de configuracion:

- imputacion por mediana para manejar faltantes
- escalado para estabilizar coeficientes
- max_iter alto para asegurar convergencia
- class_weight balanceado para compensar el desbalance

#### 4.2.2 Random Forest

Este modelo captura relaciones no lineales y combinaciones entre variables sin requerir escalado.

Criterio de configuracion:

- numero alto de arboles para reducir varianza
- class_weight balanceado por submuestra para mitigar sesgo
- configuracion conservadora de la profundidad efectiva mediante el comportamiento del ensamble

#### 4.2.3 HistGradientBoosting

Este modelo fue incluido porque suele rendir muy bien en problemas tabulares complejos. Permite modelar interacciones no lineales con buena eficiencia.

Criterio de configuracion:

- profundidad moderada para evitar sobreajuste excesivo
- tasa de aprendizaje conservadora
- numero suficiente de iteraciones para capturar patrones complejos

### 4.3 Criterio para seleccionar hiperparametros

Los hiperparametros no se eligieron por ensayo aleatorio, sino por criterio tecnico:

1. Estabilidad: evitar configuraciones que no converjan o que varien demasiado entre folds.
2. Generalizacion: preferir modelos que rindan bien fuera de la muestra de entrenamiento.
3. Robustez al desbalance: usar estrategias de ponderacion de clases y metricas apropiadas.
4. Costo computacional razonable: mantener un pipeline reproducible que pueda ejecutarse sin depender de infraestructura costosa.

Por eso el modelo usa validacion cruzada estratificada y no una sola particion aleatoria.

### 4.4 Estrategia de entrenamiento

El entrenamiento se realiza con Stratified K-Fold. Esto significa que cada fold conserva aproximadamente la misma proporcion de clases que el conjunto completo, lo que es importante para un problema multiclase con desbalance.

En cada fold se hace lo siguiente:

1. Se entrena el modelo en los datos de entrenamiento.
2. Se predice en el fold de validacion.
3. Se guardan las probabilidades de cada clase.
4. Se calcula el desempeno por fold.
5. Se agregan las metricas para comparar modelos.

Al final se elige el modelo con mejor balanced accuracy promedio.

### 4.5 Metricas utilizadas

Para un problema multiclase, las metricas elegidas deben reflejar desempeño global y no solo exito en una clase puntual. Por eso se usan:

- Balanced Accuracy
- F1 macro
- Precision macro
- Recall macro
- ROC-AUC OVR macro
- PR-AUC OVR macro

La metrica principal sigue siendo balanced accuracy porque es una de las mas estables frente a desbalance de clases. Las metricas macro complementan la lectura, ya que tratan cada clase con el mismo peso.

### 4.6 Explainability y reporteria

El pipeline no solo entrena modelos. Tambien genera artefactos de interpretacion y reporte:

- matriz de confusion multiclase
- curvas ROC OVR
- curvas Precision-Recall OVR
- permutation importance
- SHAP opcional
- reporte final en Markdown y PDF

La permutation importance se restringe a un conjunto de features candidatas para evitar que el costo computacional crezca demasiado. Esto hace que la interpretacion sea viable sin frenar el flujo de entrenamiento.

## 5. Resultados

Los resultados muestran que el enfoque multiclase es mas informativo que la version binaria previa. En particular, el sistema ya no resume todo en un solo contraste Parkinson versus no Parkinson, sino que separa claramente tres escenarios distintos.

### 5.1 Hallazgos principales

1. HistGradientBoosting tiende a ser el modelo mas fuerte en desempeño global.
2. Random Forest puede captar bastante bien la clase de Parkinson, pero su comportamiento puede ser mas agresivo en la prediccion.
3. Logistic Regression funciona como una referencia estable y util para comparar.
4. El uso de balanced accuracy evita conclusiones engañosas por desbalance.

### 5.2 Interpretacion

El resultado mas importante es que el problema no queda reducido a una pregunta demasiado simple. La nueva formulacion obliga al sistema a diferenciar entre un sujeto sano, un sujeto con Parkinson y un sujeto con otro trastorno neurologico. Esto es metodologicamente mas robusto y clinicamente mas util.

Ademas, la fusion de senales de movimiento, cuestionario y metadata mejora la capacidad del modelo para representar el caso completo y no solo una parte aislada de la informacion.

### 5.3 Conclusiones finales

1. El pipeline resuelve un problema multiclase con una estructura clara y reproducible.
2. La representacion multimodal es adecuada para este tipo de diagnostico.
3. La validacion cruzada estratificada y el uso de metricas macro hacen que la comparacion entre modelos sea mas justa.
4. El sistema queda listo para iteraciones futuras, como ajuste de umbrales, comparacion con otros algoritmos y validacion externa.

## 6. Referencias internas

- [src/pd_binary_classifier/data.py](src/pd_binary_classifier/data.py)
- [src/pd_binary_classifier/features.py](src/pd_binary_classifier/features.py)
- [src/pd_binary_classifier/training.py](src/pd_binary_classifier/training.py)
- [src/pd_binary_classifier/inference.py](src/pd_binary_classifier/inference.py)
- [scripts/train_model.py](scripts/train_model.py)
- [scripts/predict_subject.py](scripts/predict_subject.py)
- [scripts/show_results.py](scripts/show_results.py)
