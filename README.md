# Evaluación de Modelos de Lenguaje en Español para Tareas de Clasificación Multiclase

## Objetivo

Realizamos una comparación entre varios modelos de lenguaje fácilmente utilizables para tareas de clasificación de texto por medio de redes neuronales profundas. Estos modelos están disponibles en [Tensorflow Hub](img/https://tfhub.dev/s?module-type=text-embedding) y se pueden usar de forma muy simple como una capa de embeddings de texto. El objetivo de esta comparación es obtener un resumen de las ventajas y características particulares de cada modelo para tareas en español. Usamos como proxy una tarea simple de clasificación multiclase de textos cortos esperando que algunas de las generalidades encontradas se puedan aplicar a otro tipo de tareas (NER, respuesta de tareas).

## Modelos y datos

### Modelos de lenguaje

Al momento evaluamos modelos pertenecientes a la familia [Neural Nets Language Models](img/https://tfhub.dev/google/collections/nnlm/1) entrenados en el corpus Spanish Google News 50B, con embeddings de 50 y 128 dimensiones, con y sin normalización de textos. Evaluamos también los modelos equivalentes en inglés con el fin de comparar el beneficio de usar la versión de su lenguaje específico. 

Modelos evaluados. 

- nnlm-en-dim128
- nnlm-en-dim128-with-normalization
- nnlm-en-dim50
- nnlm-en-dim50-with-normalization
- nnlm-es-dim128
- nnlm-es-dim128-with-normalization
- nnlm-es-dim50
- nnlm-es-dim50-with-normalization

### Dataset y tarea de clasificación

Como tarea de clasificación se utilizó la propuesta en [MercadoLibre Data Challenge 2019](img/https://ml-challenge.mercadolibre.com/) la cual consiste en un dataset de 10 millones de títulos de anuncios los cuales se pueden clasificar en aproximadamente 1,500 categorías. Como métrica de evaluación se utiliza [Balanced Accuracy Score](https://ml-challenge.mercadolibre.com/rules) tal como se implementó en el reto original. El [dataset](https://ml-challenge.mercadolibre.com/downloads) contiene mitad de datos en español y mitad en portugués por lo que se filtró únicamente la porción en español. En la tarea de evaluación se utilizó un subconjunto aleatorio de 1,000,000 de registros, validacición de 5% del conjunto de entrenamiento y 25,000 registros de prueba (test).

### Arquitectura de la red neuronal

La arquitectura de la red neuronal consiste en conectar la salida de embeddings del modelo de lenguaje con 3 capas densas de 512 unidades las dos primeras y el número de clases en la capa final, con Dropout entre cada una de ellas.

TODO: diagramas y explicación

### Ejecución de los experimentos

Cada modelo se corrió 4 veces y en el análsis se promediaron algunos resultados.

TODO: Explicación del método, metadatos del experimento, ambiente de ejecución, etc


## Resultados

Resultados completos en este [notebook](img/analysis/lmevME-LM-Analysis_v1.ipynb) o [HTML](doc/lmevME-LM-Analysis_v1.html)

- **Los modelos de lenguaje en español tienen un mejor desempeño que los modelos en inglés,** donde los primerios tuvieron en promedio BAC=0.7434 vs 0.7190 de los segundos. Es interesante ver también el BAC de cada experimento.

![bac_by_lang.png](img/bac_by_lang.png)

- **Los modelos normalizados (en español) tuvieron los mejores resultados.** En la gráfica abajo podemos observar los resultados promedio de cada modelo.

![bac_by_lm.png](img/bac_by_lm.png)

- Es interesante ver que **en los modelos normalizados, el modelo de 50 dimensiones tuvo mejor desempeño que el modelo de 128 dimensiones.** En la siguiente gráfica de *loss vs bac* vemos que el modelo de 50 dimensiones tuvo una menor variación que el modelo de 128 dimensiones. Sin embargo en los demás casos los modelos de 128 dimensiones tuvieron mejor desempeño que los de 50 dimensiones. Sería útil evaluar mas a profundidad y determinar si las optimizaciones a hiperparámetros pudieran explicar esa diferencia.

![loss_vs_bac.png](img/loss_vs_bac.png)

- Este último punto se puede reforzar observando las curvas de aprendizaje donde podemos ver que el modelo 128dim normalizado tuvo una menor pérdida (loss) pero que comenzó a sobreajustar antes, por lo que una cuidadosa optimización de hiperparámetros podría ayudarle a tener un menor desempeño

![loss_training_curve.png](img/loss_training_curve.png)

- Los experimentos se corrieron en el ambiente de [Google Colaboratory](img/https://colab.research.google.com/) el cual nos dió dos distintos tipos de GPU, en algunas ocasiones una Tesla T4 y en otra Tesla P100. Con ello podemos observar las siguientes conclusiones:

	- Los modelos de 50 dimensiones entrenan aproximadamente en la mitad del tiempo de los de 128 dimensiones
	- Los modelos entrenados en GPU Tesla P100 entrenan aproximadamente en la mitad del tiempo de las GPU Tesla T4
	- No hay una gran diferecia entre los modelos normalizados y no normalizados

![bac_vs_time.png](img/bac_vs_time.png)


# Spanish Language Model Evaluation for Multi-class Classification Tasks

## Objective

# Code and Data

## Código 

## Config

- Crear los siguientes directorios: 

```
cache/tfhub
cache/cuda
logs/
saved_models/
```

# Referencias



