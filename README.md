[Español](#evaluación-de-modelos-neuronales-de-lenguaje-en-español-en-una-tarea-de-clasificación-multiclase) | [English](#spanish-language-model-evaluation-for-multi-class-classification-tasks)

# Evaluación de Modelos Neuronales de Lenguaje en Español en una Tarea de Clasificación Multiclase

## Objetivo

Realizamos una comparación entre varios modelos de lenguaje que son fácilmente reutilizables para tareas de clasificación de texto por medio de redes neuronales profundas (DNN). Estos modelos están disponibles en [Tensorflow Hub](https://tfhub.dev/s?module-type=text-embedding) y se pueden usar de forma muy simple como una capa de embeddings de texto que precede a un perceptrón multicapa (MLP). El objetivo de esta comparación es obtener un resumen de las características y ventajas de cada modelo para tareas en español. Para ello usamos como proxy una tarea simple de clasificación multiclase de textos cortos esperando que algunas de las generalidades encontradas se puedan aplicar después a otro tipo de tareas de procesamiento de lenguaje natural tales como análsis de sentimiento, similitud textual o reconocimiento de entidades nombradas. También esperamos que el modelo de clasificador pueda ser reutilizado, adaptado o ampliado a otras tareas.

## Modelos y datos

### Modelos de lenguaje

Hasta este momento hemos evaluado modelos pertenecientes a la familia [Neural-Net Language Models](https://tfhub.dev/google/collections/nnlm/1) entrenados en el corpus Spanish Google News 50B, con embeddings de 50 y 128 dimensiones, con y sin normalización de textos. Evaluamos también los modelos equivalentes en inglés con el fin de comparar el beneficio de usar la versión de su lenguaje específico. 

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

Como tarea de clasificación se utilizó la propuesta en [MercadoLibre Data Challenge 2019](https://ml-challenge.mercadolibre.com/) la cual consiste en un dataset de 10 millones de títulos de anuncios los cuales se pueden clasificar en algo más de 1,500 categorías. Como métrica de evaluación se utiliza [Balanced Accuracy Score](https://ml-challenge.mercadolibre.com/rules) tal como se requirió en el reto original. El [dataset](https://ml-challenge.mercadolibre.com/downloads) contiene mitad de datos en español y mitad en portugués por lo que se filtró únicamente la porción en español. En la tarea de evaluación se utilizó como conjunto de entrenamiento (training set) un subconjunto aleatorio de 1,000,000 de registros con un 5% de ellos usados para validación, y 25,000 registros adicionales para el conjunto de prueba (test set).

### Arquitectura de la red neuronal

La arquitectura de la red neuronal consiste en conectar la salida de embeddings del modelo de lenguaje con 3 capas densas de 512 unidades las dos primeras y el número de clases en la capa final, con Dropout entre cada una de ellas. Se entrenó con optimizador Adam con valores default y como pérdida (loss) se utilizó Sparse Categorical Crossentropy. El batch size en el entrenamiento se mantuvo constante en 4096. Se entrenó por un máximo de 10 épocas pero con EarlyStopping con paciencia==3 por lo que el entrenamiento terminó antes en varios casos. No se hizo otro tipo de optimización de hiperparámetros ya que el objetivo es obtener una idea general del comportamiento de los modelos de lenguaje y no un óptimo de cada uno.

TODO: diagramas y explicación

### Ejecución de los experimentos

En este [notebook](analysis/NNLM_50_es-v1_0.ipynb) se puede ver un ejemplo del proceso del experimento aunque hay que señalar que los experimentos finales se corrieron mediante procesos batch y no con este notebook específico. Los resultados de cada experimento se puede ver [aquí.](experiments/)

Cada modelo se corrió 4 veces y en el análsis se promediaron algunos resultados. Para cada experimento se generó un archivo de metadatos con la configuración del experimento y los resultados obtenidos. Se pueden observar en [la misma liga](experiments/)

Los experimentos se corrieron mediante un framework desarrollado a la medida, en el ambiente de [Google Colaboratory](https://colab.research.google.com/) y en Google Cloud Platform, y los resultados se descargaron para análisis a un ambiente local.
 
## Resultados y conclusiones

Resultados completos se pueden consultar en este [notebook](analysis/lmevME-LM-Analysis_v1.ipynb) o [HTML](doc/lmevME-LM-Analysis_v1.html)

- **Los modelos de lenguaje en español tienen un mejor desempeño que los modelos en inglés,** donde los primerios tuvieron en promedio BAC=0.7434 vs 0.7190 de los segundos. Es interesante ver también el BAC de cada experimento.

![bac_by_lang.png](img/bac_by_lang.png)

- **Los modelos normalizados (en español) tuvieron los mejores resultados.** En la gráfica abajo podemos observar los resultados promedio de cada modelo.

![bac_by_lm.png](img/bac_by_lm.png)

- Es interesante ver que **en los modelos normalizados, el modelo de 50 dimensiones tuvo mejor desempeño que el modelo de 128 dimensiones.** En la siguiente gráfica de *loss vs bac* vemos que el modelo de 50 dimensiones tuvo una menor variación que el modelo de 128 dimensiones. Sin embargo en los demás casos los modelos de 128 dimensiones tuvieron mejor desempeño que los de 50 dimensiones. Sería útil evaluar mas a profundidad y determinar si las optimizaciones a hiperparámetros pudieran explicar esa diferencia.

![loss_vs_bac.png](img/loss_vs_bac.png)

- Este último punto se puede reforzar observando las curvas de aprendizaje donde podemos ver que el modelo 128dim normalizado tuvo una menor pérdida (loss) pero que comenzó a sobreajustar antes, por lo que una cuidadosa optimización de hiperparámetros podría ayudarle a tener un menor desempeño

![loss_training_curve.png](img/loss_training_curve.png)

- Los experimentos se corrieron en el ambiente de [Google Colaboratory](https://colab.research.google.com/) el cual nos dió dos distintos tipos de GPU, en algunas ocasiones una Tesla T4 y en otra Tesla P100. Con ello podemos observar las siguientes conclusiones:

	- Los modelos de 50 dimensiones entrenan aproximadamente en la mitad del tiempo de los de 128 dimensiones
	- Los modelos entrenados en GPU Tesla P100 entrenan aproximadamente en la mitad del tiempo de las GPU Tesla T4
	- No hay una gran diferecia entre los modelos normalizados y no normalizados

![bac_vs_time.png](img/bac_vs_time.png)

## Siguientes Pasos

Iremos agregando una comparación equivalente con otros modelos de lenguaje facilmente adaptable a esta tarea. Tambien probaremos optimizaciones de hiperparámetros de los mejores models para buscar llegar a los mejores modelos. Finalmente se propondrán aplicaciones reales de estos modelos.


# Spanish Language Model Evaluation for Multi-class Classification Tasks

## Objective

We have developed a comparision study between several language models that are easy to implement in text classification tasks using deep neural networks. These models are available trough [Tensorflow Hub](https://tfhub.dev/s?module-type=text-embedding) and can be used very easily as a text embeddings layer input to a multi-layer perceptron (MLP). The objective of this comparison study is to create and overview of the characteristics and advantages of each model for tasks in Spanish. As a proxy task we use a simple multiclass classification task of short texts hoping that some generalizations found can be applied to other natural language processing tasks such as sentiment analysis, textual similarity o named-entity recognition. We also expect that the classifier model can be reused, adapted or extended to other tasks.

## Models and Data

### Language Models

To this moment we have evaluated models that belong to the [Neural-Net Language Models](https://tfhub.dev/google/collections/nnlm/1) family, trained in Spanish using Spanish Google News 50B corpus. These models are in several versions: with 50 or 128 dimensional embeddings and with or without text normalization. We also evaluated it's equivalent models trained in English to get a sense of the benefit of using language specific models.

Evaluated models:

- nnlm-en-dim128
- nnlm-en-dim128-with-normalization
- nnlm-en-dim50
- nnlm-en-dim50-with-normalization
- nnlm-es-dim128
- nnlm-es-dim128-with-normalization
- nnlm-es-dim50
- nnlm-es-dim50-with-normalization

### Dataset and classification task

The classification task that we are using as benchmark is the one proposed by the [MercadoLibre Data Challenge 2019](https://ml-challenge.mercadolibre.com/). It consists of a dataset of 10 million ad titles classified in more than 1,500 categories. [Balanced Accuracy Score](https://ml-challenge.mercadolibre.com/rules) is used as evaluation metric as required by the original challenge. As the [dataset](https://ml-challenge.mercadolibre.com/downloads) contains half of the data in Spanish and half in portuguese, only the first langague records was used. In the classification tasks we used a random subset of 1 million records as training set with a 5% of them used as validation set. 25,000 more records were used as test set.

### Neural Network Architecture

The neural network connects the output of the language model as input embeddings to a 3 layer multilayer perceptron. Each of the 2 first dense layers has 512 units and the last one has as many units as the total number of classes of the classifier. Each dense layer is connected to the next through a Dropout layer. The whole network was trained with Adam optimizer with Keras default values and Sparse Categorical Crossentropy loss. The batch size for training was set to 4096 for all the tests. Each model was trained for a maximum of 10 epochs but applying EarlyStopping with patience set to 3, and in several cases the training did stop earlier. We did not any further hyperparameter optimization as the objective of the study is to get a general idea of the behaviour of the language models and not the optimal classifier for each of them.

TODO: diagrams and explanation

### Experiments execution

In this [notebook](https://github.com/eduardofv/lang_model_eval/blob/master/analysis/NNLM_50_es-v1_0.ipynb) is an example of the experiment process. It's worth mentioning that the final experiments were run using batch processes and not this specific notebook. The results of each experiment can be seen [here](https://github.com/eduardofv/lang_model_eval/blob/master/experiments).

Each model was run 4 times (trials). In the analysis results were averaged where it made sense. Each trial generated a metadata file with the experiment configuration and results obtained. Those can be seen in the [same link](https://github.com/eduardofv/lang_model_eval/blob/master/experiments).

The experiments were run using a framework developed ad-hoc in [Google Colaboratory](https://colab.research.google.com/) and Google Cloud Platform. The results were downloaded to a local enviroment for analysis.

## Results and conclusion

The complete result analysis can be seen on this [notebook](https://github.com/eduardofv/lang_model_eval/blob/master/doc/lmevME-LM-Analysis_v1.html) or this [HTML](https://github.com/eduardofv/lang_model_eval/blob/master/doc/lmevME-LM-Analysis_v1.html).

- Not surprisingly **language models in Spanish have a better performance than models in English.** The first ones obtained an average BAC=0.7434 and the second ones got an average BAC=0.7190. It's also interesting to see the individual results in the aforementioned notebook.

![bac_by_lang.png](img/bac_by_lang.png)

- **Normalized models (in Spanish) have the best results.** In the next figure shows average results for each language model.

![bac_by_lm.png](img/bac_by_lm.png)

- It's also worth noting that **in normalized models, the 50 dimensions model had a better performance than the 128 dimensions model.** The next figure *loss vs bac for all trials* shows that the Spanish, normalized, 50 dimensions model BAC had a lower variance than the 128 dimensions equivalent despite the fact that the overall best model was 128 dim. But in general the 128 dimensions models had a better performance than their 50 dimensions equivalents. It could be useful to perform further study if hyperparameter optimization could explain that difference.

![loss_vs_bac.png](img/loss_vs_bac.png)

- This last point can be reinforced watching the learning curves where it can be observed that the 128dim Spanish normalized model had a lower loss but started to overfit earlier. This suggest that a careful hyperparameter optimization may help to improve it's overall performance.

![loss_training_curve.png](img/loss_training_curve.png)

- These experiments were run in [Google Colaboratory](https://colab.research.google.com/) in which is not possible to ensure the kind of GPU that will be available. Fortunately this can be used to observe the differences in performance of training time each has. For these experiments we got Tesla T4 and Tesla P100 GPUs. We can observe the following:

	- 50 dimensions models train in about half the time of their 128 dimensions equivalents.
	- Equivalent models trained in Tesla P100 GPUs train in about half the time as those trained on Tesla T4 GPUs.
	- There is little difference in training time between normalized and non-normalized models.

![bac_vs_time.png](img/bac_vs_time.png)

## Next Steps

We will be adding comparisons with other language models that could be easily used instead of those previously evaluated. We will also run hyperparameter optimization for some of the models to search for best models for this specific task. Finally we will propose practical real world applications for these models.

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



