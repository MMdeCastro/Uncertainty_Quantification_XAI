# Exactitud con certidumbre

Charla/taller para la [PyConES22](https://2022.es.pycon.org/)

Medir la calidad de las conclusiones de un modelo a través de métricas como accuracy (exactitud), precision, recall,... no nos da la seguridad de que el modelo esté respondiendo a la pregunta correcta, sobre todo si es un modelo no interpretable por las personas, también llamados modelos black box (caja negra). 

Un ejemplo típico son los grandes modelos de deep learning fallando al intentar reconocer una vaca en la playa. Esto sucede frecuentemente porque lo que en realidad aprendió el modelo durante su entrenamiento fue a detectar la hierba en las imágenes del training set. Son modelos muy exactos, pero en una tarea distinta, no deseada (queríamos identificar vacas, no hierba). Si solamente nos fijamos en las métricas, este inesperado cambio de tarea puede pasar fácilmente desapercibido y aumentar el tamaño del training set no necesariamente soluciona el problema.

Podemos cerciorarnos de que esto no nos pasa incorporando a nuestro modelo herramientas matemáticas (ya desarrolladas como librerías de Python) que van más allá de la optimización de las métricas. Estas herramientas sirven para cualquier modelo y proveen de intervalos de confianza a las predicciones puntuales (utilizando Conformal Predictors, Quantile Regression, Ensemble Learning,..) o explican en qué se fijó el modelo para producir los resultados (gracias a los SHAP values, LIME,...).

Esta charla/taller complementa a la mayoría de los cursos introductorios sobre aprendizaje automático supervisado, que a menudo se centran solamente en optimizar métricas.

## Instrucciones

1. Si no tienes el administrador de paquetes `conda` instalado, puedes simplemente instalar 'miniconda' siguiendo las instrucciones para tu sistema operativo [aquí](https://docs.conda.io/en/latest/miniconda.html). Después, abre una terminal (de Bash si estás en Linux o Mac, Anaconda prompt si estás en Windows) y escribe:

+ `conda env create -f environment.yml`

2. Activa el entorno escribiendo en la terminal:

+ `conda activate pycones22`

  Cuando acabes con este proyecto, desactiva el entorno escribiendo en la terminal:

+ `conda deactivate`

3. Abre la aplicación para editar y ejecutar el código en Jupyter Notebooks:

+ `jupyter notebook`

4. Jupyter se abrirá en tu browser. En la barra de herramientas, pincha en 'Nbextensions' y permite 'Collapsible Headings' para mejorar la navegación, es un Jupyter Notebook un poco largo! Abre el Jupyter Notebook clicando en el fichero con extensión .ipynb y sigue las instrucciones escritas all. 


