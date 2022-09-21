
[![NBViewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/MMdeCastro/Uncertainty_Quantification_XAI/blob/main/Exploration_and_Classification.ipynb)

# Exactitud con certidumbre

_Taller para la [PyConES22](https://2022.es.pycon.org/) viernes 30 de Septiembre de 2022 de 15:30h a 17:30h._

Medir la calidad de las predicciones de un modelo de aprendizaje supervisado a través de métricas de rendimiento como *accuracy* (o "exactitud" en castellano), *precision*, *recall*, *F1 score*,... no nos da la seguridad de que el modelo esté respondiendo a la pregunta correcta, sobre todo si es un modelo no interpretable por las personas, también llamados modelos 'black box' o caja negra. 

Un ejemplo típico son los grandes modelos de deep learning fallando al intentar identificar una vaca en la playa. Esto sucede frecuentemente porque lo que en realidad aprendió el modelo durante su entrenamiento fue a reconocer la hierba en las imágenes del training set. Son modelos muy exactos (puede que hasta obtengamos más de un 95% de *accuracy* en su validación y testado), pero en una tarea distinta, no deseada (queríamos detectar vacas, no hierba). Si solamente nos fijamos en las métricas, este inesperado cambio de tarea puede pasar fácilmente desapercibido y aumentar el tamaño del training set no necesariamente soluciona el problema.

Podemos cerciorarnos de que esto no nos pasa incorporando a nuestro modelo herramientas matemáticas (ya desarrolladas como librerías de Python) que nos facilitan ir más allá de la optimización de las métricas. Estas herramientas sirven para cualquier modelo supervisado y explican en qué se fijó el modelo para producir sus predicciones (gracias a métodos de explicabilidad o [XAI](https://en.wikipedia.org/wiki/Explainable_artificial_intelligence), como los SHAP values, LIME,...) o proveen de intervalos de confianza a las predicciones puntuales (utilizando métodos de cuantificación de incertidumbre o [UQ](https://en.wikipedia.org/wiki/Uncertainty_quantification), como los Conformal Predictors, la Quantile Regression,..).

Este taller complementa a la mayoría de los cursos introductorios sobre aprendizaje automático supervisado, que a menudo se centran solamente en optimizar métricas. Si no has realizado un curso introductorio sobre aprendizaje automático, encontrarás un resumen en el Jupyter Notebook llamado `Exploration_and_Classification.ipynb`.

## Instrucciones

0. Clona este repositorio en el ordenador o nube donde vayas a trabajar. Si no tienes el paquete `git` instalado, [aquí](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) explica cómo hacerlo. Para clonar el repositorio, pincha en el botón verde que dice 'Code', copia al portapapeles la dirección htpps, y pégala en una terminal donde vayas a trabajar, escribiendo `git clone`, un espacio, y la dirección htpps de este repositorio, tardará unos segundos en descargarse.  

1. Si no tienes el administrador de paquetes `conda` instalado, puedes simplemente instalar `miniconda` siguiendo las instrucciones para tu sistema operativo [aquí](https://docs.conda.io/en/latest/miniconda.html). Después, abre una terminal (de Bash si estás en Linux o Mac, Anaconda prompt si estás en Windows) y escribe:

+ `conda env create -f environment.yml`

2. Activa el entorno escribiendo en la terminal:

+ `conda activate pycones22_UQ_XAI`

  La primera vez tardará bastante, dependiendo de tu ancho de banda, puede que hasta una hora, porfa, tráelo ya hecho cuando vengas al taller. Cuando acabes con este proyecto, desactiva el entorno escribiendo en la terminal:

+ `conda deactivate`

3. Abre la aplicación para editar y ejecutar el código en Jupyter Notebooks:

+ `jupyter notebook`

4. Jupyter se abrirá en tu browser. En la barra de herramientas, pincha en 'Nbextensions' y permite 'Collapsible Headings' para mejorar la navegación, es un Jupyter Notebook un poco largo! Abre el Jupyter Notebook clicando en un fichero con extensión .ipynb y sigue las instrucciones escritas all. Aconsejamos empezar por `Exploration_and_Classification.ipynb`.

De hecho el análisis completo incluye los siguientes 3 notebooks, en este orden:

+ `Exploration_and_Classification.ipynb` donde exploramos y preparamos los datos, probamos varios modelos de aprendizaje automático supervisado para clasificación, explicamos las métricas de rendimiento, y elegimos el modelo que nos da mejor rendimiento (este Jupyter notebook se da por entendido y no se tratará en el taller de la PyConES22),
+ `XAI.ipynb` donde aplicamos varios métodos de explicabilidad [XAI](https://en.wikipedia.org/wiki/Explainable_artificial_intelligence), y
+ `UQ.ipynb` donde aplicamos varios métodos de cuantificación de incertidumbre [UQ](https://en.wikipedia.org/wiki/Uncertainty_quantification).


