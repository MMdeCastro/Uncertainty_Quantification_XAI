## Más allá de la métrica

Este taller complementa a la mayoría de los cursos introductorios sobre aprendizaje automático (machine learning) supervisado, que a menudo se centran solamente en optimizar métricas de rendimiento. Si no has realizado un curso introductorio sobre aprendizaje automático supervisado, encontrarás un resumen en el Jupyter Notebook llamado `Exploration_and_Classification.ipynb`.

El análisis completo es una serie de 3 Jupyter Notebooks, en este orden:

+ `Exploration_and_Classification.ipynb` donde exploramos y preparamos los datos, probamos varios modelos de aprendizaje automático supervisado para clasificación, explicamos las métricas de rendimiento y el umbral de decisión, y elegimos el modelo que nos da mejor rendimiento,
+ `XAI.ipynb` donde aplicamos varios métodos de explicabilidad o [XAI](https://en.wikipedia.org/wiki/Explainable_artificial_intelligence), y
+ `UQ.ipynb` donde aplicamos varios métodos de cuantificación de incertidumbre o [UQ](https://en.wikipedia.org/wiki/Uncertainty_quantification).

para profundizar en la aplicación de la UQ, ver el cuarto Jupyter Notebook `UQ_multiclass.ipynb` donde se incluye el código para la explicación de este [mini curso de Christoph Molnar](https://mindfulmodeler.substack.com/p/week-1-getting-started-with-conformal) y el quinto Jupyter Notebook `UQ_TimeSeriesClassification.ipynb` donde aplicamos UQ a la clasificación de series temporales.

Si prefieres solo mirar el contenido sin ejecutarlo, aquí tienes los enlaces a cada uno de los temas:

[![NBViewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/MMdeCastro/Uncertainty_Quantification_XAI/blob/main/Exploration_and_Classification.ipynb) Exploración de los datos y selección del modelo de aprendizaje supervisado para la Clasificación

[![NBViewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/MMdeCastro/Uncertainty_Quantification_XAI/blob/main/XAI.ipynb) Herramientas de Explicabilidad (XAI)

[![NBViewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/MMdeCastro/Uncertainty_Quantification_XAI/blob/main/UQ.ipynb) Herramientas de Cuantificación de Incertidumbre (UQ)

[![NBViewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/MMdeCastro/Uncertainty_Quantification_XAI/blob/main/UQ_multiclass.ipynb) Conformal Predictors (CP) para la UQ en ejemplo de clasificación multiclase

[![NBViewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/MMdeCastro/Uncertainty_Quantification_XAI/blob/main/UQ_TimeSeriesClassification.ipynb) Conformal Predictors (CP) para la UQ en un ejemplo de clasificación de series temporales

y el enlace a las diapositivas de las presentaciones introductorias (contienen animaciones, mejor verlas en modo presentación):
+ para todos los públicos ["La IA no hereda nuestros sesgos si le enseñamos cómo"](https://docs.google.com/presentation/d/1bp8rJTtZ5aAGeNwwdTdue4vGcs27QvcEeA1VbHPce9c/edit#slide=id.g211627b4636_0_101) en las II Jornadas de Ciencia y Género del IFIC y la Univ. de Valencia. [Aquí está el vídeo](https://youtu.be/89G74PBnoVc) de la presentación,
+ para estudiantes de informática y telecomunicaciones del Google Developer Student Club de la Univ. de Granada ["Herramientas para la explicabilidad y la cuantificación de incertidumbre contra los sesgos del aprendizaje automático"](https://docs.google.com/presentation/d/1p5QVf4JaDDFl7IM6XLclfWbcU0a5a8Il9MTu8-9fB84/edit#slide=id.g20f7f9abf3c_0_101),
+ para Python Granada ["Python para una IA responsable"](https://docs.google.com/presentation/d/1kil2P6pYuKcan1QcvaxNWXoyoEXkJnBWgpt0gZMfejg/edit?usp=sharing),
+ sobre los [Conformal Predictors](https://docs.google.com/presentation/d/1Q6oxcgmNv0GsmNFA5npAzQGDuy_XolBKStN4KUhP4Gk/edit?usp=sharing) (in English).

Como decíamos, ésta es una serie introductoria y no incluimos explicaciones exhaustivas ni demostraciones matemáticas (hay muchas otras fuentes, ver la lista de materiales en la intro de los Jupyter Notebooks), mejor mencionaremos algunas características intuitivas sobre la explicabiliad y la cuantificación de la incertidumbre y nos enfocaremos en su implementación en Scikit-learn (una de las librerías de [Python](https://es.wikipedia.org/wiki/Python) más utilizadas en aprendizaje automático) y paquetes compatibles.

# Exactitud con certidumbre

Medir la calidad de las predicciones de un modelo de aprendizaje automático supervisado a través de métricas de rendimiento como *accuracy* (o "exactitud" en castellano), *precision*, *recall*, *F1 score*, *MSE*, *RMSE*,... no nos da la seguridad de que el modelo esté respondiendo a la pregunta correcta, sobre todo si es un modelo no interpretable por las personas, también llamados modelos de caja opaca ('black box models'). 

Un ejemplo típico son los grandes modelos de aprendizaje profundo (deep learning) fallando al intentar identificar una vaca en la playa. Esto sucede frecuentemente porque lo que en realidad aprendió el modelo durante su entrenamiento fue a reconocer la hierba en las imágenes del conjunto de entrenamiento (training set). Son modelos muy exactos (puede que hasta obtengamos más de un 95% de *accuracy* en su validación y testado), pero en una tarea distinta, no deseada (queríamos detectar vacas, no hierba). Si solamente nos fijamos en las métricas de rendimiento, este inesperado cambio de tarea puede pasar fácilmente desapercibido y aumentar el tamaño del training set no necesariamente soluciona el problema.

Podemos cerciorarnos de que esto no nos pasa incorporando a nuestro modelo ya entrenado ciertas herramientas para que pueda indicarnos cuándo su predicción es fiable y cuándo no. Estas herramientas matemáticas (ya desarrolladas como librerías de Python) y explican en qué se fijó el modelo para producir sus predicciones (gracias a métodos de explicabilidad o [XAI](https://en.wikipedia.org/wiki/Explainable_artificial_intelligence), como los SHAP values, LIME,...) o nos proveen de probabilidades calibradas y rangos de certidumbre en lugar de predicciones heurísticas y puntuales (utilizando métodos de cuantificación de incertidumbre o [UQ](https://en.wikipedia.org/wiki/Uncertainty_quantification), como los Conformal Predictors). Y todas son herramientas post-hoc (no hay que volver a entrenar el modelo), model-agnóstic (sirven para cualquier modelo), y muy ligeras (se implementan con pocas líneas de código y se ejecutan muy rápido).

<font size="10"> 👍🤓 </font>

## Instrucciones

0. Clona este repositorio en el ordenador o nube donde vayas a trabajar. Si no tienes el paquete `git` instalado, [aquí](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) explica cómo hacerlo. Para clonar el repositorio, pincha en el botón verde que dice 'Code', copia al portapapeles la dirección htpps, y pégala en una terminal donde vayas a trabajar, escribiendo `git clone`, un espacio, y la dirección htpps de este repositorio, tardará unos segundos en descargarse.  

1. Si no tienes el administrador de paquetes `conda` instalado, puedes simplemente instalar `miniconda` siguiendo las instrucciones para tu sistema operativo [aquí](https://docs.conda.io/en/latest/miniconda.html). Al instalar `miniconda` se instala `Python` también, viene incluido. Después, abre una terminal (de Bash si estás en Linux o Mac, Anaconda Prompt si estás en Windows) y escribe:

+ `conda env create -f environment.yml`

  ⚠️ Crear el entorno por primera vez dura unos 10 minutos pero puede que más, porfa, tráelo ya hecho cuando vengas al taller. 

2. Activa el entorno escribiendo en la terminal:

+ `conda activate intro_UQ_XAI`

  Cuando acabes con este proyecto, desactiva el entorno escribiendo en la terminal:

+ `conda deactivate`

3. Con el entorno activo, vamos a instalar los paquetes de XAI `LIME` y `SHAP` y el paquete de UQ `MAPIE` que es mejor instalarlos via `pip` en lugar de usar `conda`, tardarán unos pocos minutos en instalarse, para ello escribe en la terminal

+ `python3 -m pip install lime shap MAPIE`

4. Siempre con el entorno activo, abre la aplicación de Jupyter para editar y ejecutar el código en Jupyter Notebooks escribiendo en la terminal:

+ `jupyter notebook`

Jupyter se abrirá en el browser que tengas por defecto (Firefox, Chrome,...). Los Jupyter Notebooks son los ficheros con extensión .ipynb, se abren clicando en ellos. Sigue las instrucciones escritas allí. Aconsejamos empezar por `Exploration_and_Classification.ipynb`. 

<font size="10"> 📝 </font>Este taller se realizó por primera vez en la [PyConES22](https://2022.es.pycon.org/) viernes 30 de Septiembre de 2022 de 15:30h a 17:30h. En el taller de la PyConES22 damos este Jupyter Notebook por sabido y empezamos directamente con `XAI.ipynb`. Todas las charlas de la PyConES 2022 que se mencionan como material complementario en los Jupyter Notebooks pueden encontrarse en [la lista de reprodución del canal de youtube de Python Espana](https://www.youtube.com/@PythonES). 
