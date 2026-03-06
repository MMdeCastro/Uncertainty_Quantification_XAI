## Más allá de la métrica

Este taller complementa a la mayoría de los cursos introductorios sobre aprendizaje automático (machine learning) supervisado, que a menudo se centran solamente en optimizar métricas de rendimiento. Si no has realizado un curso introductorio sobre aprendizaje automático supervisado, encontrarás un resumen en el Jupyter Notebook llamado `Exploration_and_Classification.ipynb`.

El análisis completo es una serie de 4 Jupyter Notebooks, que aunque son independientes, mejor trabajarlos en este orden:

+ `Exploration_and_Classification.ipynb` donde exploramos y preparamos los datos, probamos varios modelos de aprendizaje automático supervisado para clasificación, explicamos las métricas de rendimiento y el umbral de decisión, y elegimos el modelo que nos da mejor rendimiento,
+ `XAI.ipynb` donde explicamos métodos de explicabilidad o [XAI](https://en.wikipedia.org/wiki/Explainable_artificial_intelligence), y
+ `UQ.ipynb` donde explicamos métodos de cuantificación de incertidumbre o [UQ](https://en.wikipedia.org/wiki/Uncertainty_quantification).

Para profundizar en la aplicación de la UQ, ver el cuarto Jupyter Notebook `UQ_multiclass.ipynb` donde se incluye el código para la explicación de este [mini curso de Christoph Molnar](https://mindfulmodeler.substack.com/p/week-1-getting-started-with-conformal).

Si prefieres solo mirar el contenido sin ejecutarlo, aquí tienes los enlaces a cada uno de los temas:

[![NBViewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/MMdeCastro/Uncertainty_Quantification_XAI/blob/main/Exploration_and_Classification.ipynb) Exploración de los datos y selección del modelo de aprendizaje supervisado para la Clasificación.

[![NBViewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/MMdeCastro/Uncertainty_Quantification_XAI/blob/main/XAI.ipynb) Herramientas de Explicabilidad (XAI).

[![NBViewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/MMdeCastro/Uncertainty_Quantification_XAI/blob/main/UQ.ipynb) Herramientas de Cuantificación de Incertidumbre (UQ) y aplicación de la Conformal Prediction para una clasificación binaria.

[![NBViewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/MMdeCastro/Uncertainty_Quantification_XAI/blob/main/UQ_multiclass.ipynb) Conformal Predictors (CP) para la UQ en ejemplo de clasificación multiclase.

Aquí el enlace a las diapositivas de las presentaciones introductorias (contienen animaciones, mejor verlas en modo presentación):
+ Cuarto Real memorial Mariano Martín García ["La Inteligencia Artificial y sus sesgos"](https://docs.google.com/presentation/d/13o2Y8BrKGZR7PKA5Ia4UPwE87jRJ6Dxaqk3QCPXkWpU/edit?slide=id.p#slide=id.p).
+ Python Asturias ["Python para una IA Confiable"](https://docs.google.com/presentation/d/1FCUoBORSBP7cE0LYmVVQr6daqhCQ_ZJ5CPFltuPZlsQ/edit#slide=id.g2988a3002ed_0_0), y aquí está [el vídeo](https://youtu.be/sPCQVVFBuQc?si=jh3lhjFl9ZEb0fu3).
+ PyConES 2024 ["Predicción Conforme: el fin de la predicción puntual descalibrada"](https://pretalx.com/pycones-2024/talk/HSWECW/), en la descripción está el enlace a la presentación,
+ CITIC - UGR ["Accuracy with Certainty: beyond the performance metrics"](https://docs.google.com/presentation/d/1pXtkpzjTj94vxgBhRlsuSFIwFkx8v53u8ZP_9lZDyi4/edit?usp=sharing),
+ DyploFest24 ["Exactitud con Certidumbre"](https://docs.google.com/presentation/d/1f2LiOLh_IQfKqGiJ1X8OIee0wxKJQZiKft5riam9UA8/edit#slide=id.g2de3d8b8587_0_17),
+ para el alumnado del Máster en Ciencia de Datos de la Universidad de Granada ["Introducción a la Cuantificación de Incertidumbre en Aprendizaje Automático con Conformal Prediction"](https://docs.google.com/presentation/d/1yFHn4_Byt6_f5arFOdhUWBXOrN7BrYbmEc2gMhJ0RVY/edit?usp=sharing),
+ II Jornadas de Ciencia y Género del IFIC y la Univ. de Valencia, charla para todos los públicos ["La IA no hereda nuestros sesgos si le enseñamos cómo"](https://docs.google.com/presentation/d/1bp8rJTtZ5aAGeNwwdTdue4vGcs27QvcEeA1VbHPce9c/edit#slide=id.g211627b4636_0_101). [Aquí está el vídeo](https://youtu.be/89G74PBnoVc) de la presentación,
+ para estudiantes de informática y telecomunicaciones del Google Developer Student Club de la Univ. de Granada ["Herramientas para la explicabilidad y la cuantificación de incertidumbre contra los sesgos del aprendizaje automático"](https://docs.google.com/presentation/d/1p5QVf4JaDDFl7IM6XLclfWbcU0a5a8Il9MTu8-9fB84/edit#slide=id.g20f7f9abf3c_0_101),
+ Python Granada ["Python para una IA responsable"](https://docs.google.com/presentation/d/1kil2P6pYuKcan1QcvaxNWXoyoEXkJnBWgpt0gZMfejg/edit?usp=sharing),
+ sobre los Conformal Predictors:
  + [Intro general](https://docs.google.com/presentation/d/1Q6oxcgmNv0GsmNFA5npAzQGDuy_XolBKStN4KUhP4Gk/edit?usp=sharing) (in English).
  + [Key Concepts of Conformal Prediction](https://docs.google.com/presentation/d/1bQYIFyQysQPx79wJq1mltylsH_aeO2AnLYRlHiLq6vo/edit#slide=id.g2dc32e8043a_0_5) (in English).

Como decíamos, ésta es una serie introductoria y no incluimos explicaciones exhaustivas ni demostraciones matemáticas (hay muchas otras fuentes, ver la lista de materiales en la intro de los Jupyter Notebooks), mejor mencionaremos algunas características intuitivas sobre la explicabiliad y la cuantificación de la incertidumbre y nos enfocaremos en su implementación en Scikit-learn (una de las librerías de [Python](https://es.wikipedia.org/wiki/Python) más utilizadas en aprendizaje automático) y paquetes compatibles.

# Exactitud con certidumbre

En inteligencia artificial, medimos la calidad de las predicciones de un modelo de aprendizaje automático supervisado a través de métricas de rendimiento. Por ejemplo, la métrica de exactitud (accuracy) en una tarea de clasificación cuenta cuántas veces el modelo nos dió la solución correcta. El error cuadrático medio (MSE) en una tarea de regresión mide a qué distancia se quedó el modelo de la solución correcta. Sin embargo, las métricas de rendimiento no nos ofrecen certeza sobre las predicciones del modelo y esto es un gran problema, sobre todo si es un modelo no interpretable por las personas, también llamados modelos de caja opaca (‘black box models’).

Un ejemplo es el de un modelo utilizado para clasificar radiografías de tórax según la gravedad de una neumonía. Había detectado una marca de la placa cuando la radiografía se había realizado en la planta UCI y ese factor circunstancial era el que primaba en su predicción  ([_"AI for radiographic COVID-19 detection selects shortcuts over signal"_](https://www.nature.com/articles/s42256-021-00338-7)). Eran un modelo muy exacto, con una métrica de exactitud de más del 90% de aciertos, pero en una tarea distinta, no deseada (queríamos detectar una enfermedad, no de dónde venía la radiografía). El modelo simplemente encontró una correlación espuria y cogió el atajo. Y sin poder avisarnos, arrojó predicciones sesgadas.

Como segundo ejemplo imaginemos otro modelo clasificador, esta vez entrenado para predecir si llueve (clase 1) o no llueve (clase 0) dada la temperatura y la humedad del día. Si cuando lo ponemos en producción el resultado de un día es “0.8”, muchos libros, cursos,… concluyen que eso significa que “hay un 80% de probabilidad de que llueva ese día y un 20% de probabilidad de que no llueva ese día”. Sin embargo, esos resultados no están calibrados, es decir, no tienen en cuenta cuántos días llovió de verdad, y por tanto no son verdaderas probabilidades. Pero la cosa empeora, ¿qué ocurre si lo que le mostramos al modelo son los datos de un día con temperatura y humedad muy distintas a las que se usaron en su entrenamiento? Por ejemplo, las de un día en el planeta Venus. El modelo no tiene manera de decirnos “no tengo ni idea de qué es esto”, simplemente arrojará un resultado, que en el mejor de los casos estará cerca de “0.5” pero no hay ninguna garantía de que así sea. Y se quedará tan pancho, porque no le hemos dado herramientas para mostrar la incertidumbre.

Al automatizar predicciones, si solamente nos fijamos en las métricas de rendimiento, estos problemas pueden pasar fácilmente desapercibidos y aumentar el tamaño del training set no necesariamente los soluciona. Podemos cerciorarnos de que esto no nos pasa incorporando a nuestro modelo ya entrenado ciertas herramientas sencillas para que pueda indicarnos cuándo su predicción es fiable y cuándo no. Estas herramientas matemáticas (ya desarrolladas como librerías de Python y R) explican en qué se fijó el modelo para producir sus predicciones (gracias a métodos de explicabilidad o “XAI”, como los SHAP values, LIME,…) o nos proveen de probabilidades calibradas e intervalos de predicción que incluyen con certeza el valor verdadero para que el modelo pueda avisarnos cuando no debemos confiar en su predicción (gracias a métodos de cuantificación de incertidumbre o “UQ”, como los Conformal Predictors). Y todas son herramientas post-hoc (no hay que volver a entrenar el modelo), model-agnostic (sirven para cualquier modelo y tarea), y muy ligeras (se implementan con pocas líneas de código y se ejecutan muy rápido).

<font size="10"> 👍🤓 </font>

## Instrucciones
Teniendo Python 3 instalado, hay que seguir estas instrucciones:

0. Clonar el repositorio del taller con el paquete `git`

Para poder descargar los programas, "scripts", o “notebooks” donde se encuenta el código de este taller (tienen extensión .ipynb porque son Jupyter Notebooks) hay que clonar este repositorio en el ordenador o nube donde vayas a trabajar y para ello necesitamos instalar el paquete de control de versiones `git`, [aquí](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) explica cómo hacerlo si no estuviera ya instalado. Una vez instalado `git`, pincha en el botón verde con la palabra "Code" que encontrarás en esta página arriba a la derecha, copia al portapapeles la dirección https, abre una Terminal, navega a la carpeta donde quieras trabajar y péga la dirección copiada en el portapapeles en la Terminal escribiendo antes `git clone`, un espacio, y después dichs dirección https, es decir, hay que escribir y ejecutar:

`git clone https://github.com/MMdeCastro/Uncertainty_Quantification_XAI.git`  

1. Crear entorno con el paquete `venv` 

Siempre es recomendable crear un entorno para evitar incompatibilidad de versiones. Abrí una Terminal nueva y al iniciar estaba en mi carpeta de usuaria en home y he seguido los pasos indicados en la documentación de Python https://docs.python.org/es/3/tutorial/venv.html (yo seguí las instrucciones para mi sistema operativo Ubuntu 24.04 con Python 3.12 pero ahí se indican las instrucciones para otros sistemas operativos):

+ Primero instalé el paquete `venv` escribí y ejecuté en la Terminal los comandos:

`sudo apt install python3-venv`

+ Después con `mkdir` creé una carpeta para guardar los entornos, he elegido ocultarla, por eso lleva un punto delante del nombre y he elegido colocarla aquí mismo donde estoy, en mi home, así que escribí y ejecuté:

`mkdir .venv`

+ Luego creé el entorno de `venv` llamado "intro_trustAI_venv" e indiqué que lo quería alojar en la carpeta `.venv`que acababa de crear, así que  escribí y ejecuté:

`python3 -m venv .venv/intro_trustAI_env`

+ Por último, activé el entorno escribiendo y ejecutando:

`source .venv/intro_trustAI_venv/bin/activate`

+ Una vez activado el entorno, instalé el paquete `pip` para instalar el resto de paquetes que vamos a utilizar en el taller, así que escribí y ejecuté:

`sudo apt install python3-pip`

2. Instalar paquetes dentro del entorno
   
+ Con el entorno activo, instalé los paquetes necesarios para el taller navegando a la carpeta donde cloné el repositorio de GitHub (ver instrucciones en esta página más arriba) y escribiendo y ejecutando:

`pip install -r requirements.txt`

3. Abrir y ejecutar el código del taller
   
+ Para ver el código de los Jupyter Notebooks, escribí y ejecuté:

`Jupyter Notebook`

  y se abre un gestor de archivos en el navegador web que tengas por defecto (el mío es Chrome). Se recomienza comenzar por el Jupyter Notebook llamando “Exploration_and_Classification.ipnyb” que explica cómo usar Jupyter Notebooks.

+ Cuando termine, saldré del entorno simplemente escribiendo y ejecutando

`deactivate`

+ La siguiente vez que vuelva a trabajar con el código de este taller sólo tendré que activar el entorno escribiendo y ejecutando esto en una terminal

`source .venv/intro_trustAI_venv/bin/activate`

y no tendré que volver a instalar nada.


Nota: Muchos de los vídeos mencionados en los Jupyter Notebooks son de la PyConES22, donde se impartió este taller por primera vez, y pueden encontrarse en [la lista de reprodución del canal de youtube de Python Espana](https://www.youtube.com/@PythonES). 
