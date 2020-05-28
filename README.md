# MinimizacionCostes_IA

## Caso Práctico: Minimización de Costes en el Consumo Energético de un Centro de Datos

### Problema a resolver
 
Configurar el entorno de un servidor y construir una IA que controlará el enfriamiento / calentamiento del servidor 
para que se mantenga en un rango óptimo de temperaturas mientras se ahorra la máxima energía, minimizando así los costes.

Se utiliza un modelo IA DQN (Deep Q-Learning) y el objetivo será lograr al menos un 40% de ahorro de energía.

### Definción del entorno

Antes de definir los estados, las acciones y las recompensas, vamos a explicar cómo funciona el servidor. Primero, enumeraremos todos los parámetros y variables del entorno por los cuales se controla el servidor. 

#### Parámetros

- la temperatura atmosférica promedio durante un mes el rango óptimo de temperaturas del servidor, que será  (18∘C,24∘C)
- la temperatura mínima del servidor por debajo de la cual no funciona, que será  20∘C
- la temperatura máxima del servidor por encima de la cual no funciona, que será de  80∘C
- el número mínimo de usuarios en el servidor, que será 10
- el número máximo de usuarios en el servidor, que será de 100
- el número máximo de usuarios en el servidor que puede subir o bajar por minuto, que será 5
- la tasa mínima de transmisión de datos en el servidor, que será 20
- la velocidad máxima de transmisión de datos en el servidor, que será de 300
- la velocidad máxima de transmisión de datos que puede subir o bajar por minuto, que será 10

#### Variables

- la temperatura del servidor en cualquier momento
- la cantidad de usuarios en el servidor en cualquier momento
- la velocidad de transmisión de datos en cualquier minuto
- la energía gastada por la IA en el servidor (para enfriarlo o calentarlo) en cualquier momento
- la energía gastada por el sistema de enfriamiento integrado del servidor que automáticamente lleva la temperatura del servidor al rango óptimo cada vez que la temperatura del servidor sale de este rango óptimo

Todos estos parámetros y variables serán parte de nuestro entorno de servidor e influirán en las acciones de la IA en el servidor.

A continuación, expliquemos los dos supuestos básicos del entorno. Es importante comprender que estos supuestos no están relacionados con la inteligencia artificial, sino que se utilizan para simplificar el entorno para que podamos centrarnos al máximo en la solución de inteligencia artificial.

#### Suposiciones:

Nos basaremos en los siguientes dos supuestos esenciales:

Supuesto 1: la temperatura del servidor se puede aproximar mediante Regresión lineal múltiple, mediante una función lineal de la temperatura atmosférica, el número de usuarios y la velocidad de transmisión de datos:
Supongamos que después de realizar esta Regresión lineal múltiple, obtuvimos los siguientes valores de los coeficientes:  

<div align="center">temp. del server = temp. atmosf. + 1.25 x n. de usuarios + 1.25 x ratio de transf. de datos</div>





Supuesto 2: la energía gastada por un sistema (nuestra IA o el sistema de enfriamiento integrado del servidor) que cambia la temperatura del servidor de  Tt a  Tt+1 en 1 unidad de tiempo (aquí 1 minuto), se puede aproximar nuevamente mediante regresión mediante una función lineal del cambio absoluto de temperatura del servidor:

<div align="center">Et=|ΔTt|=|Tt+1−Tt|</div>

{Tt+1−Tt si Tt+1>Tt, es decir, si el servidor se calienta

 Tt−Tt+1 si Tt+1<Tt, es decir, si el servidor se enfria}

#### Simulación

El número de usuarios y la velocidad de transmisión de datos fluctuarán aleatoriamente para simular un servidor real. Esto lleva a una aleatoriedad en la temperatura y la IA tiene que entender cuánta potencia de enfriamiento o calefacción tiene que transferir al servidor para no deteriorar el rendimiento del servidor y, al mismo tiempo, gastar la menor energía optimizando su transferencia de calor.

### Funcionamiento general

Dentro de un centro de datos, estamos tratando con un servidor específico que está controlado por los parámetros y variables enumerados anteriormente. Cada minuto, algunos usuarios nuevos inician sesión en el servidor y algunos usuarios actuales cierran sesión, por lo tanto, actualizan el número de usuarios activos en el servidor. Igualmente, cada minuto se transmiten algunos datos nuevos al servidor, y algunos datos existentes se transmiten fuera del servidor, por lo tanto, se actualiza la velocidad de transmisión de datos que ocurre dentro del servidor. Por lo tanto, según el supuesto 1 anterior, la temperatura del servidor se actualiza cada minuto. 
**Dos posibles sistemas pueden regular la temperatura del servidor: la IA o el sistema de enfriamiento integrado del servidor.**

El sistema de enfriamiento integrado del servidor es un sistema no inteligente que automáticamente devolverá la temperatura del servidor a su temperatura óptima: cuando la temperatura del servidor se actualiza cada minuto, puede mantenerse dentro del rango de temperaturas óptimas (18∘C,24∘C), o salir de este rango. Si sale del rango óptimo, como por ejemplo 30∘C, el sistema de enfriamiento integrado del servidor llevará automáticamente la temperatura al límite más cercano del rango óptimo, que es  24∘C. Sin embargo, el sistema de enfriamiento integrado de este servidor lo hará solo cuando la IA no esté activada. 

Si la IA está activada, en ese caso el sistema de enfriamiento integrado del servidor se desactiva y es la IA la que actualiza la temperatura del servidor para regularlo de la mejor manera. Pero la IA hace eso después de algunas predicciones previas, no de una manera determinista como con el sistema de enfriamiento integrado del servidor no inteligente. Antes de que haya una actualización de la cantidad de usuarios y la velocidad de transmisión de datos que hace que cambie la temperatura del servidor, la IA predice si debería enfriar el servidor, no hacer nada o calentar el servidor. Entonces ocurre el cambio de temperatura y la IA reitera. Y dado que estos dos sistemas son complementarios, los evaluaremos por separado para comparar su rendimiento.

El objetivo es que nuestra IA gaste menos energía que la energía gastada por el sistema de enfriamiento no inteligente en el servidor. Y dado que, según el supuesto 2 anterior, la energía gastada en el servidor (por cualquier sistema) es proporcional al cambio de temperatura dentro de una unidad de tiempo. Eso significa que la energía ahorrada por la IA en cada instante  
t (cada minuto) es, de hecho, la diferencia en los cambios absolutos de temperatura causados en el servidor entre el sistema de enfriamiento integrado del servidor no inteligente y la IA de  t y  t+1

<div align="center">Energia ahorrada por la IA entre t y t+1=|ΔT Sistema de Enfriamiento Integrado del Servidor|−|ΔT IA| =|ΔTno IA|−|ΔTIA|</div>
 
donde:

ΔTnoIA  es el cambio de temperatura que causaría el sistema de enfriamiento integrado del servidor sin la IA en el servidor durante la iteración t, es decir, del instante t al instante t+1

ΔTAI    es el cambio de temperatura causado por la IA en el servidor durante la iteración t, es decir, del instante t al instante t+1

Nuestro objetivo será ahorrar la energía máxima cada minuto, por lo tanto, ahorrar la energía total máxima durante 1 año completo de simulación y, finalmente, ahorrar los costos máximos en la factura de electricidad de refrigeración / calefacción.

### Definición de los estados

El estado de entrada st en el momento t se compone de los siguientes tres elementos:

- La temperatura del servidor en el instante t.
- El número de usuarios en el servidor en el instante t.
- La velocidad de transmisión de datos en el servidor en el instante t.

Por lo tanto, el estado de entrada será un vector de entrada de estos tres elementos. Nuestra IA tomará este vector como entrada y devolverá la acción para ejecutar en cada instante t.

### Definición de las acciones

Las acciones son simplemente los cambios de temperatura que la IA puede causar dentro del servidor, para calentarlo o enfriarlo. Para que nuestras acciones sean discretas, consideraremos 5 posibles cambios de temperatura de  −3∘C a  +3∘C, para que terminemos con las 5 acciones posibles que la IA puede llevar a cabo para regular la temperatura del servidor:
Accion|¿Que hace?
------|---------------------------------
0     | La IA enfría el servidor 3∘C
1     | La IA enfría el servidor 1.5 ∘C
2     | La IA no transfiere calor ni frio al servidor (sin cambio de temperatura
3     | La IA calienta el servidor 1.5 ∘C
4     | La IA caliente el servidor 3 ∘C


### Definición de las recompensas

La recompensa en la iteración t es la energía gastada en el servidor que la IA está ahorrando con respecto al sistema de enfriamiento integrado del servidor, es decir, la diferencia entre la energía que gastaría el sistema de enfriamiento no inteligente si la IA fuera desactivada y la energía que la IA gasta en el servidor:

<div align="center">Rewardt = Et no IA − Et IA</div>

Y como (Supuesto 2), la energía gastada es igual al cambio de temperatura causado en el servidor (por cualquier sistema, incluido el AI o el sistema de enfriamiento no inteligente):

<div align="center">Reward t =|ΔT no IA | −|ΔTIA|</div>

donde:

ΔT no IA es el cambio de temperatura que causaría el sistema de enfriamiento integrado del servidor sin la IA en el servidor durante la iteración t, es decir, del instante tal instante t+1

ΔTAI es el cambio de temperatura causado por la IA en el servidor durante la iteración t, es decir, del instante tal instante t+1

**Nota importante:** es importante comprender que los sistemas (nuestra IA y el sistema de enfriamiento del servidor) se evaluarán por separado para calcular las recompensas. Y dado que cada vez que sus acciones conducen a temperaturas diferentes, tendremos que realizar un seguimiento por separado de las dos temperaturas  TIA y T no IA.

### Implementación

Esta implementación se dividirá en 5 partes, cada parte con su propio archivo de Python. 

- Construcción del entorno.
- Construcción del cerebro.
- Implementación del algoritmo de aprendizaje por refuerzo profundo (en nuestro caso será el modelo DQN).
- Entrenar a la IA.
- Probar de la IA.

#### Paso 1: Construción del Entorno "enviroment.py"

En este primer paso, vamos a construir el entorno dentro de una clase. ¿Por qué una clase? Porque nos gustaría tener nuestro entorno como un objeto que podamos crear fácilmente con cualquier valor de algunos parámetros que elijamos. Por ejemplo, podemos crear un objeto de entorno para un servidor que tenga un cierto número de usuarios conectados y una cierta velocidad de datos en un momento específico, y otro objeto de entorno para otro servidor que tenga un número diferente de usuarios conectados y un número diferente tasa de datos en otro momento. Y gracias a esta estructura avanzada de la clase, podemos conectar y reproducir fácilmente los objetos del entorno que creamos en diferentes servidores que tienen sus propios parámetros, por lo tanto, regulamos sus temperaturas con varias IA diferentes, de modo que terminamos minimizando el consumo de energía de un centro de datos completo.

- 1-1: Introducción e inicialización de todos los parámetros y variables del entorno.
- 1-2: Hacer un método que actualice el entorno justo después de que la IA ejecute una acción.
- 1-3: Hacer un método que restablezca el entorno.
- 1-4: hacer un método que nos proporcione en cualquier momento el estado actual, la última recompensa obtenida y si el juego ha terminado.

#### Paso 2: Contrucción del cerebro "brain.py"

En este Paso 2, vamos a construir el cerebro artificial de nuestra IA, que no es más que una red neuronal completamente conectada

![Brain](https://raw.githubusercontent.com/mcpade/MinimizacionCostes_IA/master/images/brain.png)

Nuevamente, construiremos este cerebro artificial dentro de una clase, por la misma razón que antes, que nos permite crear varios cerebros artificiales para diferentes servidores dentro de un centro de datos. De hecho, tal vez algunos servidores necesitarán cerebros artificiales diferentes con hiperparámetros diferentes que otros servidores. Es por eso que gracias a esta estructura avanzada de python de clase / objeto, podemos cambiar fácilmente de un cerebro a otro para regular la temperatura de un nuevo servidor que requiere una IA con diferentes parámetros de redes neuronales.

Construiremos este cerebro artificial gracias a la biblioteca **Keras**. Desde esta librería utilizaremos la clase Dense() para crear nuestras dos capas ocultas completamente conectadas, la primera con 64 neuronas ocultas y la segunda con 32 neuronas. Y luego, utilizaremos la clase Dense() nuevamente para devolver los valores Q, que tienen en cuenta las salidas de las redes neuronales artificiales. Luego, más adelante en el entrenamiento y los archivos de prueba, utilizaremos el método argmax para seleccionar la acción que tenga el valor Q máximo. Luego, ensamblamos todos los componentes del cerebro, incluidas las entradas y las salidas, creándolo como un objeto de la clase Model() (muy útil para luego guardar y cargar un modelo en producción con pesos específicos). Finalmente, lo compilaremos con una función de pérdidas que medirá el error cuadrático medio y el optimizador de Adam. 

- 2-1: Construir la capa de entrada compuesta de los estados de entrada.
- 2-2: Construir las capas ocultas con un número elegido de estas capas y neuronas dentro de cada una, completamente conectadas a la capa de entrada y entre ellas.
- 2-3: Construir la capa de salida, completamente conectada a la última capa oculta.
- 2-4: Ensamblar la arquitectura completa dentro de un modelo de Keras.
- 2-5: Compilación del modelo con una función de pérdida de error cuadrático medio y el optimizador elegido.

Se ha creado un segundo brain llamado **new_brain.py** donde se hace uso de la técnica **Dropout**. Es una técnica de regularización que evita el sobreajuste. Simplemente consiste en desactivar una cierta proporción de neuronas aleatorias durante cada paso de propagación hacia adelante y hacia atrás. De esa manera, no todas las neuronas aprenden de la misma manera, evitando así que la red neuronal sobreajuste los datos de entrenamiento.

#### Paso 3: Implementación del algoritmo de Deep Reinforcement Learning  "dqn.py"

En este nuevo archivo de python, seguimo el algoritmo Deep Q-Learning. Por lo tanto, esta implementación sigue los siguientes subpasos:

- 3-1: Introducción e inicialización de todos los parámetros y variables del modelo de DQN.
- 3-2: Hacer un método que construya la memoria en Repetición de Experiencia.
- 3-3: Hacer un método que construya y devuelva dos lotes de 10 entradas y 10 objetivos

#### Paso 4: Entrenar la IA  "training.py"

Ahora que nuestra IA tiene un cerebro completamente funcional, es hora de entrenarlo. Y esto es exactamente lo que hacemos en este cuarto archivo de python. Comenzamos estableciendo todos los parámetros, luego construimos el entorno creando un objeto de la clase Environment(), luego construimos el cerebro de la IA creando un objeto de la clase Brain(), luego construimos el modelo de Deep Q-Learning creando un objeto de la clase DQN(), y finalmente lanzamos la fase de entrenamiento que conecta todos estos objetos, durante 1000 epochs de 5 meses cada uno. 
En la fase de entrenamiento también exploramos un poco cuando llevamos a cabo las acciones las acciones. Esto consiste en ejecutar algunas acciones aleatorias de vez en cuando. En nuestro Caso Práctico, esto se realizará el 30% de las veces, ya que usamos un parámetro de exploración  ϵ=0.3, y luego lo forzamos a ejecutar una acción aleatoria al obtener un valor aleatorio entre 0 y 1 que está por debajo de  ϵ=0.3. La razón por la que hacemos un poco de exploración es porque mejora el proceso de aprendizaje por refuerzo profundo. Este truco se llama: Exploración vs. 

- 4-1: Construcción del entorno creando un objeto de la clase Environment.
- 4-2: Construyendo el cerebro artificial creando un objeto de la clase de Brain
- 4-3: Construyendo el modelo DQN creando un objeto de la clase DQN.
- 4-4: Elección del modo de entrenamiento.
- 4-5: Comenzar el entrenamiento con un bule for durante más de 100 epochs de períodos de 5 meses.
- 4-6: Durante cada epoch, repetimos todo el proceso de Deep Q-Learning, al tiempo que exploramos el 30% de las veces.

Después de ejecutar el código, ya vemos un buen rendimiento de nuestra IA durante el entrenamiento, gastando la mayor parte del tiempo menos energía que el sistema alternativo, es decir, el sistema de enfriamiento integrado del servidor. Pero ese es solo el entrenamiento, ahora necesitamos ver si también obtenemos un buen rendimiento en una nueva simulación de 1 año. Ahí es donde entra en juego nuestro próximo y último archivo de python.
El modelo obtenido se ha guardado en **model.h5**

![Brain](https://raw.githubusercontent.com/mcpade/MinimizacionCostes_IA/master/images/training.png)

#### Paso 5: Probar la IA  "testing.py"

Ahora tenemos que probar el rendimiento de nuestra IA en una situación completamente nueva. Para hacerlo, ejecutaremos una simulación de 1 año, solo en modo de inferencia, lo que significa que no habrá entrenamiento en ningún momento. Nuestra IA solo devolverá predicciones durante un año completo de simulación. Luego, gracias a nuestro objeto Environment, obtendremos al final la energía total gastada por la IA durante este año completo, así como la energía total gastada por el sistema de enfriamiento integrado del servidor. Eventualmente compararemos estas dos energías totales gastadas, simplemente calculando su diferencia relativa (en %), lo que nos dará exactamente la energía total ahorrada por la IA. 

En términos de nuestro algoritmo de IA, aquí para la implementación de prueba casi tenemos lo mismo que antes, excepto que esta vez, no tenemos que crear un objeto Brain ni un objeto modelo DQN, y por supuesto no debemos ejecutar el proceso de Deep Q-Learning durante las épocas de entrenamiento. Sin embargo, tenemos que crear un nuevo objeto de Environment, y en lugar de crear un cerebro, cargaremos nuestro cerebro artificial con sus pesos pre-entrenados del entrenamiento anterior que ejecutamos en el Paso 4 - Entrenamiento de la IA (model.h5). 

- 5-1: Construcción de un nuevo entorno creando un objeto de la clase Environment.
- 5-2: Carga del cerebro artificial con sus pesos pre-entrenados del entrenamiento anterior.
- 5-3: Elección del modo de inferencia.
- 5-4: Iniciación de la simulación de 1 año.
- 5-5: En cada iteración (cada minuto), nuestra IA solo ejecuta la acción que resulta de su predicción, y no se lleva a cabo ninguna exploración o entrenamiento de Deep Q-Learning.

#### Early Stopping

El entrenamiento de soluciones de Inteligencia Artificial puede ser muy costoso, especialmente si se entrenan para muchos servidores en varios centros de datos. Por lo tanto, debemos optimizar absolutamente el tiempo de entrenamiento de estas IA. Una solución para esto es la detención anticipada. Consiste en detener el entrenamiento si el rendimiento no mejora después de un cierto período de tiempo (por ejemplo, después de un cierto número de epochs). Esto plantea la siguiente pregunta: ¿Cómo evaluar la mejora del rendimiento? 

**training_earlystopping1.py**
Forma número 1: Comprobando si la recompensa total acumulada durante todo el período de 5 meses (= 1 epoch de entrenamiento) sigue aumentando, después de un número determinado de epochs, para nuestro ejemplo 10 épocas. En este caso el modelo generado es **modelearlyst.h5** y se debe modificar el fichero **testing.py** para cargar ese modelo





Forma número 2: Comprobando si la pérdida se sigue reduciendo, al menos en un porcentaje elegido, a lo largo de las epochs.


