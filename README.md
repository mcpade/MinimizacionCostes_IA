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
