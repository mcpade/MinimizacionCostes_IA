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

### Variables

- la temperatura del servidor en cualquier momento
- la cantidad de usuarios en el servidor en cualquier momento
- la velocidad de transmisión de datos en cualquier minuto
- la energía gastada por la IA en el servidor (para enfriarlo o calentarlo) en cualquier momento
- la energía gastada por el sistema de enfriamiento integrado del servidor que automáticamente lleva la temperatura del servidor al rango óptimo cada vez que la temperatura del servidor sale de este rango óptimo

Todos estos parámetros y variables serán parte de nuestro entorno de servidor e influirán en las acciones de la IA en el servidor.

A continuación, expliquemos los dos supuestos básicos del entorno. Es importante comprender que estos supuestos no están relacionados con la inteligencia artificial, sino que se utilizan para simplificar el entorno para que podamos centrarnos al máximo en la solución de inteligencia artificial.

### Suposiciones:

Nos basaremos en los siguientes dos supuestos esenciales:

Supuesto 1: la temperatura del servidor se puede aproximar mediante Regresión lineal múltiple, mediante una función lineal de la temperatura atmosférica, el número de usuarios y la velocidad de transmisión de datos:
Supongamos que después de realizar esta Regresión lineal múltiple, obtuvimos los siguientes valores de los coeficientes:  

temp. del server = temp. atmosf. + 1.25 x n. de usuarios + 1.25 x ratio de transf. de datos

Supuesto 2: la energía gastada por un sistema (nuestra IA o el sistema de enfriamiento integrado del servidor) que cambia la temperatura del servidor de  Tt a  Tt+1 en 1 unidad de tiempo (aquí 1 minuto), se puede aproximar nuevamente mediante regresión mediante una función lineal del cambio absoluto de temperatura del servidor:




