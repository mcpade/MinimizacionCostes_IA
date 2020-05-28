#Creación del cerebro 

#Importar las librerías
from keras.layers import Input, Dense, Dropout
from keras.models import Model
from keras.optimizers import Adam

# CONSTRUCCIÓN DEL CEREBRO

class Brain (object):
    #menor learning_rate, tarda más en aprender pero evitamos overfiting
    #numero_actions: número de acciones finales, número de neuronas
    #de la última capa
   def __init__(self, learning_rate = 0.001, number_actions = 5):
        self.learning_rate = learning_rate

        
        #Capa de entrada, una columna con 3 filas:
        #Temperatura del Servidor, Número de Usuarios, Tasa de transmisión de Datos    
        #En el shape no le indico el número de columnas para dejar
        #la puerta abierta a hacer un entrenamiento por bloques
        states = Input(shape = (3,))
        
        #Segunda Capa: Capa densa de 64 neuronas, con función
        #de activación sigmoide
        #La capa Densa se aplica a la capa de entrada states
        x = Dense(units = 64, activation = "sigmoid")(states)
         
        
        #Tercera Capa: Capa densa de 32 neuronas aplicada sobre 
        #la anterior
        y = Dense(units = 32, activation = "sigmoid")(x)
       
        
        #Ultima capa: la de salida. Es también una capa Densa
        #Para la capa de salida utilizo como función de actividad
        #softmax (regresión)
        q_values = Dense(units = number_actions, activation = "softmax")(y)
        
        #Creamos el modelo indicando la capa de entrada y la salida
        #el modelo es la arquitectura completa
        #hacemos que sea visible utilizando self
        self.model = Model(inputs = states, output = q_values)
        
        
        #Pasamos ahora a compilar el modelo indicando una función de error(pérdidas)
        #y el optimizador
        
        #Es un problema de regresión(continuo), así que la fución de error será la del
        #error cuadrático medio
        
        self.model.compile(loss = "mse", optimizer = Adam(lr = learning_rate))      
        
        