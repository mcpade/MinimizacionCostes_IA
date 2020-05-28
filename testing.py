
#Fase de testing

import os
import numpy as np
import random as rn

#Para cargar un modelo en formato h5
from keras.models import load_model

import enviroment



#Configurar las semillas para reproducibilidad
#Para reproducir experimento.
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(12345)

#CONFIGURACIÓN DE LOS PARÁMETROS


#Numero de acciones de nuestro problema
number_actions = 5
#Valor central de las acciones, donde la IA no hace nada
direction_boundary = (number_actions -1)/2

#Diferencia de tempertaura entre dos indices consecutivos
# -3, 1.5, 0, 1.5, 3  la diferencia es de 1.5 grados
temperature_step = 1.5

#CONSTRUCCIÓN DEL ENTORNO CREANDO UN OBJETO DE LA CLASE ENVIRONMENT
env = enviroment.Environtment(optimal_temperature = (18.0, 24.0), initial_month = 0, initial_number_users = 20, initial_rate_data = 30)

#CARGA DE UN MODELO PREENTRENADO
#model = load_model("model.h5")
#model = load_model("modelearlyst.h5")
model = load_model("modelearlyst.h5")

#ELECCIÓN DEL MODO DE ENTRENAMIENTO
train = False


# EJECUCION DE UN AÑO DE SIMULACION EN MODO INFERENCIA

env.train = train

#Obtengo el estado actual que será la entrada de la red neuronal
current_state, _,_=env.observe()

#bucle for cada minuto durante un año
for timestep in range (0,12*30*24*60):
    
    #Hago la preccición de los valores q y me quedo con la acción que maximice 
    q_values = model.predict(current_state) #El modelo devuelve 5 filas y en la primera columna están las predicciones
                #Me quedo con la posición del máximo
    action = np.argmax(q_values[0])
              
                
    #Calculo la energía gastada (igual a diferencia de temp según modelo)
    if (action < direction_boundary):
        #estamos enfriando
        direction = -1
    else:
        #estamos calentando
        direction = 1
            
            
    #energia gastada = incremento de tempertaura
    energy_ai = abs(action - direction_boundary) * temperature_step #en direction boundary tenemos la acción que no varía la temperatura
    
    #Actualizar el entorno y alcanzar el suguiente estado
    next_state, reward, game_over = env.update_env(direction, energy_ai, int(timestep/(30*24*60)))        
            
    current_state = next_state


            
#IMPRIMIR LOS RESULTADOS DEL ENTRENAMIENTO AL FINAL DEL EPOCH
        
print("\n")
print(" - Energia total gastada por el sistema con IA: {:.0f} J.".format(env.total_energy_ai))
print(" - Energia total gastada por el sistema sin IA: {:.0f} J.".format(env.total_energy_noai))

print("ENERGIA AHORRADA: {:.0f} % ".format(100*(env.total_energy_noai-env.total_energy_ai)/env.total_energy_noai))
 
        
        




