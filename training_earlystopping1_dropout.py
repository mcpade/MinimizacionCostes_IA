#Fase de entrenamiento

#Importar las librerías y otros ficheros de python

import os
import numpy as np
import random as rn

import enviroment
import new_brain
import dqn


#Configurar las semillas para reproducibilidad
#Para reproducir experimento.
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(12345)

#CONFIGURACIÓN DE LOS PARÁMETROS

#Exploración del entorno. El 30% de las veces la inteligencia artificial
#ignora sus predicciones y se dedica a explorar el entorno
epsilon = 0.3
#Numero de acciones de nuestro problema
number_actions = 5
#Valor central de las acciones, donde la IA no hace nada
direction_boundary = (number_actions -1)/2
number_epochs = 100
#Maximo tamaño de la memoria
max_memory = 3000
batch_size = 512

#Diferencia de tempertaura entre dos indices consecutivos
# -3, 1.5, 0, 1.5, 3  la diferencia es de 1.5 grados
temperature_step = 1.5

#CONSTRUCCIÓN DEL ENTORNO CREANDO UN OBJETO DE LA CLASE ENVIRONMENT
env = enviroment.Environtment(optimal_temperature = (18.0, 24.0), initial_month = 0, initial_number_users = 20, initial_rate_data = 30)

#CONSTRUCCIÓN DEL CEREBRO CREANDO UN OBJETO DE LA CLASE BRAIN
brain = new_brain.Brain(learning_rate = 0.00001, number_actions = number_actions)


#CONSTRUCCIÓN DEL MODELO CREANDO UN OBJETO DE LA CLASE DQN

dqn = dqn.DQN(max_memory = max_memory, discount_factor = 0.9)

#ELECCIÓN DEL MODO DE ENTRENAMIENTO
train = True


# ENTRENAR LA IA
env.train = train
#Para acceder mas rápido al modelo
model = brain.model

##### EARLYSTOP ######
#Defino una constante con un valor de épocas para el earlystop
#Si durante más de 5 épocas no se consigue mejorar la recompensa
#paro el entrenamiento

early_stopping = True
num_epoch_wait = 20
cont_epoch_wait = 0

#El mejor valor de recompensa comienza siendo el menor posible
best_total_reward = -np.inf

if (env.train):
    #INICIAR EL BUCLE DE TODAS LAS ÉPOCAS (1 Epoch = 5 Meses)
    #se recorren las épocas
    for epoch in range(1, number_epochs):
        #INICIALIZACIÓN DE LAS VARIABLES DEL ENTORNO Y DEL BUCLE DE ENTRENAMIENTO
        total_reward = 0
        loss = 0.
        #podemos empezar en diferentes meses cada vez
        new_month = np.random.randint(0, 12)
        #Reseteamos el entorno
        env.reset(new_month = new_month) 
        #En una nueva época ponemos game_over=False
        game_over = False
        
        #Obtengo el current_state (no necesito los tres valores que devuelve)
        current_state, _, _ = env.observe()
        
        #Vamos a iterar durante 5 meses en cada época
        timestep = 0
        #5 meses lo pasamos a minutos. Se ejecuta cada minuto
        #5*30*24*60
        #INICIALIZACIÓN DE BUCLE DE TIMESTEP (Timestep=1 minuto) EN UNA EPOCA
        while ((not game_over) and (timestep <= 5*30*24*60)):
            
            #EJECUTAR LA SIGUIENTE ACCIÓN POR EXPLORACIÓN (Epsion 30%, selecciono aleatoriamente una de las opciones)
            
            #Genero un número aleaotorio entre 0 y 1 y si estoy por debajo de epsilon hago exploración
            #y si estoy por encima hago inferencia
            if np.random.rand() <= epsilon: #Si no se pone nada se interpreta que es entre 0 y 1
                #Escojo una acción aleatoria    
                action = np.random.randint(0, number_actions)
                
                
            else:
            #EJECUTAR LA SIGUIENTE ACCIÓN POR INFERENCIA (acción con mayor Q)
                #El cerebro hará la predicción a partir del current_state, nos devolverá el Q para cada acción y me quedo con aquella acción donde Q sea el máximo
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
            
            
            
     
            #ACTUALIZAR EL ENTORNO Y ALCANZAR EL SIGUIENTE PASO
            
            #Para calcular el mes de los 5 en los que estoy en cada época
               
            #Para calcular el mes en el que estoy tengo que tener en cuenta que en timestamp
            #voy acumulando los minutos de 5 meses, así que divido timestpe por el número de minutos
            #que hay en un mes y quedándome con la parte entera tengo el mes por el que voy
            next_state, reward, game_over = env.update_env(direction, energy_ai, int(timestep/(30*24*60)))
            
            total_reward += reward
                   
            
            #ALMACENAR LA NUEVA TRANSICIÓN EN LA MEMORIA
            #Genero el objeto transition: estado actual, acción, recompensa, siguiente estado
            #Utilizo el método remember al que hay que pasarle la transición y el gameover
            dqn.remember([current_state, action, reward, next_state], game_over)
            
            
            #OBTENER LOS DOS BLOQUES SEPARADOS DE ENTRADAS Y OBJETIVOS
            
            inputs, targets = dqn.get_batch(model, batch_size)
            
            #CALCULAR LA FUNCIÓN DE PÉRDIDAS UTILIZANDO TODO EL BLOQUE DE ENTRADA Y OBJETIVO
            #Método entrenar en un bloque
            loss += model.train_on_batch(inputs, targets)
            
            #Este método devuelve la cantidad de error devuelta por este modelo
            #Guardando loss podemos hacer un tracking del algoritmo
            
            #Para la siguiente iteración
            timestep += 1
            current_state = next_state
            
        #IMPRIMIR LOS RESULTADOS DEL ENTRENAMIENTO AL FINAL DEL EPOCH
        
        print("\n")
        print("Epoch: {:03d}/{:03d}.".format(epoch, number_epochs))
        print(" - Energia total gastada por el sistema con IA: {:.0f} J.".format(env.total_energy_ai))
        print(" - Energia total gastada por el sistema sin IA: {:.0f} J.".format(env.total_energy_noai))
        print(" - Total reward: {:.2f} ".format(total_reward))
        print(" - Best Total reward: {:.2f} ".format(best_total_reward))
        print(" - Cont: {:.0f} ".format(cont_epoch_wait))
        
        
        #EARLY STOPPING
        
        #Compruebo la recompensa obtenida y la comparo con la mejor
        
        
        if early_stopping:
        
            if (total_reward <= best_total_reward):
            #En esta época no he conseguido mejorar la recompensa
                cont_epoch_wait +=1
                if cont_epoch_wait >= num_epoch_wait:
                    #Si he llegado al límite de épocas esperando que mejore
                    #la recompensa me salgo del entrenamiento
                    print ("Parada temprana")
                    break  
            
       
            else:
                #En esta época he mejorado 
                #Actualizo el valor de la mejor recompensa
                best_total_reward = total_reward
                #Pongo el contador a 0
                cont_epoch_wait = 0
            
         
            
            
        
        
        #GUARDAR EL MODELO PARA SU USO FUTURO
        #La extensión .h5 es para ser interpretado por Keras
        model.save ("modelearlyst_dropout.h5")




