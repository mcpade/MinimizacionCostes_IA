#Creacion de la red Q profunda

#Deep Q - Learning

#Importar las librerias
import numpy as np

#IMPLEMENTAR EL ALGORITMO DE DEEP Q-LEARNING CON REPETICIÓN DE EXPERIENCIA

class DQN (object):
    
    #1. INTROCUCCIÓN E INICIALIZACIÓN DE LOS PARÁMETROS Y VARIABLES DEL DQN
    
    #parámetros: tamaño máximo de memoria (número de iteraciones)
    #discount_factor (gamma) es un hyperparámetro del modelo y se le da ese valor
   def __init__(self, max_memory = 100, discount_factor = 0.9):
        
        #inicializamos la memoria, como una lista vacia
        self.memory = list()
        
        #tamaño máximo de la memoria, viene por parámetro
        self.max_memory = max_memory
        
        #discount_factor
        self.discount_factor = discount_factor
    
    
    
    #2. CREACIÓN DE UN MÉTODO QUE CONSTRUYA LA MEMORIA DE REPETICIÓN DE LA EXPERIENCIA
    
   def remember(self, transition, game_over):
        #Añade elementos a la memoria revisando que no superemos el máximo y 
        #si lo superamos eliminamos el primero
        
        #Añado a la memoria la transición y el valor de game_over
        #de esa forma veo si la transición ha sido buena o no
        self.memory.append([transition, game_over])
        
        #Compruebo el tamaño
        if len(self.memory) > self.max_memory:
            #Si es superior que el tamaño máximo borro el primero
            #y me quedo en 100
            del self.memory[0]
    
    

    #3. CREACIÓN DE UN MÉTODO QUE CONSTRUYA DOS BLOQUES DE ENTRADAS
    # Y TARGETS EXTRAYENDO TRANSICIONES
    
    #el batch_size se selecciona para que tenga un valor 10 
    #como parámetro también recibimos el modelo
   def get_batch(self, model, batch_size = 10):
        #para garantizar que hay 10 elementos en la memoria
        len_memory = len(self.memory)
        
        #longitud de los datos de entrada = 3 (estado actual, acción a tomar, recompensa)
        #Para hacerlo genérico
        #las transiciones tienen esa longitud
        #Accedemos a un elemento de la memoria (el primero [0]), de aquí
        #me interesa la transición (transición, game_over) así que también es el elemento
        #0 y de aquí me interesa el primer elemento de la transición que es el estado actual
        #(estado actual, accion, recompensa, estado siguiente) y busco la dimensión (3 columnas)
        num_inputs = self.memory[0][0][0].shape[1]
        
        
        
        #número de acciones posibles = 5
        #Para hacerlo genérico
        #voy a acceder a la dimensión de capa de salida del modelo 
        #para ello uso el método output_shape y su último parámetro es
        #la dimensión que necesito
        num_outputs = model.output_shape[-1]
        
        #Bloque de entradas: creamos un vector de datos de entrada que contenga 10 observaciones
        #con 3 valores (matriz)
        
        #La inicializamos todo a ceros  (10,3)
        #Pero si el tamaño no llega a 10 porque estemos en las primeras interacciones
        #debe tener el tamaño que tenga en ese momento la memoria
        #Comparo batch_size y len_memory y me quedo con el menor
        inputs = np.zeros((min(batch_size, len_memory), num_inputs))   
        
        #Targets
        #El bloque de objetivo también necesita las mismas filas, 10 y las columnas
        #sería una por acción
        
        targets = np.zeros((min(batch_size, len_memory), num_outputs))
        
        #Extraigo 10 transiciones de la memoria de forma aleatoria (de las 100 posibles)
        #Tengop que tener en cuenta que quizás no tenga 100 sino menos y que quizás no tenga
        #10 transicione sino menos
        
        #el  bucle for toma 10 (batch_size) valores aleatorios de 0 a 100(longitud de la memoria)
        for i, idx in enumerate(np.random.randint(0, len_memory, size=min(len_memory, batch_size))):
            #En idx tengo la posición de memoria al el vector de números aleatorios
            #i: interacción por la que vamos
            
            #Extraigo las transicciones [0] (estado actual, acción actual, recompensa actual, estado siguiente )
            current_state, action, reward, next_state = self.memory[idx][0] 
            game_over = self.memory[idx][1]
            
            #Rellenamos las entradas del bloque de entradas, estas entradas será el 
            #current_state
            inputs[i] = current_state
            
            #Rellenamos el bloque de objetivos
            #Algoritmo: colocar en cada acción el valor Q para el estado y acción actual
            #Calculamos el valor Q, predicción que el algoritmo me devuelve para cada estado
            #Predicción a partir del current_state, me dará una predicción para cada una de las 5 valores acción
            #Estas 5 predicciones en fila son las que debe ir en la fila i del bloque de salida
            
            #Uso el modelo para predecir
            #El modelo nos devuelve en la primera posición las predicciones
            
            targets[i] = model.predict(current_state)[0]
            
            #Para el estado siguiente sumamos a la acción que maximiza el valor Q*gamma y sumamos la recompensa
            #Obtenemos los Q máximos
            
            Q_sa = np.max(model.predict(next_state)[0])

            
            if game_over:
                #Si estamos en game_over hemos llegado a un estado en el que es imposible llevar a cabo otra acción
                #No incrementamos la recompensa, le pongo simplemente el valor dado
                targets[i, action] = reward
                
            else:
                #recompensa + gamma*Q
                targets[i, action] = reward + self.discount_factor*Q_sa
                
        return inputs, targets
            
            
    
    
    
    
    
    
    
    
    
