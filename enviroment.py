
#Creación del entorno


#Importa libreria

import numpy as np


#CONSTRUIR EL ENTORNO EN UNA CLASE

class Environtment (object):
    
    # INTRODUCIR E INICIALIZAR LOS PARÁMETROS Y VARIBLES DEL ENTORNO
    
    #Constructor, debe estar escrito así para que se interprete como constructor
    #Primer método que se llama al instanciar una clase
    #self hace referencia al propio objeto
    #parámetros de inicio, de creación de entorno:
        #temperatura optima (18º-24º)
        #mes de inicio
        #número inicial de usuarios conectados: 10 chatbots por ejemplo
        #transmisión de datos al inicio: 60 (por ejemplo)
    def __init__(self, optimal_temperature=(18.0,24.0), initial_month=0, 
                 initial_number_users=10, initial_rate_data=60):
        
        #Temperatura atmosférica promedio en cada mes
        self.monthly_atmospheric_temperature = [1.0, 5.0, 7.0, 10.0, 11.0, 20.0, 23.0, 24.0, 22.0, 10.0, 5.0, 1.0]
    
        #mes inicial, nos quedamos con el valor recibido por parámetro
        self.initial_month = initial_month
        
        #Temperatura inicial
        self.atmospheric_temperature = self.monthly_atmospheric_temperature[initial_month]
    
        
        #Rango de temperatura ótpima, cogemos las que nos viene por parámetro
        self.optimal_temperature = optimal_temperature
        
        #Temperatura mínima
        self.min_temperature = -20
        
        #Temperatura máxima
        self.max_temperature = 80
        
        #Mínimo número de usuarios
        self.min_number_users=10
        
        #Máximo número de usuarios
        self.max_number_users=100
        
        #Incremento o decremento máximo de usuarios por minuto
        self.max_update_users=5
        
        #Minimo intercambio de datos por unidad de tiempo
        self.min_rate_data = 20
        
        #Maximo intercambio de datos
        self.max_rate_data=300
         
        #Incremento o decremento máximo
        self.max_update_data = 10
        
        #Número inicial de usuarios, me quedo con el valor del parámetro
        self.initial_number_users = initial_number_users
        
        #Número actual de usuarios
        self.current_number_users = initial_number_users
        
        #Inicial rate date, me quedo con el valor recibido por parámetro
        self.initial_rate_data = initial_rate_data
        
        #Actual rate date
        self.current_rate_data = initial_rate_data
        
        
        #Temperatura del servidor
        #Modelo para la temperatura (obtenidio de un machine learning previo)
        #Temperatura = Tempertura atmosferica + 1.25*nº usuerios actuales + 1.25 rate data
        self.intrinsec_temperature = self.atmospheric_temperature + 1.25*self.current_number_users+1.25*self.current_rate_data
        
        
        #Temperatura inicial del server con IA, partimos del valor inicial de temperatura de server
        self.temperature_ai = self.intrinsec_temperature
        
        #Temperatura del server sin IA, la media de la óptima. El sistema autónomo va a colocar el server
        #en la media del rango óptimo
        self.temperature_noai = (self.optimal_temperature[0]+self.optimal_temperature[1])/2.0
        
        #Energia inicial gastada por la IA: 0
        self.total_energy_ai = 0.0
        
        #Energia inicial gasta por el sistema autónomo sin IA: 0
        self.total_energy_noai = 0.0
        
        #Recompensa. Inicialmente es 0
        self.reward = 0.0
        
        #Game over: variable que se usa para saber si la simulación ha terminado
        #se pone a 1 y ha acabado y a 0 si no. Inicialmete será 0        self.game_over = 0
        self.game_over = 0
        
        
        
        #Variable que se usa para saber si estamos en training (1) o test (0)
        self.train = 1
        
        
    
    # CREAR UN MÉTODO QUE ACTUALICE EL ENTORNO JUSTO DESPUÉS DE QUE LA IA EJECTUTE UNA ACCIÓN
    # Parametros de entrada:
    # dirección de cambio de temperatura de la IA (1 si caliente el server, -1 si lo enfría)
    # energia gastada por la IA para subir o bajar la temperatura
    # mes en el que estamos
    
    def update_env(self, direction, energy_ai, month):

        
        #1.- OBTENCIÓN DE LA RECOMPENSA
        #Obtenemos recompensa si la energía que gasta por el sistema con IA es menor
        #que la energía gastada por el sistema sin IA
        
        #1.1.- Calcula la energía gastada por el sistema de refrigeración del server sin IA
        energy_noai=0   #si estamos en temperatura óptima 
        #Si la temperatura del server sin IA baja por debajo de la mínima óptima
        #tengo que gasta energía para subirla.
        #Et = Tt+1 - Tt  (modelo obtenido con un machine learning anterior)
        if(self.temperature_noai  < self.optimal_temperature[0]):
            energy_noai = self.optimal_temperature[0] - self.temperature_noai

            #Actualizo a la nueva temperatura
            self.temperature_noai = self.optimal_temperature[0]  
        elif(self.temperature_noai > self.optimal_temperature[1]):
            energy_noai = self.temperature_noai - self.optimal_temperature[1]
            self.temperature_noai = self.optimal_temperature[1]
        
        
    
        #1.2.- Calcular la recompensa
        #Recompensa = E(no IA) - E(si IA) ahorro de energía
        
        self.reward = energy_noai - energy_ai
        
        
        #Escalar la recompensa --> se consiguen mejores resultados, converge antes
        #El rango va entre -20 y 80: 100 puntos. En este caso dividimos por 1000 para
        #que quede normalizada con un factor de 10 (elevado a -3)
        #Con este se consiguen mejores resultados de convergencia
        
        self.reward = 1e-3*self.reward
        
        
        
    
        #2.- OBTENCIÓN DEL SIGUIENTE ESTADO
        
        #2.1.- Actualizar la temperatura atmosférica (en el mes indicado por el método)
        self.atmospheric_temperature = self.monthly_atmospheric_temperature[month]
        
        #2.2.- Actualizar el número de usuarios. Al ser una simulación tomo un valor aleatorio
        #5 es el número máximo de usuarios que se pueden conectar o desconectar
        #cogemos un número aletorio entre -5 y 5
        self.current_number_users += np.random.randint(-self.max_update_users, self.max_update_users)
        
        #Miro que no me haya salido de los márgenes
        if(self.current_number_users < self.min_number_users):
            self.current_number_users = self.min_number_users
        elif(self.current_number_users > self.max_number_users):
            self.current_number_users = self.max_number_users
            
            
        #2.3.- Actualizar la tasa de transferencia de datos
        
        self.current_rate_data += np.random.randint(-self.max_update_data, self.max_update_data)
        if(self.current_rate_data < self.min_rate_data):
            self.current_rate_data = self.min_rate_data
        elif(self.current_rate_data > self.max_rate_data):
            self.current_rate_data = self.max_rate_data
        
        #2.4.- Calcular la variación de temperatura intrínseca
        #Me quedo con el valor anterior
        past_intrinsic_temperature =  self.intrinsec_temperature 
        
        #Actualizo
        self.intrinsec_temperature = self.atmospheric_temperature + 1.25*self.current_number_users+1.25*self.current_rate_data
        
        #Variación
        delta_intrinsec_temperaure = self.intrinsec_temperature - past_intrinsic_temperature
        
        
        #2.5.-Calcular la variación de temperatura causada por la IA
        #Me baso en el modelo que tengo para la energía
        if(direction==-1):
            delta_temperature_ai = -energy_ai
        elif(direction == 1):
            delta_temperature_ai = energy_ai
            
            
        #2.6.-Calcular la nueva temperatura del server cuando hay IA conectada
        
        self.temperature_ai += delta_intrinsec_temperaure + delta_temperature_ai
        
            
            
        #2.7.- Calcular la nueva temperatura del server cuando no hay IA conectada
        #la temperatura_noai ya la hemos calculado antes en el apartado de recompensa
        #así que aquí solo tenemos en cuenta la variación de la temperatura intrínsica
        
        self.temperature_noai += delta_intrinsec_temperaure

        
        #3.- OBTENCION DEL GAME OVER
        #Para poder parar la simulación y pasar al siguiente
        #temperatura de trabajo [-20,80]
        #Si la IA se va de este margen vamos a pararlo
        
        #Si la temperatura_ai ha bajado de la mínima permitida compruebo
        #si estoy en training o en test. Si estoy en train corto la ejecución
        #si estoy en test no puedo dejar que el sevidor se congele
        if(self.temperature_ai < self.min_temperature):
            if (self.train==1):
                #Corto esa ejecución si estoy en train
                self.game_over=1
            else: #si estoy en test subo la temperatura a la temperatura mínima óptima
                #gasto de energia por subir la tempertaura a la óptima
                self.total_energy_ai += self.optimal_temperature[0] - self.temperature_ai
                #subo la temperatura a la mínima óptima
                self.temperature_ai = self.optimal_temperature[0]
                
                           
        #Si la temperatura_ai ha subido por encima de la permitida compruebo
        #si estoy en training o en test. Si estoy en train corto la ejecución
        #si estoy en test no puedo dejar que el sevidor se queme
        if(self.temperature_ai > self.max_temperature):
             if (self.train==1):
                #Corto esa ejecución si estoy en train
                self.game_over=1
             else: #si estoy en test bajo la temperatura a la temperatura máxima óptima
                #gasto de energia por bajar la tempertaura a la óptima
                self.total_energy_ai += self.temperature_ai - self.optimal_temperature[1]
                #bajo la temperatura a la máxima óptima
                self.temperature_ai = self.optimal_temperature[1]   
                
                
                
        # 4.- ACTUALIZAR LOS SCORES
        
        #4.1.- Calcular la energia total gastada por la IA
        
        #energy_ai nos viene dada como parámetro
        self.total_energy_ai += energy_ai
        #4.2.- Calcular la energia total gatada por el sistema de refrigeración del server sin IA
        self.total_energy_noai += energy_noai
        
        
        # 5. ESCALAR EL SIGUIENTE PASO
        
        #Siempre hay que escarlar las variables antes de pasarlas a la rede neuronal 
        #para mejorar la convergencia
        #Debo suministrar la temperatura actual, nº usuarios, ratio para cada instante t de tiempo
        
        
        #Escalado de la temperatura
        #Transformación lineal:  (valor - valor mínimo) / diferencia entre max y min
        #Con esto se consigue que el valor esté entre 0 y 1
        
        scaled_temperature_ai = (self.temperature_ai - self.min_temperature)/(self.max_temperature - self.min_temperature)
        
        
        #Escalado del número de usuarios
        
        scaled_number_users = (self.current_number_users - self.min_number_users)/(self.max_number_users - self.min_number_users)
        
        
        #Escalado rate
        scaled_rate_data = (self.current_rate_data - self.min_rate_data)/(self.max_rate_data - self.min_rate_data)
        
        
        #Creamos un vector que combine estos 3 elementos y ese será el objeto de entrada a la red neuronal
        #Se crea una matriz, porque una lista de python no puede ser la entrada de una red neuronal
        #las librerías no está preparadas para recibir ese tipo de datos, si pueden recibir matrices
        
        next_state = np.matrix([scaled_temperature_ai, scaled_number_users, scaled_rate_data])
        
    
        # 6.- DEVOLVER EL SIGUIENTE ESTADO, RECOMPENSA Y GAME OVER
    
        return next_state, self.reward, self.game_over
    
    
    
    
    # CREAR UN MÉTODO QUE REINICIE EL ENTORNO
    
    def reset (self, new_month):
        
        #Reiniciamos datos que no son constantes
        self.atmospheric_temperature = self.monthly_atmospheric_temperature[new_month]
        self.initial_month = new_month
        self.current_number_users = self.initial_number_users
        self.current_rate_data = self.initial_rate_data
        self.intrinsec_temperature = self.atmospheric_temperature + 1.25*self.current_number_users+1.25*self.current_rate_data
        self.temperature_ai = self.intrinsec_temperature
        self.temperature_noai = (self.optimal_temperature[0]+self.optimal_temperature[1])/2.0
        self.total_energy_ai = 0.0
        self.total_energy_noai = 0.0
        self.reward = 0.0
        self.game_over = 0
        self.train = 1
        
        
    
    # CREAR UN MÉTODO QUE NOS DE CUALQUIER EN INSTANTE EL ESTADO ACTUAL, LA ÚLTIMA RECOMPENSA Y EL VALOR DE GAME OVER
    def observe (self):
        
        scaled_temperature_ai = (self.temperature_ai - self.min_temperature)/(self.max_temperature - self.min_temperature)
        scaled_number_users = (self.current_number_users - self.min_number_users)/(self.max_number_users - self.min_number_users)
        scaled_rate_data = (self.current_rate_data - self.min_rate_data)/(self.max_rate_data - self.min_rate_data)
        current_state = np.matrix([scaled_temperature_ai, scaled_number_users, scaled_rate_data])
        
        return current_state, self.reward, self.game_over
        
    
    
    
    
    