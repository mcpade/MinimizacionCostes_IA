# MinimizacionCostes_IA

## Practical Case: Minimization of Costs in the Energy Consumption of a Data Center

### Problem to solve
 
Set up a server environment and build an AI that will control the cooling / heating of the server so that it stays in an optimal temperature range while saving maximum energy, thus minimizing costs.

An IA DQN (Deep Q-Learning) model is used and the goal will be to achieve at least 40% energy savings.

### Definition of the environment

Before defining statuses, actions, and rewards, let's explain how the server works. First, we will list all the environment parameters and variables by which the server is controlled.

#### Parameters

- the average atmospheric temperature during a month
- the optimal range of server temperatures, which will be (18∘C, 24∘C)
- the minimum temperature of the server below which it does not work, which will be 20∘C
- the maximum temperature of the server above which it does not work, which will be 80∘C
- the minimum number of users on the server, which will be 10
- the maximum number of users on the server, which will be 100
- the maximum number of users on the server that can go up or down per minute, which will be 5
- the minimum data transmission rate on the server, which will be 20
- the maximum data transmission speed on the server, which will be 300
- the maximum speed of data transmission that can go up or down per minute, which will be 10

#### Variables

- server temperature at any time
- the number of users on the server at any time
- the speed of data transmission at any minute
- the energy expended by the AI on the server (to cool or heat it) at any time
- the energy expended by the server's built-in cooling system that automatically brings the server temperature to the optimal range whenever the server temperature falls outside of this optimal range

All these parameters and variables will be part of our server environment and will influence the actions of the AI on the server.

Next, let's explain the two basic assumptions of the environment. It is important to understand that these assumptions are not related to artificial intelligence but are used to simplify the environment so that we can fully focus on the artificial intelligence solution.

#### Assumptions:

We will build on the following two essential assumptions:

Assumption 1: The server temperature can be approximated by Multiple Linear Regression, using a linear function of the atmospheric temperature, the number of users, and the data transmission rate:
Suppose that after performing this Multiple Linear Regression, we obtained the following values of the coefficients:

<div align="center">temp. del server = temp. atmosf. + 1.25 x n. de usuarios + 1.25 x ratio de transf. de datos</div>





Assumption 2: The energy expended by a system (our AI or the server's embedded cooling system) that changes the server temperature from Tt to Tt + 1 in 1 unit of time (here 1 minute), can be approximated again by regression Using a linear function of the absolute change in server temperature:

<div align="center">Et=|ΔTt|=|Tt+1−Tt|</div>

{Tt+1−Tt si Tt+1>Tt, i.e. if the server gets hot

 Tt−Tt+1 si Tt+1<Tt, that is, if the server gets cold}

#### Simulation

The number of users and the data transmission speed will fluctuate randomly to simulate a real server. This leads to randomness in temperature and the AI has to understand how much cooling or heating power it has to transfer to the server so as not to deteriorate the server's performance and at the same time spend the least energy optimizing its heat transfer.

### General operation

Within a data center, we are dealing with a specific server that is controlled by the parameters and variables listed above. Every minute some new users log into the server and some current users log out, thus updating the number of active users on the server. Likewise, every minute some new data is transmitted to the server, and some existing data is transmitted outside the server, therefore the data transmission rate that occurs within the server is updated. Therefore, based on assumption 1 above, the server temperature is updated every minute.
**Two possible systems can regulate the server temperature: the AI or the integrated server cooling system.**

The server's built-in cooling system is a non-intelligent system that will automatically return the server temperature to its optimal temperature - when the server temperature is updated every minute, it can stay within the optimal temperature range (18∘C, 24∘C ), or go out of this range. If it falls outside the optimal range, such as 30∘C, the server's built-in cooling system will automatically bring the temperature to the closest limit of the optimal range, which is 24∘C. However, the built-in cooling system of this server will only do so when AI is not activated.

If AI is enabled, then the server's built-in cooling system is disabled and the AI ​​updates the server temperature to better regulate it. But the AI ​​does that after some previous predictions, not in a deterministic way like with the non-smart server's built-in cooling system. Before there is an update to the number of users and the data transmission speed that causes the server temperature to change, the AI ​​predicts whether it should cool down the server, do nothing, or warm up the server. Then the temperature change occurs and the AI ​​reiterates. And since these two systems are complementary, we will evaluate them separately to compare their performance.

The goal is for our AI to use less energy than the energy wasted by the non-intelligent cooling system on the server. And since, according to assumption 2 above, the energy expended on the server (by any system) is proportional to the change in temperature within a unit of time. That means that the energy saved by the AI at every moment
t (every minute) is in fact the difference in absolute temperature changes caused in the server between the integrated cooling system of the non-intelligent server and the AI of t and t + 1

<div align="center">Energia ahorrada por la IA entre t y t+1=|ΔT Sistema de Enfriamiento Integrado del Servidor|−|ΔT IA| =|ΔTno IA|−|ΔTIA|</div>
 
where:

ΔTnoIA  is the change in temperature that the integrated server cooling system would cause without the AI in the server during iteration t, that is, from time t to time t + 1

ΔTAI    is the temperature change caused by the AI in the server during iteration t, that is, from time t to time t + 1

Our goal will be to save the maximum energy every minute, thus save the maximum total energy for 1 full simulation year, and finally save the maximum costs on the cooling / heating electricity bill.

### Definition of states

The input state st at time t consists of the following three elements:

- The server temperature at time t.
- The number of users on the server at time t.
- The data transmission speed on the server at time t.

Therefore, the input state will be an input vector of these three elements. Our AI will take this vector as input and will return the action to execute at every instant t.

### Definition of actions

Actions are simply the temperature changes that AI can cause inside the server, to heat it up or cool it down. To keep our actions discrete, we will consider 5 possible changes in temperature from −3∘C to + 3∘C, so that we end up with the 5 possible actions that the AI can take to regulate the server temperature:

Action| What does?
------|---------------------------------
0     | AI cools server 3∘C
1     | AI cools server 1.5 ∘C
2     | AI does not transfer heat or cold to the server (no temperature change)
3     | AI heats up the server 1.5 ∘C
4     | AI heat server 3 ∘C


### Definition of rewards

The reward in iteration t is the energy expended on the server that the AI is saving relative to the server's integrated cooling system, that is, the difference between the energy that the non-intelligent cooling system would use if the AI were deactivated and the energy the AI spends on the server:

<div align="center">Rewardt = Et no IA − Et IA</div>

And since (Assumption 2), the energy expended equals the temperature change caused in the server (by any system, including AI or non-smart cooling system):

<div align="center">Reward t =|ΔT no IA | −|ΔTIA|</div>

where:

ΔT no IA is the change in temperature that the integrated server cooling system would cause without the AI in the server during iteration t, that is, from time to time t + 1

ΔTAI is the change in temperature caused by the AI in the server during the iteration t, that is, from the instant such instant t + 1

**Important note:** It is important to understand that the systems (our AI and the server cooling system) will be evaluated separately to calculate the rewards. And since each time your actions lead to different temperatures, we will have to separately track the two temperatures TIA and T not IA.

### Implementation

This implementation will be divided into 5 parts, each part with its own Python file.

- Construction of the environment.
- Construction of the brain.
- Implementation of the deep reinforcement learning algorithm (in our case it will be the DQN model).
- Train the AI.
- Test the AI.

#### Step 1: Building the Environment "enviroment.py"

In this first step, we are going to build the environment inside a class. Why a class? Because we would like to have our environment as an object that we can easily create with any value of some parameters that we choose. For example, we can create an environment object for a server that has a certain number of connected users and a certain data rate at a specific time, and another environment object for another server that has a different number of connected users and a number different data rate at another time. And thanks to this advanced class structure, we can easily connect and reproduce the environment objects that we create on different servers that have their own parameters, therefore we regulate their temperatures with several different AIs, so that we end up minimizing the consumption of power of an entire data center.

- 1-1: Introduction and initialization of all parameters and environment variables.
- 1-2: Make a method that updates the environment right after the AI ​​executes an action.
- 1-3: Make a method that restores the environment.
- 1-4: make a method that provides us at any time the current status, the last reward obtained and if the game is over.

#### Step 2: Building the brain "brain.py"

In this Step 2, we are going to build the artificial brain of our AI, which is nothing more than a fully connected neural network

![Brain](https://raw.githubusercontent.com/mcpade/MinimizacionCostes_IA/master/images/brain.png)

Again, we will build this artificial brain within a class, for the same reason as before, which allows us to create multiple artificial brains for different servers within a data center. In fact, maybe some servers will need different artificial brains with different hyperparameters than other servers. That is why thanks to this advanced class / object python structure, we can easily switch from one brain to another to regulate the temperature of a new server that requires an AI with different neural network parameters.

We will build this artificial brain thanks to the **Keras** library. From this library we will use the Dense () class to create our two completely connected hidden layers, the first with 64 hidden neurons and the second with 32 neurons. And then we'll use the Dense () class again to return the Q values, which take into account the outputs of artificial neural networks. Then later in the training and test files we will use the argmax method to select the action that has the maximum Q value. Then, we assemble all the components of the brain, including the inputs and outputs, creating it as an object of the Model () class (very useful for later saving and loading a model in production with specific weights). Finally, we will compile it with a loss function that will measure the root mean square error and Adam's optimizer.

- 2-1: Build the input layer composed of the input states.
- 2-2: Build the hidden layers with a chosen number of these layers and neurons within each one, fully connected to the input layer and between them.
- 2-3: Build the output layer, completely connected to the last hidden layer.
- 2-4: Assemble the complete architecture within a Keras model.
- 2-5: Compilation of the model with a mean square error loss function and the chosen optimizer.

A second brain called **new_brain.py** has been created where the **Dropout** technique is used. It is a regularization technique that avoids overfitting. It simply consists of deactivating a certain proportion of random neurons during each step of forward and backward propagation. In this way, not all neurons learn in the same way, thus preventing the neural network from overfitting the training data.

#### Step 3: Implementing the Deep Reinforcement Learning algorithm  "dqn.py"

In this new python file, I follow the Deep Q-Learning algorithm. Therefore, this implementation follows the following substeps:

- 3-1: Introduction and initialization of all the parameters and variables of the DQN model.
- 3-2: Make a method that builds memory in Repetition of Experience.
- 3-3: Make a method that builds and returns two batches of 10 inputs and 10 goals

#### Step 4: Train the AI  "training.py"

ATime that our AI has a fully functional brain, it's time to train it. And this is exactly what we do in this fourth python file. We start by setting all the parameters, then we build the environment by creating an Environment () class object, then we build the AI ​​brain by creating an object of the Brain () class, then we build the Deep Q-Learning model by creating an object of the DQN () class, and finally we launch the training phase that connects all these objects, for 1000 epochs of 5 months each.
In the training phase we also explore a bit when we carry out the actions the actions. This consists of executing some random actions from time to time. In our Case Study, this will be done 30% of the time, since we use a scan parameter ϵ = 0.3, and then we force it to execute a random action by obtaining a random value between 0 and 1 that is below ϵ = 0.3. The reason we do a little exploring is because it improves the deep reinforcement learning process. This trick is called: Exploration vs.

- 4-1: Construction of the environment by creating an object of the Environment class.
- 4-2: Building the artificial brain by creating an object of Brain's class
- 4-3: Building the DQN model by creating an object of the DQN class.
- 4-4: Choice of training mode.
- 4-5: Begin training with a bule for over 100 epochs of 5-month periods.
- 4-6: During each epoch, we repeat the entire Deep Q-Learning process, while exploring 30% of the time.

After running the code, we already see a good performance of our AI during training, spending most of the time less energy than the alternative system, that is, the integrated cooling system of the server. But that's just the training, now we need to see if we also get a good performance in a new 1 year simulation. That's where our next and final python file comes in.
The obtained model has been saved in **model.h5**

![Train](https://raw.githubusercontent.com/mcpade/MinimizacionCostes_IA/master/images/training.png)

#### Step 5: Test the AI  "testing.py"

Now we have to test the performance of our AI in a completely new situation. To do this, we will run a 1-year simulation, in inference mode only, which means there will be no training at any time. Our AI will only return predictions for a full year of simulation. Then, thanks to our Environment object, we will finally get the total energy expended by the AI ​​during this entire year, as well as the total energy expended by the server's integrated cooling system. We will eventually compare these two total energies expended, simply by calculating their relative difference (in%), which will give us exactly the total energy saved by the AI.

In terms of our AI algorithm, here for the test implementation we have pretty much the same as before, except this time, we don't have to create a Brain object or a DQN model object, and of course we shouldn't run the Deep Q process. -Learning during training periods. However, we have to create a new Environment object, and instead of creating a brain, we will load our artificial brain with its pre-trained weights from the previous training that we ran in Step 4 - AI Training (model.h5).

- 5-1: Construction of a new environment by creating an object of the Environment class.
- 5-2: Loading the artificial brain with its pre-trained weights from the previous training.
- 5-3: Choice of inference mode.
- 5-4: Initiation of the 1-year simulation.
- 5-5: In each iteration (every minute), our AI only executes the action that results from its prediction, and no exploration or Deep Q-Learning training is carried out.

![Resultado](https://raw.githubusercontent.com/mcpade/MinimizacionCostes_IA/master/images/ResultadoTest.png)

It is seen that an energy saving of **49%** is achieved

#### Early Stopping


533/5000
Training Artificial Intelligence solutions can be very expensive, especially if training for many servers in multiple data centers. Therefore, we must absolutely optimize the training time of these AIs. One solution for this is early stopping. It consists of stopping training if performance does not improve after a certain period of time (for example, after a certain number of epochs). This raises the question: How to evaluate performance improvement?


**training_earlystopping1.py**

Way number 1: Checking if the total reward accumulated during the whole period of 5 months (= 1 epoch of training) continues to increase, after a certain number of epochs, (for our example 10 epochs). In this case, the generated model is **modelearlyst.h5** and the **testing.py** file must be modified to load that model.

![EarlyS1](https://raw.githubusercontent.com/mcpade/MinimizacionCostes_IA/master/images/trainingearlys1.png)

It can be seen that the stop was made in epoch 30. Therefore we would not need to train until 100 epochs

This would be the result in test:

![TestEarlyS1](https://raw.githubusercontent.com/mcpade/MinimizacionCostes_IA/master/images/ResultadoTestEarlys1.png)

You can see that the result is even better than the previous one and we achieved a saving of **57%**



**training_earlystopping2.py**

Way number 2: Checking if the loss continues to decrease, at least by a chosen percentage, throughout the epochs (in my example 5%). In this case, the generated model is **modelearlyst2.h5** and the **testing.py** file must be modified to load that model.

![EarlyS1](https://raw.githubusercontent.com/mcpade/MinimizacionCostes_IA/master/images/trainingearlys2.png)

It can be seen that the stop was made at epoch 31. Therefore we would not need to train until 100 epochs

This would be the result in test:

![TestEarlyS2](https://raw.githubusercontent.com/mcpade/MinimizacionCostes_IA/master/images/ResultadoTestEarlys2.png)

You can see that the result is similar to the one obtained in the previous formula, a saving of **55%**

#### Dropout

**training_earlystopping1_dropout.py**

I'm going to do a test with a second brain called **new_brain.py** where the ** Dropout ** technique is used. I will also use the first form of early stop. The model I generate is called **modelearlyst_dropout.h5** and the file **testing.py** must be modified to load that model.
Leaving 10 times for the early stop, this occurs at time 23 and a 25% saving is achieved. To achieve a better result, I have increased the number of waiting times for the early stop to 20, so I have the stop at time 33

This would be the result in test:

![TestEarlyS1Dropout](https://raw.githubusercontent.com/mcpade/MinimizacionCostes_IA/master/images/ResultadoTestEarlys1_Dropout.png)

I get a **79%** which is a very good energy saving result.




