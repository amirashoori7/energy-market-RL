## Overview
A researche study on the problem of Energy Market's Demand Side Management (DSM) using Reinforcement Learning techniques.
Local energy markets (LEMs) are targeted towards establishing a balance between the local generation and consumption which may facilitate a reduction in energy transmission, network congestion and expedite proper inclusion of decentralised RES (Mengelkamp et al. 2018a).

### Goals
Here we try to to optimize building energy consumption by understanding individual consumption behaviors based on the extensive data collected from the Advanced Metering Infrastructure (AMI).

This work is motivated by the hypothesis that an optimal resource allocation of end-user patterns based on daily smart electrical device profiles could be used to smoothly reconcile differences in future energy consumption patterns and the supply of variable sources such as wind and solar. 
It is expected that a cost minimization problem could be solved to activate real-time price responsive behavior

### Approach
Our objectives can be done by performing successive transformation of the historical data to learn powerful machine learning models to cope with the high uncertainty of the electrical patterns. Moreover, these models will be capable of generalization and they could be exploited in an on-line manner (i.e. few milliseconds) to minimize the cost or the energy consumption in newly encountered situations.

The building environment is modeled using a Markov Decision Process and it can be used to find the best long-term strategies. 

Prior studies showed that RL methods are able to solve stochastic optimal control problems in the power system area as well as an energy consumption scheduling problem with dynamic pricing. For instance, A batch reinforcement learning method was introduced to schedule a cluster of domestic electric water heaters, and further on applied for smart home energy management

In this research, we propose the use of the Deep Policy Gradient method, as part of Deep Reinforcement Learning algorithms, in the large-scale physical
context of smart grid - smart building, as following:
• We propose for an approach to optimize directly on-line the building energy consumption and the cost.
• We propose a new way to adapt DRL algorithms to the smart grid context, with the aim of conceiving a fast algorithm to learn the electrical patterns and to optimize on-line either the building energy consumption or the cost.
• We investigate two DRL algorithms, namely Deep Q-learning (DQN) and Deep Policy Gradient (DPG).
• DPG in its current state-of-the-art form is capable to take just one action at a specific time. As in the building
context multiple actions have to be taken at the same moment, we propose a novel gradient method to enhance DPG with the capability of handling multiple actions simultaneously.

We evaluate our proposed methods on the PecanStreet database at both the building and aggregated level. In the end, we prove that our proposed methods are able to efficiently cope with the inherent uncertainty and variability in the generation of energy. 

### Problem Formulation
In this context, we aim to reduce load peaks as well as to minimize the cost of energy. 

Let B denote the set of buildings, such that B i ∈ B, ∀i ∈ N representing the index of the building analyzed. 
The total building energy consumption E i is a sum over all power generation P + and consumption in a specific interval of time ∆t. 
Therein, based on the shifting capabilities of appliances present in a building we differentiate between flexible power P d − , 
e.g. electric devices d ∈ {1, .., m i }, and fixed consumption P − .

a) Cost minimization problem: In this paper, we assume two price components over the space of B, such that λ −t is the price value set by the utility company for the time-slot t and λ +t represents the price value at which the utility company buys energy from end-users at time-slot t. 
Therefore, the optimal cost associated with customer i at time t for an optimization time horizon T can be calculated as:

![alt text](https://github.com/amirashoori7/energy-market-RL/blob/8fca5c2360c83735a3c3e7944a323eb6f8ca00cb/fig/Optimal%20Cost.jpg)

where a i,d,t = 1 if the electrical device is on at that specific moment in time, and 0 otherwise. 
Please note that, in our proposed method, computing a i,d,t is equivalent with the estimation of the actions (see Fig.1).

b) Peak reduction problem: In the special case of constant price, for electricity generation and consumption, with λ+t = λ-t , the cost minimization problem becomes a peak reduction problem, defined as

![alt text](https://github.com/amirashoori7/energy-market-RL/blob/df27ee8752bdb8909e87d5a35fdf0ca149386486/fig/Cost%20Min.jpg)


### Background: 

A. Reinforcement Learning
In a Reinforcement Learning (RL) context, an agent learns to act using a (Partial Observable) Markov Decision Process (MDP) formalism. 
MDPs are defined by a 4-tuple

![alt text](https://github.com/amirashoori7/energy-market-RL/blob/8fca5c2360c83735a3c3e7944a323eb6f8ca00cb/fig/MDP.jpg)

Under structure of finite states and actions of the environment, the Markov decision problem is typically solved using dynamic programing. 
However, in our built environment, the model has a large (continuous) states space. Therein, the state space is given by the building energy consumption and price at every moment in time, while the action space is highly dependent on the electric device constrains. 
The success of every action a is measured by a reward r. Learning to act in an environment will make the agent to choose actions to maximize future rewards. 
The value function Q π (s, a) is an expected total reward in state s using action a under a policy π. 
Q-learning is one of the most popular reinforcement learning algorithms. 

B. Deep Neural Networks
......................................................
......................................................
......................................................

### PROPOSED METHOD
In this research we propose the use of Deep Reinforcement Learning (DRL) as an on-line method to perform optimal building resource allocation at different levels of aggregation.
The general architecture of our proposed method is depicted in the following Fig:

![alt text](https://github.com/amirashoori7/energy-market-RL/blob/b1f4c221e3fbe3467144cc3facabc0955970d5f1/fig/drl-arch.jpg)

DRL (RL combined with DNNs of k hidden layers) can learn to act better than the standard RL by automatically extracting patterns, such as those of electricity consumption.

Learning in DRL is done as follows: 
The DNN is trained with a variant of the Q-learning algorithm, using stochastic gradient descent to update its parameters. 
Firstly, the value-function from the standard RL algorithm is replaced by a deep Q-network with parameters θ, given by the weights and biases of DNN, such that Q(s, a, θ) ≈ Q π (s, a). 
This approximation is used further to define the objective function by mean-squared error in Q-values


### Implementation Milestones
To train the DQN and DPG models as a starting point an off-line database: 
1- we should build the environment game
2- for the assumed flexible loads, possibilities should be considered and evaluated
(We gave a positive reward if estimation was close to our optimization goal. If not, we assigned to that possibility a negative reward.) 
At the begin of the learning there are a lot of random choices, but in time (many iterations), the reinforcement learning model converges and will learn to choose just the possibilities which are close to the optimization goal. 
3- for DQN we must define possible actions on flexible devices, like switching off air conditioner, washing machines and so on
4- setting hyper-parameters of all experiments like learning rate α = 10 −2 , the discount factor to γ = 0.99, and η = 0.01
5- then we must train our models for x episodes where an episode is composed by y random chosen days. 
Also weights should be updated, for example after every 2 episodes
6- Reward function should be computed at the 
Finally, we obtain an alternative optimized version of the starting off-line database, which would be better if this strategy would have been used in reality.


The theory of RL is founded on two important principles: Bellman's equation and the theory of stochastic approximation.

Any RL model contains 4 basic elements:
1. System Environment (Simulation Model)
2. Learning Agent(s) (market participants)
3. Set of Actions for each Agent (Action Spaces)
4. System Response (participant Rewards)
