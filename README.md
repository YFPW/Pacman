# Pacman
# Implementation
Team name: Pretty Rooster  
Fan YANG - 931031   
Xinhao ZHANG - 951247    
Yinhao ZHU - 887402  
Qitong YANG - 889222     
Video link: https://youtu.be/74mcu3HC3MQ
  
## Introduction
In this project, we implement the Pacman agents by using three approaches, Weighted $A^{*}$, Monte-Carlo Tree Search and Deep Q-learning. The implementations will be introduced and there is a comparison at the last section of this wiki.

## Implementation
Heuristic Search: myTeam.py

Monte Carlo Tree Search: MCTS.py

Deep Q-learning: DQNAgentTeam.py

## Heuristic Search Algorithm

Heuristic function is used in our implementation. It is used to estimate successors from a known stage to a given goal. Due to the complex situations of the game, goals are different in various states, for example when the agent is chased by enemy, the goal is to go to a safe point, otherwise the goal is to eat more food. We have two agents which take different strageties.

For the first agent, the action selection of it is based on whether it is chased by the ghost. If it is chased, it would choose to go to a nearest safe point (a position on border).Otherwise, this agent will keep eating food until there are only 2 foods left. Then, it will go back. The goals of the agent will be switched between each other. The capsule is also considered in the heuristic function. The agent will tend to eat more food after the ghost turns to white and it will check being chased situation when the ghost is about to return the same color before.

For the other agent, this agent is slightly different from the first one. During the travel, if it encounters any enemy, it will chase the enemy. Otherwise, its behaviors are the same as the first agent.

Both agents will consider the reverse actions. If they occur frequently, the goals will changed to be random subset of origin goals, so the agents may change a route to achieve the goals.

### Action Selection 
We have implemented different heuristic functions such as Minimum Spanning Tree and Minimum Distance to Goal State. We find that using the minimum distance to the goal state has a better performance and it is also faster. Using this heuristic function means that in our project, we always choose the action that achieves a state nearest the given goal.

### Drawbacks and Potential Improvements
There are drawbacks of this approach. First, the pacman will encounter the suituation that after it runs to a safe point, it will try to go back to get foods. Unfortunately, it encounters the enemy again. In this situation it might be stuck. Second, the policy in this apprach is unchangeable, there are suituations that the agent is not able to choose suitable actions according to our policies.

To solve the first situation, our strategy will chosose a random goal, then the agent will find another route to achieve that goal. However, this is a compromised solution.To improvement this, the enemy should be considered when the heuristic value is calculated. For the sencond situation, more detailed policies should be designed, there may be some tricks to achieve that.

## Monte Carlo Tree Search
### Implementation
Monte Carlo Tree Search (MCTS) is another algorithm implemented in this project. Usually, MCTS consists of four process: Selection, Expansion, Simulation, Backpropagation.
#### Selection

MCTS starts at a game state that is considered as the tree root. The selection is based on the Upper Confidence Bounds (UCB1) strategy, which is shown below.
```math
V_i + C\times \sqrt{\frac{ln N}{n_i}}
```

In each iteration, MCTS selects the next node using the UCB1 strategy until the maximum tree depth or time limit achieved, which are 20 steps and 0.8s respectively in our project. The maximum tree depth is utilized to restrict the exploitation of the search tree because short term performance of the Pac-Man is valued highly in this case. In addition, due to the limited time interval of each Pac-Man movement, the computation of MCTS is restricted by the maximum calculation time.

#### Expand
After selecting a node in the search tree, it is expanded to achieve next possible actions for the current state. A specific action is chosen by UCB1 strategy to perform simulation process.

#### Simulation
The selected action is applied to get the next game state, which is then used to simulate a possible playout of the game. Actions of both Pac-Man and opponents are simulated simultaneously during this process. Approximate reward shaping strategy is implemented to mimic the behaviours of Pac-man and Ghost, since it has less computational burden for simulation. For Pac-man, the rewards are designed to eat closest foods and avoid being chased by ghosts at the same time. By contrast, the rewards for ghost are devised to chase detected invaders while guarding foods within its region. Similar to the purpose of setting maximum tree depth in selection process, maximum simulation steps are utilized to restrict the simulation progress. The maximum simulation steps are set to 15 in out implementation.

#### Backpropagation
The rewards achieved in simulation process is then backpropagated from the expanded node to its ancestors iteratively. The value of each ancestor is updated

## Deep Q-learning
To implements Deep Q-learning, we abstract the running environment into matrixes, all features such as, wall, capsule, ghost, pacman are represented with different values in a matrix. Then, a convolutional neural network is built, this CNN is used to solve we do not have knowledge about $`max_{a^{'}}Q(s^{'}, a^{'})`$ when we chose an action in $`Q(s,a) = Q(s,a) + \alpha[r + \gamma max_{a^{'}}Q(s^{'}, a^{'})- Q(s, a)]`$. The model we use is below and some important parst of DQN will be introduce later.

![CNN](https://gitlab.eng.unimelb.edu.au/931031/comp90054-pacman-931031/raw/master/asset/%E5%B1%8F%E5%B9%95%E5%BF%AB%E7%85%A7_2019-10-13_%E4%B8%8B%E5%8D%881.24.49.png?inline=false)

``` python
#Neural net for Deep-Q-Learning
model = Sequential()
model.add(Conv2D(32, 5, 3, activation='relu',
			input_shape=(32, 16)))
model.add(Conv2D(64, 4, 2, activation='relu'))
model.add(Conv2D(64, 3, 1, activation='relu'))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(self.action_size))
model.compile(loss='mse', optimizer=RMSprop(lr=self.lr)) 
```
### Exploit and Exploration
We use $`\epsilon-greedy`$ to choose action. At the beginning, the $`\epsilon = 1.0`$, because the DQN does not have a better understanding of the running environment. With  $`\epsilon = 1.0`$, the action is selected randomly. When there are enough data to train DQN, everytime after training, there is a decay of  $`\epsilon`$,  $`\epsilon = \epsilon\times decay`$. This enables our agent to rely more and more on DQN when it chooses an action.

### Memory and Replay
Memory is a significant opinion in DQN. Memory is used to keep the history states and actions of pacman, and then we will use these history to training the DQN. The data structure we stored in memory is $`(state, action, rewar, next state)`$. The memory size in our project is 500000. We only store the most recent 500000 data points. Replay means using the history data points as training data to train DQN. 

### Action Selection
We use Keras to implement this CNN and train it in Colab. At every state, we use this CNN to predict the possibilities of $`a\in{North, South, East, West, Stop}`$ and then choose the one with highest value as the next action. 

### Drawbacks and Possible improvements
Although Deep Q-learning is a great approach to solve unknown $`max_{a^{'}}Q(s^{'}, a^{'})`$, its drawbacks are also obvious in our project. It needs large computational capability to train this DQN with a long peroid and large training data, because the Pacman game is complex. Without enough computational resources, it can not obtain enough knowledge. Reward policies in our implementation are also not enough for the suituations in the game.

To improve the performance of DQN in this game, we can obtain more computing resources and training DQN with more data, which is expensive. Moreover, a more detailed reward policy may be also helpful to achieve a better performance.

## Discussion

![data](https://gitlab.eng.unimelb.edu.au/931031/comp90054-pacman-931031/raw/master/asset/data.png)
The table above is shown the game results between our agents and staff team agents. Our heuristic search agent has the best record among all agents. It can definitely beat the baseline, basic and medium staff team agents. It has about 75% winning rate against the top agent and 50% winning rate against super agent. In comparison, the MCTS agent only has decent performance against the baseline agent (about 75% winning rate). It can barely beat other agents. That may because the simulation part of our MCTS may not apply the strategy of the game well, so the agents seem to be too “cautious”. The DQN agent has the worst performance. It cannot win any agent in the contest. With insufficient training data and time and also low computing power, the DQN cannot gain enough knowledge to play the game. For MCTS agent, it would have a better performance with more detailed reward policy and better simulation process. Overall, our heuristic search agent has the most comprehensive design which makes it work well during the game contest.

Nevertheless, the heuristic search agents cannot always defeat staff-team-super. In some situation, the agents are still be stuck by defenders although we have designed a mechanism that if the agents doing reversed actions frequently, the agents will change the goal of food randomly. This is because in these situations, the shortest paths to all of the food are all starting with the same direction. Thus, the agents will keep trying to do this action, find there is an enemy and then move backwards. This loop may continue until the enemy moves away or the game ends. In this way, our agents are defeated many times. 
Another loss situation of our heuristic search agents is that the agents carry a lot food, move to the dead end and finally are captured by defenders. This is because we have not fully analysed the map of the game. If our agents know the situation, they may probably move back to their side or eat other food, and then consider whether they should get the food according to the position of defenders.

To solve these problems, we decide to add a more complex analysis of map based on junctions, i.e. the agents consider moving to next junctions instead of adjacent position. This needs our agent analyse the map at the beginning of game. If agent at a position can perform more than two actions (except stop), we consider it as a junction. If the number of legal actions is only one, these points are recorded as dead end. The other points are considered to be an “edge”. In this structure, we may easily get if we move to next goal safely or have food in the dead-end edge safely. Moreover, we can record the junctions history to know if we are stuck so that we can move to other safe junctions. However, we just implemented some functions of that. If we had enough time, that would be our next stage of this project.
