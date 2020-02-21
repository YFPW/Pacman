'''
This is owned by pretty_rooster.
Eamil: yinghaoz2@student.unimelb.edu.au
Date: 01/10/2019
This file impletents a Deep-Q-Learning Agent
'''
import sys
sys.path.append('teams/Pretty Rooster/')

from captureAgents import CaptureAgent
from collections import deque
from keras import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.optimizers import RMSprop
from keras.models import load_model
from game import Actions, Directions
import random
import numpy as np
import time
import util
import os
from heuristicTeam import HeuristicOffensiveAgent2

ACTIONS =[Directions.NORTH, Directions.SOUTH, Directions.EAST,
         Directions.WEST, Directions.STOP]
ACTION_SIZE = len(ACTIONS)
LEARNING_RATE=0.01
REWARD_DECAY=0.9
E_GREEDY=1.0
REPLACE_TARGET_ITER=300  # UPDATE q TABLE
MEMORY_SIZE=500000  # HOW MANY ACTIONS IN MEMORY 1000GAMES
BATCH_SIZE=32
E_GREEDY_DECAY=0.999
NUMTRAINING=100
MODEL_PATH = 'DQNModel.hdf5'

def createTeam(firstIndex, secondIndex, isRed,
               first = 'DQNAgent1', second = 'DQNAgent2'):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """

    # The following line is an example only; feel free to change it.
    return [eval(first)(firstIndex), eval(second)(secondIndex)]

class DQNAgent(HeuristicOffensiveAgent2):
    def __init__( self, index, timeForComputing = .1 ):
        super().__init__(index)
        self.lr = LEARNING_RATE
        self.gamma = REWARD_DECAY
        self.epsilon = E_GREEDY
        #self.replace_target_iter = REPLACE_TARGET_ITER
        self.memory_size = int(MEMORY_SIZE)
        self.batch_size = BATCH_SIZE
        self.epsilon_decay = E_GREEDY_DECAY
        self.epsilon_min = 0.1
        self.episodesSoFar = 0
        self.accumTrainRewards = 0.0
        self.accumTestRewards = 0.0
        self.numTraining = int(NUMTRAINING)
        self.memory = deque(maxlen=self.memory_size) #the memory for replay
        if os.path.exists(MODEL_PATH):
            # print('Loading model...')
            self.model = load_model(MODEL_PATH)
        else:
            # print('New model')
            self.model  = self.build_model() #the neural network for deep learning

    def build_model(self, input_shape = (16, 32, 1)):
        #Neural net for Deep-Q-Learning
        model = Sequential()
        model.add(Conv2D(32, 5, 3, activation='relu',
                                input_shape=input_shape))
        model.add(Conv2D(64, 4, 2, activation='relu'))
        model.add(Conv2D(64, 3, 1, activation='relu'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(ACTION_SIZE))
        model.compile(loss='mse', optimizer=RMSprop(lr=self.lr))
        return model

    def registerInitialState(self, state):
        super().registerInitialState(state)
        self.pre_state = None
        self.action  = None
        self.episodeRewards = 0.0

    def observationFunction(self, gameState):
        " Changing this won't affect pacclient.py, but will affect capture.py "
        if not self.pre_state is None:
            reward = self.getReward(gameState)
            # print('reward: ', reward)
            self.observeTransition(self.pre_state, self.action, gameState, reward, False)
        return gameState.makeObservation(self.index)

    def getReward(self, gameState):
        currentPosition = gameState.getAgentPosition(self.index)
        foodList = self.getFood(gameState).asList()
        foodReward = -self.heuristicMST(currentPosition, foodList) if len(foodList) > 2 else -self.getMazeDistance(currentPosition, self.start)
        reward = foodReward
        return reward

    def heuristicMST(self, currentPosition, foodList):
        '''
        MST heuristic function
        '''
        h = 0
        seletedNodes = [currentPosition]
        candidateNodes = foodList.copy()

        while len(candidateNodes) > 0:
            endNode, minWeight = 0, 999999
            for i in seletedNodes:
                for j in candidateNodes:
                    #if util.manhattanDistance(i, j) < minWeight:
                    if self.getMazeDistance(i, j) < minWeight:
                        minWeight = util.manhattanDistance(i, j)
                        endNode = j
            h += minWeight
            seletedNodes.append(endNode)
            candidateNodes.remove(endNode)
        return h



    def observeTransition(self, state, action, nextState, deltaReward, done):
        self.episodeRewards += deltaReward
        self.update(state, action, nextState, deltaReward, done)

    def update(self, state, action, nextState, reward, done):
        '''
        from observer, the state, action, the reward of action adn next state are observed
        we use this to update our Q-value
        '''

        ###  change state to matrix #####
        self.nextState = self.makeObservation(nextState)

        self.memory_replay(self.currentState, self.action, reward, self.nextState, done)
        if len(self.memory) > 2 * self.batch_size:
            self.replay(self.batch_size)
    
    def memory_replay(self, state, action, reward, next_state, done):
        '''
        :param state:  s
        :param action: a
        :param reward: r
        :param next_state: s'
        :param done: Is terminal state or not
        :return: None
        '''
        action_index = ACTIONS.index(action)
        self.memory.append((state, action_index, reward, next_state, done))

    def replay(self, batch_size):
        # get training data from memory for training
        x_batch, y_batch = [], []
        minibatch = random.sample(self.memory, batch_size)
        #这里的action_index应该是个数字
        for state, action_index, reward, next_state, done in minibatch:
            y_target = self.model.predict(state)
            y_target[0][action_index] = reward if done else reward + self.gamma * np.max(
                self.model.predict(next_state)[0])
            x_batch.append(state[0])
            y_batch.append(y_target[0])
        # increasing epsilon
        self.model.fit(np.array(x_batch), np.array(y_batch), batch_size=len(x_batch), verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def chooseAction(self, gameState):
        '''
        Choose Action according to e-greedy
        '''
        legalactions = gameState.getLegalActions(self.index)

        self.currentState = self.makeObservation(gameState)

        if np.random.random() < self.epsilon:
            '''
            if len(legalactions) > 1 and Directions.STOP in legalactions:
                legalactions.remove(Directions.STOP)
            '''
            # action = random.choice(legalactions)
            action = super().chooseAction(gameState)

        else:
            #print('This action is from DQN')
            actions = self.model.predict(self.currentState)
            action = ACTIONS[np.argmax(actions[0])]
            if action not in legalactions:
                action = Directions.STOP
        self.doAction(gameState, action)
        return action

    def doAction(self, state, action):
        self.pre_state = state
        self.action = action

    def makeObservation(self, gameState):
        DIFFERENCE = 32
        MY_FOOD_SIGN = 10
        FOOD_SIGN = MY_FOOD_SIGN + DIFFERENCE
        MY_CAPSULE_SIGN = 20
        CAPSULE_SIGN = MY_CAPSULE_SIGN + DIFFERENCE
        GHOST_ME = 71
        GHOST_TEAMMATE = GHOST_ME + DIFFERENCE 
        GHOST_ENEMY = GHOST_TEAMMATE + DIFFERENCE
        PACMAN_ME = 80
        PACMAN_TEAMATE = PACMAN_ME + DIFFERENCE
        PACMAN_ENEMY = PACMAN_TEAMATE + DIFFERENCE
        SCARED_ME = 87
        SCARED_TEAMATE = SCARED_ME + DIFFERENCE
        SCARED_DEFENDER = SCARED_TEAMATE + DIFFERENCE
        gameStateStr = str(gameState.deepCopy())
        gameStateStr = gameStateStr[:gameStateStr.rfind('\n', 0, len(gameStateStr) - 1)]
        gameStateStr = gameStateStr.replace("\n", "")
        gameStateStr = gameStateStr.replace(' ', chr(0))
        gameStateList = list(gameStateStr)
        gameStateList = [ord(c) for c in gameStateList]
        gameStateList = np.array(gameStateList, dtype=np.float32)
        gameStateList = gameStateList.reshape(gameState.data.layout.height, gameState.data.layout.width)

        myFoodList = self.getFoodYouAreDefending(gameState).asList()
        for x, y in myFoodList:
            gameStateList[gameState.data.layout.height - 1 - y][x] = MY_FOOD_SIGN

        foodList = self.getFood(gameState).asList()
        for x, y in foodList:
            gameStateList[gameState.data.layout.height - 1 - y][x] = FOOD_SIGN

        myCapsuleList = self.getCapsulesYouAreDefending(gameState)
        for x, y in myCapsuleList:
            gameStateList[gameState.data.layout.height - 1 - y][x] = MY_CAPSULE_SIGN

        capsuleList = self.getCapsules(gameState)
        for x, y in capsuleList:
            gameStateList[gameState.data.layout.height - 1 - y][x] = CAPSULE_SIGN

        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        for a in enemies:
            if not a.isPacman and a.getPosition() != None:
                if a.scaredTimer > 0:
                    x, y = util.nearestPoint(a.getPosition())
                    gameStateList[gameState.data.layout.height - 1 - y][x] = SCARED_DEFENDER
                else:
                    x, y = util.nearestPoint(a.getPosition())
                    gameStateList[gameState.data.layout.height - 1 - y][x] = GHOST_ENEMY
            elif a.isPacman and a.getPosition() != None:
                x, y = util.nearestPoint(a.getPosition())
                gameStateList[gameState.data.layout.height - 1 - y][x] = PACMAN_ENEMY

        teams = [gameState.getAgentState(i) for i in self.getTeam(gameState) if i != self.index]
        teammate = teams[0]
        me = gameState.getAgentState(self.index)

        if not me.isPacman and me.getPosition() != None:
            if me.scaredTimer > 0:
                x, y = util.nearestPoint(me.getPosition())
                gameStateList[gameState.data.layout.height - 1 - y][x] = SCARED_ME
            else:
                x, y = util.nearestPoint(me.getPosition())
                gameStateList[gameState.data.layout.height - 1 - y][x] = GHOST_ME
        elif me.isPacman and me.getPosition() != None:
            x, y = util.nearestPoint(me.getPosition())
            gameStateList[gameState.data.layout.height - 1 - y][x] = PACMAN_ME

        if not teammate.isPacman and teammate.getPosition() != None:
            if teammate.scaredTimer > 0:
                x, y = util.nearestPoint(teammate.getPosition())
                gameStateList[gameState.data.layout.height - 1 - y][x] = SCARED_TEAMATE
            else:
                x, y = util.nearestPoint(teammate.getPosition())
                gameStateList[gameState.data.layout.height - 1 - y][x] = GHOST_TEAMMATE
        elif teammate.isPacman and teammate.getPosition() != None:
            x, y = util.nearestPoint(teammate.getPosition())
            gameStateList[gameState.data.layout.height - 1 - y][x] = PACMAN_TEAMATE

        gameStateList = np.reshape(gameStateList, (1, ) + gameStateList.shape + (1, ))

        return gameStateList

    def getSuccessor(self, gameState, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != util.nearestPoint(pos):
          return successor.generateSuccessor(self.index, action)
        else:
          return successor

    # this is copy from others, hahahah so fool!
    def final(self, state):
        """
          Called by Pacman game at the terminal state
        """

        deltaReward = state.getScore() - self.pre_state.getScore()
        self.observeTransition(self.pre_state, self.action, state, deltaReward, True)
        self.stopEpisode()

        # Make sure we have this var
        if not 'episodeStartTime' in self.__dict__:
            self.episodeStartTime = time.time()
        if not 'lastWindowAccumRewards' in self.__dict__:
            self.lastWindowAccumRewards = 0.0
        self.lastWindowAccumRewards += state.getScore()

        NUM_EPS_UPDATE = 10000
        if self.episodesSoFar % NUM_EPS_UPDATE == 0:
            print('episode: ', self.episodesSoFar)
            print('Saving model...')
            self.model.save(MODEL_PATH)
            print('Reinforcement Learning Status:')
            windowAvg = self.lastWindowAccumRewards / float(NUM_EPS_UPDATE)
            if self.episodesSoFar <= self.numTraining:
                trainAvg = self.accumTrainRewards / float(self.episodesSoFar)
                print('\tCompleted %d out of %d training episodes' % (self.episodesSoFar, self.numTraining))
                print('\tAverage Rewards over all training: %.2f' % (trainAvg))

            else:
                testAvg = float(self.accumTestRewards) / (self.episodesSoFar - self.numTraining)
                print('\tCompleted %d test episodes' % (self.episodesSoFar - self.numTraining))

                print('\tAverage Rewards over testing: %.2f' % testAvg)

            print('\tAverage Rewards for last %d episodes: %.2f' % (NUM_EPS_UPDATE, windowAvg))

            print('\tEpisode took %.2f seconds' % (time.time() - self.episodeStartTime))

            self.lastWindowAccumRewards = 0.0
            self.episodeStartTime = time.time()

        if self.episodesSoFar == self.numTraining:
            msg = 'Training Done (turning off epsilon and alpha)'
            print('%s\n%s' % (msg, '-' * len(msg)))

    def stopEpisode(self):
        """
          Called by environment when episode is done
        """
        if self.episodesSoFar < self.numTraining:
            self.accumTrainRewards += self.episodeRewards
        else:
            self.accumTestRewards += self.episodeRewards
        self.episodesSoFar += 1
        if self.episodesSoFar >= self.numTraining:
            # Take off the training wheels
            self.epsilon = 1.0  # no exploration
            self.lr = 0.0  # no learning

class DQNAgent1(DQNAgent):
    def registerInitialState(self, gameState):
        super().registerInitialState(gameState)
        self.minDistanceFoodStart = 0

class DQNAgent2(DQNAgent):
    def registerInitialState(self, gameState):
        super().registerInitialState(gameState)
        self.minDistanceFoodStart = 1

