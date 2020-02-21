# myTeam.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

import sys
sys.path.append('teams/Pretty Rooster/')

from captureAgents import CaptureAgent
import random, time, util
from game import Directions,Actions
import game
import sys
import math
import time
from game import Directions
import numpy as np
from heuristicTeam import HeuristicOffensiveAgent2

MAX_DEPTH = 10
MAX_VALUE = 999999
MAX_TIME = 0.8
MAX_SIMULATE_STEP = 40
EXPLORATION_CONSTANT = 10000


#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'DummyAgent', second = 'DummyAgent'):
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

##########
# Agents #
##########

class DummyAgent(HeuristicOffensiveAgent2):
    """
    A Dummy agent to serve as an example of the necessary agent structure.
    You should look at baselineTeam.py for more details about how to
    create an agent as this is the bare minimum.
    """

    def registerInitialState(self, gameState):

        self.start = gameState.getAgentPosition(self.index)
        self.maxFood = len(self.getFood(gameState).asList())
        super().registerInitialState(gameState)

    def chooseAction(self, gameState):
        """
        Picks among actions randomly.
        """
        print('------------------------------------------------------------')
        print('Current position: ', gameState.getAgentPosition(self.index))
        print(gameState.getAgentState(self.index).getDirection())
        root = Node(gameState)
        mct = MonteCarloTree(root, self)
        action = mct.MCTS()
        return action

    def chooseHeuristicAction(self, gameState):
        return super().chooseAction(gameState)
    
    def getFeatures(self, successor):
        features = util.Counter()
        foodList = self.getFood(successor).asList()
        features['remainingFood'] = self.maxFood - len(foodList)
        myPos = successor.getAgentState(self.index).getPosition()
        features['score'] = self.getScore(successor)
        if len(foodList) > 0:
            minDistance = 1 / (min([self.getMazeDistance(myPos, food) for food in foodList]) + 0.5)
            features['distanceToFood'] = minDistance
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        currentGhost = [a for a in enemies if not a.isPacman and a.getPosition() != None and self.getMazeDistance(myPos, a.getPosition()) < 5]
        if len(currentGhost) > 0:
            features['distanceToEnemy'] = 1 / min([self.getMazeDistance(myPos, enemy.getPosition()) for enemy in currentGhost])
        return features

    def getWeights(self, successor):
        weights = util.Counter()
        foodList = self.getFood(successor).asList()
        myPos = successor.getAgentState(self.index).getPosition()
        weights['remainingFood'] = 1000
        weights['score'] = 500
        weights['distanceToFood'] = 250
        weights['distanceToEnemy'] = -1000
        scaredTimer = max([successor.getAgentState(i).scaredTimer for i in self.getOpponents(successor)])
        if scaredTimer > 0 and scaredTimer < 35:
            weights['distanceToEnemy'] = 0
        return weights

    def getSimulateFeatures(self, successor):
        features = util.Counter()
        foodList = self.getFood(successor).asList()
        features['remainingFood'] = self.maxFood - len(foodList)
        myPos = successor.getAgentState(self.index).getPosition()
        if len(foodList) > 0:
            minDistance = 1 / (min([self.getMazeDistance(myPos, food) for food in foodList]) + 0.5)
            features['distanceToFood'] = minDistance

        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        currentGhost = [a for a in enemies if
                        not a.isPacman and a.getPosition() != None and self.getMazeDistance(myPos, a.getPosition()) < 5]
        if len(currentGhost) > 0:
            features['distanceToEnemy'] = min([self.getMazeDistance(myPos, enemy.getPosition()) for enemy in currentGhost])
        else:
            features['distanceToEnemy'] = 10
        features['notBeingChased'] = 1

        borderList = self.getSafePoints(successor, 2).asList()
        minBoard = min([self.getMazeDistance(myPos, border) for border in borderList])
        features['distanceToBorder'] = 1 / (minBoard + 0.5)

        features['score'] = self.getScore(successor)

        features['numCarrying'] = successor.getAgentState(self.index).numCarrying

        features['scaredTimer'] = max([successor.getAgentState(i).scaredTimer for i in self.getOpponents(successor)])

        capsules = self.getCapsules(successor)
        if len(capsules) != 0:
            features['distanceToCapsule'] = 1 / min([self.getMazeDistance(myPos, capsule) for capsule in capsules])

        if self.isDeath(successor):
            features['Death'] = 1

        return features


    def getSimulateWeights(self, successor):
        weights = util.Counter()
        foodList = self.getFood(successor).asList()
        myPos = successor.getAgentState(self.index).getPosition()
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        currentGhost = [a for a in enemies if
                        not a.isPacman and a.getPosition() != None and self.getMazeDistance(myPos, a.getPosition()) < 6]

        if len(foodList) > 0:
            minFood = 1 / (min([self.getMazeDistance(myPos, food) for food in foodList]))

        numCarrying = successor.getAgentState(self.index).numCarrying
        scaredTimer = max([successor.getAgentState(i).scaredTimer for i in self.getOpponents(successor)])

        weights['score'] = 100
        weights['remainingFood'] = 40
        weights['distanceToFood'] = 25
        weights['distanceToEnemy'] = 50
        weights['distanceToBorder'] = 5 * (numCarrying)
        weights['notBeingChased'] = 200
        if len(currentGhost) > 0:
            weights['remainingFood'] = 0
            minDistToEnemy = min([self.getMazeDistance(myPos, g.getPosition()) for g in currentGhost])
            weights['distanceToFood'] = 0.025 * minDistToEnemy
            weights['distanceToEnemy'] = 25 + 10 * numCarrying
            weights['distanceToCapsule'] = 1 / (minDistToEnemy + 0.5) * 25 + 80
            weights['distanceToBorder'] = 10 * (numCarrying + 1) + 1 / (minDistToEnemy + 0.5) * 150
            weights['notBeingChased'] = 0
        if scaredTimer > 0 and scaredTimer < 35:
            weights['distanceToEnemy'] = 0

        if len(foodList) <= 2:
            weights['distanceToBorder'] = 300

        if self.isDeath(successor):
            weights['Death'] = -99999

        return weights

    def getSafePoints(self, gameState, safePositionX = 0):
        ''' Get the grid of valid border points on own side '''
        borderGrid = game.Grid(gameState.data.layout.width, gameState.data.layout.height)
        borderLine = gameState.data.layout.width // 2
        if self.red:
            for i in range(gameState.data.layout.height):
                safeX = 0
                if borderLine - 1 - safePositionX > safeX:
                    safeX = borderLine - 1 - safePositionX
                if not gameState.getWalls()[safeX][i]:
                    borderGrid[safeX][i] = True
        else:
            for i in range(gameState.data.layout.height):
                safeX = gameState.data.layout.width - 1
                if borderLine + safePositionX < safeX:
                    safeX = borderLine + safePositionX
                if not gameState.getWalls()[safeX][i]:
                    borderGrid[safeX][i] = True
        return borderGrid

    def isDeath(self, successor):
        if successor.getAgentPosition(self.index) == self.start:
            return True
        return False


class Node:
    def __init__(self, gameState):
        self.state = gameState
        self.parent = None
        self.children = None
        self.visited = 0
        self.value = 0
        self.reward = 0

    def __hash__(self):
        gameStateData = self.state.data
        agentStates = gameStateData.agentStates
        agentStatesHashVals = []
        for agentState in agentStates:
            configuration = agentState.configuration
            configurationHashVal = 0
            if configuration is None:
                configurationHashVal = hash(None)
            else:
                configurationHashVal = hash(configuration.pos)
            agentStateHashVal = hash(configurationHashVal + 13 * hash(agentState.scaredTimer))
            agentStatesHashVals.append(agentStateHashVal)
        agentStatesHashVals = tuple(agentStatesHashVals)
        return int((hash(agentStatesHashVals)) + 13 * hash(gameStateData.food) \
                + 113* hash(tuple(gameStateData.capsules)) \
                + 7 * hash(gameStateData.score) % 1048575)

    def __eq__(self, other):
        if other == None: return False
        gameStateData = self.state.data
        otherGameStateData = other.state.data
        if gameStateData.food != otherGameStateData.food: return False
        if gameStateData.capsules != otherGameStateData.capsules: return False
        if gameStateData.score != otherGameStateData.score: return False

        agentStates = gameStateData.agentStates
        otherAgentStates = otherGameStateData.agentStates
        agentStates = []
        for agentState in agentStates:
            if agentState.configuration is None:
                agentStates.append((None, agentState.scaredTimer))
            else:
                agentStates.append((agentState.configuration.pos, agentState.scaredTimer))
        for i in range(len(agentStates)):
            otherAgentState = otherAgentStates[i]
            otherAgentStateInfo = (None, otherAgentState.scaredTimer)
            if otherAgentState.configuration is not None:
                otherAgentStateInfo = (otherAgentState.configuration.pos, otherAgentState.scaredTimer)
            if otherAgentStateInfo != agentStates[i]:
                return False
        return True

    def addChild(self, agentIndex, action, subNode):
        if self.children is None:
            self.children = {}
        subNode.parent = self
        self.children[(agentIndex, action)] = subNode

    def getPosition(self, agentIndex):
        return self.state.getAgentPosition(agentIndex)

    def getDirection(self, agent):
        return self.state.getAgentState(agent.index).getDirection()

class ChoicePoint(Node):
    def __init__(self, gameState):
        super().__init__(gameState)
        self.prob = {}

class MonteCarloTree:
    def __init__(self, root, agent, gamma = 0.95, epsilon = 0.3):
        self.root = root
        self.agent = agent
        self.nodeSet = set()
        self.nodeSet.add(root)
        self.gamma = gamma
        self.epsilon = epsilon

    def MCTS(self):
        startTime = time.time()
        currentTime = time.time()
        i = 0
        while currentTime - startTime < MAX_TIME: 
           
            print('---------------------start--------------------')
            print('node set: ', len(self.nodeSet))
            for n in list(self.nodeSet):
                print(n.getPosition(self.agent.index))

            ''' Select '''
            print('------------------- select -------------------')
            selectNode = self.select(self.root)
            if selectNode is None:
                currentTime = time.time()
                continue
            print('selectNode: ', selectNode.getPosition(self.agent.index))
            print(selectNode.state.getAgentState(self.agent.index).getDirection())

            ''' Expand '''
            print('------------------- expand -------------------')
            expandNode = self.expand(selectNode)
            print('expandNode: ', expandNode.getPosition(self.agent.index))
            print(expandNode.state.getAgentState(self.agent.index).getDirection())

            ''' Simulate '''
            print('------------------- simulate -------------------')
            reward = self.simulate(expandNode.state, 0)
            expandNode.visited += 1
            expandNode.value = reward
            expandNode.children = None

            ''' Back propagate '''
            print('------------------- back propogate -------------------')
            self.backPropagate(selectNode, reward)
            currentTime = time.time()
            i += 1
        # print(i)
        action = self.bestAction(self.root)
        print('best action: ', action) 
        return action

    def select(self, root):
        leafNode = self.findLeafNode(root)
        ''' No leaf node '''
        if leafNode is None:
            return None
        print('LeafNode: ', leafNode.getPosition(self.agent.index))
        ''' This node is choice point '''
        if isinstance(leafNode, ChoicePoint):
            return leafNode
        ''' This node is state node '''
        leafNode.children = {}
        actions = leafNode.state.getLegalActions(self.agent.index)
        if Directions.STOP in actions:
            actions.remove(Directions.STOP)
        for a in actions:
            successor = self.getSuccessor(leafNode.state, a, self.agent.index) 
            childNode = ChoicePoint(successor)
            if childNode in self.nodeSet:
                continue
            print('child ', childNode.getPosition(self.agent.index))
            leafNode.addChild(self.agent.index, a, childNode)
            self.nodeSet.add(childNode)

        if len(leafNode.children) == 0:
            self.increaseVisitedTillRoot(leafNode)
            return None
        ucbDict = {}
        for _, child in leafNode.children.items():
            ucbDict[child] = self.ucb(child)
        nextChild = max(ucbDict, key=lambda x:ucbDict[x])
        return nextChild

    def findLeafNode(self, node, depth = 0):
        if node.children is None:
            return node
        if len(node.children) == 0 or depth > MAX_DEPTH:
            self.increaseVisitedTillRoot(node)
            return None
        #print(node.getPosition())
        #print(node.state.getAgentState(node.agent.index).getDirection())
        ucbDict = {}
        for _, child in node.children.items():
            ucbDict[child] = self.ucb(child)
        #    print(child.getPosition())
        #    print(ucbDict[child])
        nextChild = max(ucbDict, key=lambda x:ucbDict[x])
        #print('Max: ', nextChild.getPosition())
        #print(ucbDict[nextChild])
        return self.findLeafNode(nextChild, depth + 1)

    def increaseVisitedTillRoot(self, node):
        currentNode = node
        while currentNode is not None:
            currentNode.visited += 1
            currentNode = currentNode.parent

    def ucb(self, node):
        c = EXPLORATION_CONSTANT
        if node.visited == 0:
            return MAX_VALUE
        score = node.value + c * math.sqrt(2 * math.log(node.parent.visited) / node.visited) 
        return score

    def expand(self, node):
        node.children = {}
        opponents = self.agent.getOpponents(node.state)
        for a in opponents:
            opponentState = node.state.getAgentState(a)
            if opponentState.getPosition() != None:
                enemiesPosition = util.nearestPoint(opponentState.getPosition()) 
                if opponentState.isPacman:
                    self.enemyOffence(node, a)

                else:
                    self.enemyDefend(node, a)

        ''' No opponent is found '''
        if len(node.children) == 0:
            childNode = Node(node.state)
            node.addChild(None, None, childNode)
            node.prob[(None, None)] = 1.0
        
        ucbDict = {}
        for _, child in node.children.items():
            ucbDict[child] = self.ucb(child)
        nextChild = max(ucbDict, key=lambda x:ucbDict[x])
        return nextChild

    def getSuccessor(self, gameState, action, index):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = gameState.generateSuccessor(index, action)
        pos = successor.getAgentState(index).getPosition()
        if pos != util.nearestPoint(pos):
            return successor.generateSuccessor(index, action)
        else:
            return successor

    def enemyOffence(self, node, index):
        actions = [a for a in node.state.getLegalActions(index)]
        for a in actions:
            successor = self.getSuccessor(node.state, a, index)
            childNode = Node(successor)
            if childNode not in self.nodeSet:
                node.addChild(index, a, childNode)
                node.prob[(index, a)] = 1.0 / len(actions)

    def enemyDefend(self, node, index):
        actions = [a for a in node.state.getLegalActions(index)]
        for a in actions:
            successor = self.getSuccessor(node.state, a, index)
            childNode = Node(successor)
            if childNode not in self.nodeSet:
                node.addChild(index, a, childNode)
                node.prob[(index, a)] = 1.0 / len(actions)

    def simulate(self, gameState, step):
        if step == MAX_SIMULATE_STEP or gameState.isOver():
            nodeReward = self.agent.getFeatures(gameState) * self.agent.getWeights(gameState)
            return nodeReward

        actions = gameState.getLegalActions(self.agent.index)
        if Directions.STOP in actions:
            actions.remove(Directions.STOP)
        ''' Simulate move based on e-greedy and reward '''
        nextState = None
        if np.random.random() < self.epsilon:

            actionReward = {}
            bestReward = -MAX_VALUE
            for a in actions:
                successor = self.getSuccessor(gameState, a, self.agent.index)
                features = self.agent.getFeatures(successor)
                weights = self.agent.getWeights(successor)
                reward = features * weights
                if reward > bestReward:
                    bestReward = reward
                actionReward[successor] = features * weights
            nextStates = [k for k, v in actionReward.items() if v == bestReward]
            nextState = random.choice(nextStates)

            # a = self.agent.chooseHeuristicAction(gameState)
            # nextState = self.getSuccessor(gameState, a, self.agent.index)
        else:
            a = random.choice(actions)
            nextState = self.getSuccessor(gameState, a, self.agent.index)

        opponents = self.agent.getOpponents(gameState)
        enemies = []
        nextEnemyState = None

        for a in opponents:
            opponentState = gameState.getAgentState(a)
            if opponentState.getPosition() != None:
                enemiesPosition = util.nearestPoint(opponentState.getPosition()) 
                enemies.append(opponentState)
                if opponentState.isPacman:
                    nextEnemyState = self.simulateEnemyOffence(nextState, a)

                else:
                    teammates = [nextState.getAgentState(i) for i in self.agent.getTeam(nextState)]
                    nextEnemyState = self.simulateEnemyDefence(nextState, a, teammates)

        if len(enemies) == 0:
            nextEnemyState = nextState

        if self.agent.isDeath(nextEnemyState):
            return -1000
        futureReward = self.simulate(nextEnemyState, step + 1)
        print('node reward: ', gameState.getAgentPosition(self.agent.index), ' ', self.gamma * futureReward )

        return self.gamma * futureReward 

    def simulateEnemyOffence(self, gameState, agentIndex):
        actions = gameState.getLegalActions(agentIndex)
        action = random.choice(actions)
        return self.getSuccessor(gameState, action, agentIndex)

    def simulateEnemyDefence(self, gameState, agentIndex, enemies):

        actions = gameState.getLegalActions(agentIndex)
        values = []
        for action in actions:

            features = util.Counter()
            successor = self.getSuccessor(gameState, action, agentIndex)

            myState = successor.getAgentState(agentIndex)
            myPos = myState.getPosition()

            # Computes whether we're on defense (1) or offense (0)
            features['onDefense'] = 1
            if myState.isPacman: features['onDefense'] = 0

            # Computes distance to invaders we can see
            invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]

            features['numInvaders'] = len(invaders)
            if len(invaders) > 0:
                dists = [self.agent.getMazeDistance(myPos, a.getPosition()) for a in invaders]
                features['invaderDistance'] = min(dists)

            if action == Directions.STOP: features['stop'] = 1
            rev = Directions.REVERSE[gameState.getAgentState(agentIndex).configuration.direction]
            if action == rev: features['reverse'] = 1

            weights = {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2}
            values.append(features * weights)

        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]

        return self.getSuccessor(gameState, random.choice(bestActions), agentIndex)

    def backPropagate(self, node, reward):
        currentNode = node
        while currentNode is not None:
            currentNode.visited += 1
            print(currentNode.getPosition(self.agent.index))
            if isinstance(currentNode, ChoicePoint):
                probCounter = util.Counter(currentNode.prob)
                valueCounter = util.Counter()
                for key, child in currentNode.children.items():
                    valueCounter[key] = child.value
                currentNode.value = probCounter * valueCounter
                print('value: ', currentNode.value)
            else:
                currentNode.value = max([a.value for a in currentNode.children.values()])      
                print('value: ', currentNode.value)
            currentNode = currentNode.parent

    def bestAction(self, node):
        ''' Find the best score in all child nodes and return the action to reach child state '''
        
        if len(node.children) == 0:
            actions = node.state.getLegalActions(self.agent.index)
            if len(actions) > 1 and Directions.STOP in actions:
                actions.remove(Directions.STOP)
            return random.choice(actions)

        bestScore = -MAX_VALUE
        #print(node.getPosition())
        #print("node value")
        #print(node.value)
        #print("child value")
        qsas = {}
        for key, child in node.children.items():
            if child.value > bestScore:
                bestScore = child.value
            qsas[key[1]] = child.value
        bestActions = [k for k, v in qsas.items() if v == bestScore]
        bestAction = random.choice(bestActions)

        return bestAction

