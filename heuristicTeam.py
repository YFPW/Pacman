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
from game import Directions
import game
from collections import defaultdict
from collections import Counter
import math
import random
import numpy as np

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'HeuristicOffensiveAgent', second = 'HeuristicOffensiveAgent2'):
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

def nullHeuristic(state):
    return 0


##########
# Agents #
##########

class DummyAgent(CaptureAgent):
    """
    A Dummy agent to serve as an example of the necessary agent structure.
    You should look at baselineTeam.py for more details about how to
    create an agent as this is the bare minimum.
    """

    def registerInitialState(self, gameState):
        """
        This method handles the initial setup of the
        agent to populate useful fields (such as what team
        we're on).

        A distanceCalculator instance caches the maze distances
        between each pair of positions, so your agents can use:
        self.distancer.getDistance(p1, p2)

        IMPORTANT: This method may run for at most 15 seconds.
        """

        '''
        Make sure you do not delete the following line. If you would like to
        use Manhattan distances instead of maze distances in order to save
        on initialization time, please take a look at
        CaptureAgent.registerInitialState in captureAgents.py.
        '''
        self.start = gameState.getAgentPosition(self.index)
        CaptureAgent.registerInitialState(self, gameState)

        '''
        Your initialization code goes here, if you need any.
        '''


    def chooseAction(self, gameState):
        """
        Picks among actions randomly.
        """
        actions = gameState.getLegalActions(self.index)

        '''
        You should change this in your own agent.
        '''

        return random.choice(actions)

class HeuristicAgent(CaptureAgent):
    '''
    Agent for heuristic search.
    Attributes:
        start:          The start position of agent in a game
        goal:           The goal position
        opponents:      The opponent list
    '''
    def registerInitialState(self, gameState):
        ''' This method handles the initial setup of the agent to populate useful fields '''
        self.start = gameState.getAgentPosition(self.index)
        self.goal = []
        self.opponents = set()
        super().registerInitialState(gameState)

    def waStarSearch(self, gameState, heuristic, W = 2):
        '''
        WA* algorithm

        Args
            gameState: gameState used in Agent
            heuristic: heuristic function
            W: weight in WA*

        Returns
            actions: Action list
        '''
        closedList = set()
        queue = util.PriorityQueue()
        ''' In the queue, there are start states, cost, action list '''
        queue.push((gameState, 0, []), W * heuristic(gameState))
        bestCost = {}
        i = 0

        while not queue.isEmpty():
            (currentState, currentCost, actions) = queue.pop()
            currentPosition = currentState.getAgentPosition(self.index)
            if currentPosition not in closedList or currentPosition not in bestCost or currentCost < bestCost[currentPosition]:
                closedList.add(currentPosition)
                bestCost[currentPosition] = currentCost
                if self.isGoalState(currentState):
                    #print(i)
                    return actions 
                legalActions = self.getLegalActions(currentState)
                i += 1
                for legalAction in legalActions:
                    state = self.getSuccessor(currentState, legalAction)
                    g = currentCost + 1
                    h = heuristic(state)
                    nextActions = actions.copy()
                    nextActions.append(legalAction)
                    queue.push((state, g, nextActions), g + W * h)
        return None

    def isGoalState(self, gameState):
        return gameState.getAgentPosition(self.index) in self.goal

    def getLegalActions(self, currentState):
        ''' get away from opponents '''
        legalActions = currentState.getLegalActions(self.index)
        currentPositionX, currentPositionY = currentState.getAgentPosition(self.index)
        for a in legalActions:
            dx, dy = game.Actions.directionToVector(a)
            newPosition = (currentPositionX + dx, currentPositionY + dy)
            if newPosition in self.opponents:
                legalActions.remove(a)
        return legalActions

    def getSuccessor(self, gameState, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != util.nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

    def heuristicMST(self, gameState):
        '''
        MST heuristic function
        '''
        h = 0
        seletedNodes = [gameState.getAgentPosition(self.index)]
        candidateNodes = self.goal.copy()

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

    def heuristicZero(self, gameState):
        return 0

    def heuristicMin(self, gameState):
        distanceToGoal = [self.getMazeDistance(gameState.getAgentPosition(self.index), eachGoal) for eachGoal in self.goal]
        return min(distanceToGoal)

    def heuristicMax(self, gameState):
        distanceToGoal = [self.getMazeDistance(gameState.getAgentPosition(self.index), eachGoal) for eachGoal in self.goal]
        return max(distanceToGoal)

    def isOnDeffenceSide(self, currentPosition, gameState):
        ''' Check if agent is on its deffensive side'''
        borderLine = gameState.data.layout.width // 2
        x, y = currentPosition
        if self.red:
            return x < borderLine
        else:
            return x >= borderLine

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

    def findCloestPositionAndDistance(self, positionList, currentPosition):
        '''
        Find the cloest position to agent and distance between them

        Args:
            positionList:               List of positions
            currentPosition:            Position to start

        Returns:
            cloestPosition:             Cloest position
            minDistance:                Minimum Distance between the position and agent

        '''
        minDistance = 999999
        cloestPosition = None
        if len(positionList) > 0:
            for pos in positionList:
                distanceToPosition = self.getMazeDistance(currentPosition, pos)
                if distanceToPosition < minDistance:
                    minDistance = distanceToPosition
                    cloestPosition = pos
        return cloestPosition, minDistance

    def findFurthestPositionAndDistance(self, positionList, currentPosition):
        '''
        Find the furthrest position to agent and distance between them

        Args:
            positionList:               List of position 
            currentPosition:            Position to start

        Returns:
            furthestPosition:           Furthest position
            furthestDistance:           Maximum Distance between the position and agent

        '''

        furthestPosition = None
        furthestDistance = 0
        if len(positionList) > 0:
            for pos in positionList:
                distanceToPosition = self.getMazeDistance(currentPosition, pos)
                if distanceToPosition > furthestDistance:
                    furthestPosition = pos
                    furthestDistance = distanceToPosition
        return furthestPosition, furthestDistance

class MyQueue(util.Queue):
    '''
    Queue that has capicity
    Attributes:
        capicity:           The maximum capicity
    '''
    def __init__(self, capicity = 0):
        super().__init__()
        self.capicity = capicity

    def push(self, item):
        ''' Push 'item' onto the stack and if it is full pop it '''
        super().push(item)
        if len(self.list) > self.capicity:
            self.pop()

    def remove(self, item):
        ''' Remove item from list '''
        self.list.remove(item)

    def copyList(self):
        return self.list.copy()

class HeuristicOffensiveAgent(HeuristicAgent):
    '''
    Offence agent in heuristic search
    
    Attributes:
        safeDistance:               Minimum distance to defenders is considered as safe state
        safeDistanceBehindBorder:   Minimum distance to border
        safePoints:                 Grid of positions that is safe for agent (Grid object)
        borderPoints:               Grid of positions that is on border
        beingChased:                A boolean variable that record if the agent is chased
        numCapsule:                 Number of capsules on map
        scaredTimer:                Timer after eating capsules
        safeTimer:                  The timer of returning escaping
        catchup:                    Defference between number of food I got and total score to decide coming back
        returnFood:                 The number of food I ate to return 
        maxQueueCapicity:           Maximum capicity of MyQueue 
        numHistoryObserve:          Number of history observed my positions
        observedDefender:           MyQueue object record defender position
        observeCurrentPosition:     My history position 
        repeatLimit:                Limit time of repeat
        returnFlag:                 if repeat too much, return
        minDistanceFoodStart:       Start searching the food first smallest food or second smallest
        myFood:                     The food I am resposible to eat (only initialized at start, or changed if it is empty)

    '''
    def registerInitialState(self, gameState):
        super().registerInitialState(gameState)
        self.safeDistance = 4
        self.safeDistanceBehindBorder = 1
        self.safePoints = self.getSafePoints(gameState, self.safeDistanceBehindBorder)
        self.borderPoints = self.getSafePoints(gameState)
        self.beingChased = False
        self.numCapsule = len(self.getCapsules(gameState))
        self.scaredTimer = 0
        self.safeTimer = 4
        self.catchup = 3
        self.returnFood = 10
        self.maxQueueCapicity = 4
        self.observedDefender = MyQueue(self.maxQueueCapicity)
        self.numHistoryObserve = 16
        self.observeCurrentPosition = MyQueue(self.numHistoryObserve)
        self.repeatLimit = 2
        self.returnFlag = False
        self.minDistanceFoodStart = 1
        self.myFood = []

    def chooseAction(self, gameState):
        '''
        Choose action according to the state whether the agent is chased.
        If it is chased, use WA* back to safe points
        Otherwise, eating food until getting 18 food
        Specially, if the capsule is eaten within safe time, eat food normally
        '''
        currentPosition = util.nearestPoint(gameState.getAgentPosition(self.index))
        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        currentObserveDefenders = [util.nearestPoint(a.getPosition()) for a in enemies if not a.isPacman and a.getPosition() != None]

        ''' Expected defenders position '''
        for opponentPosition in self.observedDefender.copyList():
            if opponentPosition not in currentObserveDefenders:
                self.observedDefender.remove(opponentPosition)
        for cod in currentObserveDefenders:
            self.observedDefender.push(cod)
        expectedDefenders = self.observedDefender.copyList()
        self.opponents = set(self.getDefenderBlockingPosition(gameState, expectedDefenders))

        ''' Observed defenders position and state within safe distance '''
        distanceToDefenders = [self.getMazeDistance(currentPosition, d) for d in currentObserveDefenders]
        cloestDefender, cloestDistanceToDefender = self.findCloestPositionAndDistance(currentObserveDefenders, currentPosition)
        if cloestDistanceToDefender > self.safeDistance:
            cloestDefender = None
        cloestDefenderState = None
        for a in enemies:
            if cloestDefender is not None and a.getPosition() == cloestDefender:
                cloestDefenderState = a
                break
        
        ''' Record current position in history '''
        self.observeCurrentPosition.push(currentPosition) 

        if self.beingChased and cloestDefender is None or \
                cloestDefenderState is not None and cloestDefenderState.scaredTimer > self.safeTimer:
            ''' In capsule power time '''
            self.beingChased = False
            self.opponents = set()

        elif (not self.beingChased and self.isReadyOffensing(gameState.getAgentPosition(self.index)) and 
                cloestDefender is not None and min(distanceToDefenders) < self.safeDistance):
            self.beingChased = True

        #print(self.opponents)

        ''' counting food I ate '''
        myScore = gameState.getAgentState(self.index).numCarrying 

        ''' Initialize returnFlag '''
        if currentPosition in self.safePoints.asList():
            self.returnFlag = False
        
        ''' Number of food left '''
        numOfFood = len(self.getFood(gameState).asList())

        ''' Adjust my food '''
        if currentPosition == self.start:
            distanceToFoods = self.getFoodSortedByDistance(currentPosition, gameState)
            self.myFood = [foods[0] for i, foods in enumerate(distanceToFoods) if i >= self.minDistanceFoodStart]
        self.myFood = [food for food in self.myFood if food in self.getFood(gameState).asList()]
        if len(self.myFood) == 0:
            self.myFood = self.getFood(gameState).asList()

        ''' Action '''
        if numOfFood < 3:
            ''' Number of food is smaller than 3 '''
            self.goal = self.borderPoints.asList()
        elif self.beingChased or self.returnFlag:
            ''' Chased by defender '''
            self.goal = self.safePoints.asList()
            self.goal.extend(self.getCapsules(gameState))
        elif self.getScore(gameState) < 0 and myScore + self.getScore(gameState) > self.catchup:
                #or myScore >= self.returnFood:
            ''' Come back if I ate a lot or I can reverse score '''
            self.goal = self.borderPoints.asList()
        else:
            positionCounter = Counter(self.observeCurrentPosition.list)
            numRepeat = positionCounter[currentPosition] - self.repeatLimit

            if numRepeat <= 0:
                self.goal = self.myFood

            elif numRepeat >= self.numHistoryObserve // 2 - self.repeatLimit or numRepeat >= numOfFood - 1:
                ''' Repeat too much and return to safe points '''
                self.goal = self.safePoints.asList()
                self.returnFlag = True
            else:
                ''' Pick another goal '''
                minStart = (self.minDistanceFoodStart + random.randint(numRepeat, numOfFood - 1)) % (numOfFood)
                distanceToFoods = self.getFoodSortedByDistance(currentPosition, gameState)
                self.goal = [foods[0] for i, foods in enumerate(distanceToFoods) if i >= minStart]
                if len(self.goal) == 0:
                    self.goal = self.safePoints.asList()

        actions = self.waStarSearch(gameState, self.heuristicMin, 10)
        if actions is None or len(actions) == 0:
            randomActions = self.getLegalActions(gameState)
            if len(randomActions) > 1 and Directions.STOP in randomActions:
                randomActions.remove(Directions.STOP)
            return random.choice(self.getLegalActions(gameState))
        return actions[0]

    def isReadyOffensing(self, currentPosition):
        ''' Check if agent is about to cross border or has crossed '''
        return self.red and currentPosition[0] >= self.borderPoints.asList()[0][0] \
                or not self.red and currentPosition[0] <= self.borderPoints.asList()[0][0]

    def getDefenderBlockingPosition(self, gameState, opponentsPos):
        '''
        Block position around defenders
        Args:
            opponentsPos:           List of positions of defenders
        Return:
            blockingPositions:      List of positions at and around defenders
        '''
        blockingPositions = opponentsPos.copy()
        for opponentX, opponentY in opponentsPos:
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    if gameState.hasWall(opponentX + dx, opponentY + dy):
                        continue
                    blockingPositions.append((opponentX + dx, opponentY + dy))
        return blockingPositions

    def getFoodSortedByDistance(self, currentPosition, gameState):
        ''' Get Food sorted by distance to current position '''
        foodDistance = {}
        for food in self.getFood(gameState).asList():
            foodDistance[food] = self.getMazeDistance(currentPosition, food)
        return sorted(foodDistance.items(), key = lambda item: item[1])


class HeuristicOffensiveAgent2(HeuristicOffensiveAgent):
    '''
    Defence agent in heuristic search
    
    Attributes:
        safeInvaderDistance:        Safe Distance to invaders
        chaseDistance:              Distance to invader to chase

    '''
    def registerInitialState(self, gameState):
        super().registerInitialState(gameState)
        self.safeInvaderDistance = 3
        self.chaseDistance = 999999
        self.minDistanceFoodStart = 0

    def observationFunction(self, gameState):
        #print(gameState)
        return gameState.makeObservation(self.index)

    def chooseAction(self, gameState):
        '''
        Choose action of defending.
        If it see invaders, use WA* back to chase invaders
        Otherwise, Offence!!
        '''

        currentPosition = gameState.getAgentPosition(self.index)
        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        invaders = [a.getPosition() for a in enemies if a.isPacman and a.getPosition() != None]
        cloestInvader, cloestDistanceToInvader = self.findCloestPositionAndDistance(invaders, currentPosition)
        nextAction = None

        if gameState.getAgentState(self.index).scaredTimer > 0 and cloestDistanceToInvader < self.safeInvaderDistance:
            ''' I am chased by power invader '''
            if self.red and cloestInvader[0] >= currentPosition[0] \
                    or not self.red and cloestInvader[0] <= currentPosition[0]:
                ''' Invader chases from their side '''
                self.goal = [self.start]
            else:
                self.goal = self.getFood(gameState).asList()
            actions = self.waStarSearch(gameState, self.heuristicMin, 10)
            if actions is None or len(actions) == 0:
                randomActions = self.getLegalActions(gameState)
                if len(randomActions) > 1 and Directions.STOP in randomActions:
                    randomActions.remove(Directions.STOP)
                nextAction = random.choice(self.getLegalActions(gameState))
            else:
                nextAction = actions[0]

        elif cloestInvader is not None and cloestDistanceToInvader < self.chaseDistance:
            ''' I am chasing '''
            self.goal = [cloestInvader]
            actions = self.waStarSearch(gameState, self.heuristicMin, 10)
            if actions is None or len(actions) == 0:
                randomActions = self.getLegalActions(gameState)
                if len(randomActions) > 1 and Directions.STOP in randomActions:
                    randomActions.remove(Directions.STOP)
                nextAction = random.choice(self.getLegalActions(gameState))
            else:
                nextAction = actions[0]

        else:
            nextAction = super().chooseAction(gameState)

        #print(self.getSuccessor(gameState, nextAction).isOver())

        return nextAction

class HeuristicDefensiveAgent(HeuristicAgent):
    '''
    Defence agent in heuristic search
    
    Attributes:
        safePosition:               Safe position of defensice agent
        safeDistance:               Safe Distance to invaders
        keyDefendFood:              The "key" food I am defending
        numDefendFood:              Number of food I am defending
        thresholdFood:              Number of food being eaaten then start patrolling
        patrolling:                 Flag of patrolling
        chaseDistance:              Distance to invader to chase

    '''
    def registerInitialState(self, gameState):
        super().registerInitialState(gameState)
        self.goal = self.getFoodYouAreDefending(gameState).asList()

        self.safePosition = None
        self.safeDistance = 3
        self.keyDefendFood = None
        self.numDefendFood = len(self.getFoodYouAreDefending(gameState).asList())
        self.thresholdFood = 2
        self.patrolling = False
        self.chaseDistance = 6

    def chooseAction(self, gameState):
        '''
        Choose action of defending.
        If it see invaders, use WA* back to chase invaders
        Otherwise, travel to center of food it is defending
        Specially, if the my capsule is eaten, back to start
        '''

        currentPosition = gameState.getAgentPosition(self.index)
        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        invaders = [a.getPosition() for a in enemies if a.isPacman and a.getPosition() != None]
        cloestInvader, cloestDistanceToInvader = self.findCloestPositionAndDistance(invaders, currentPosition)


        if gameState.getAgentState(self.index).scaredTimer > 0 and cloestDistanceToInvader < self.safeDistance:
            ''' I am chased by power invader '''
            if self.red and cloestInvader[0] >= currentPosition[0] \
                    or not self.red and cloestInvader[0] <= currentPosition[0]:
                ''' Invader chases from their side '''
                self.safePosition = self.start
            else:
                safePoints = self.getSafePoints(gameState).asList()
                if self.safePosition is None or self.safePosition not in safePoints and currentPosition == self.safePosition:
                    furthestPosition, furthestDistance = self.findFurthestPositionAndDistance(safePoints, currentPosition)
                    self.safePosition = furthestPosition
            #print(cloestDistanceToInvader)
            self.goal = [self.safePosition]

        elif cloestInvader is not None:
            ''' I am chasing '''
            if cloestDistanceToInvader < self.chaseDistance and self.patrolling:
            #if self.patrolling:
                self.patrolling = False
                self.keyDefendFood = None
                #print('here')
            self.goal = [cloestInvader]

        else:
            ''' No one chases me and I do not observe invader '''
            if (self.keyDefendFood is None or 
                self.keyDefendFood not in self.getFoodYouAreDefending(gameState).asList()):
                ''' Just start or be captured or the food is eaten by invader'''
                predictedOpponentStartPosition = (gameState.data.layout.width - 1 - self.start[0], gameState.data.layout.height - 1 - self.start[1])
                self.keyDefendFood, _ = self.findCloestPositionAndDistance(self.getFoodYouAreDefending(gameState).asList(), predictedOpponentStartPosition)
                self.numDefendFood = len(self.getFoodYouAreDefending(gameState).asList())
                self.goal = [self.keyDefendFood]
                #print('111')

            elif self.keyDefendFood is not None and currentPosition == self.keyDefendFood:
                ''' I have arrived the furthest food and make sure not cross the border'''
                #print('222')
                actions = gameState.getLegalActions(self.index)
                borderPoints = self.getSafePoints(gameState).asList()
                if self.red and currentPosition in borderPoints and Directions.EAST in actions:
                    actions.remove(Directions.EAST)
                if not self.red and currentPosition in borderPoints and Directions.WEST in actions:
                    actions.remove(Directions.WEST)
                return random.choice(actions)

            elif not self.patrolling and len(self.getFoodYouAreDefending(gameState).asList()) < self.numDefendFood - self.thresholdFood:
                self.keyDefendFood = self.getCenterDefendFood(gameState)
                self.patrolling = True
                self.goal = [self.keyDefendFood]
                #print('333')

            else:
                #print('444')
                self.goal = [self.keyDefendFood]
            #print(self.goal)

        actions = self.waStarSearch(gameState, self.heuristicMin, 10)
        if actions is None or len(actions) == 0:
            return Directions.STOP
        return actions[0]

    
    def getCenterDefendFood(self, gameState):
        ''' Get next food from food I am defending '''
        defendfoodPositions = self.getFoodYouAreDefending(gameState).asList()
        distanceToFoods = {}
        for d in defendfoodPositions:
            distanceToFoods[d] = self.getMazeDistance(gameState.getAgentPosition(self.index), d)
        distanceToFoods = sorted(distanceToFoods.items(), key = lambda item: item[1])
        return distanceToFoods[len(defendfoodPositions) // 2][0]
    
