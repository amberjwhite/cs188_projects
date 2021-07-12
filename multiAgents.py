# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        print("legalMoves = ", legalMoves)
        print("scores =", scores)
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)


        successorGameState = currentGameState.generatePacmanSuccessor(action)
        currPos = currentGameState.getPacmanPosition()
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        currFood = currentGameState.getFood()
        newGhostPos = successorGameState.getGhostPosition(1)
        newGhostStates = successorGameState.getGhostStates()
        currGhostStates = currentGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        currScaredTimes = [ghostState.scaredTimer for ghostState in currGhostStates]

        "*** YOUR CODE HERE ***"

        #print("successorGameState : \n", currentGameState.generatePacmanSuccessor(action))

        #check if its legal move
        if (not action in currentGameState.getLegalActions()):
            return -1

        #ghost scared?
        ghostIndex = 1
        if currScaredTimes[0] > 0:   #if ghosts are scared
            #find the current closest ghost
            distFromGhosts = []
            i = 1
            for ghostState in currGhostStates:
                distFromGhosts += [manhattanDistance(currentGameState.getGhostPosition(i), currentGameState.getPacmanPosition())]
                i += 1
            #find index of closest ghost
            minDist = min(distFromGhosts)
            closestGhostIndex = [index for index in range(0, len(distFromGhosts)) if distFromGhosts[index] == minDist]

            #check if pacman can reach ghost
            if (minDist < currScaredTimes[closestGhostIndex[0]]):     #pacman can reach ghost
                #check if action gets pacman closer to ghost
                currGhostPos = currentGameState.getGhostPosition(closestGhostIndex[0] + 1)
                if(manhattanDistance(newPos, newGhostPos) < manhattanDistance(currPos, currGhostPos)): 
                    
                    return 200;

        #check if new pos is ghost pos
        for i in range(1, len(newGhostStates) + 1):
            if (successorGameState.getGhostPosition(i) == newPos) | (successorGameState.getGhostPosition(i) == currPos) | (currentGameState.getGhostPosition(i) == newPos):
                return -1
            elif (manhattanDistance(successorGameState.getGhostPosition(i), newPos) <= 2):
                return -0.1


        top = currFood.height - 2
        right = currFood.width - 2
        left = 1
        bottom = 1

        #check if newPos is food

        if (currFood[newPos[0]][newPos[1]]) | ((newPos[0],newPos[1]) in currentGameState.getCapsules()):    
            return 100
        else:
            if (action == 'North') | (action == 'South'):
                #choose between N and S
                foodNorth = 0
                foodSouth = 0
                for x in range(left, right + 1):
                    for y in range(currPos[1], top + 1):
                        if (currFood[x][y]):
                            foodNorth += 1
                        elif ((x,y) in currentGameState.getCapsules()):
                            foodNorth += 4
                        elif ((x,y) in currentGameState.getGhostPositions()):
                            if (foodNorth > 3):
                                foodNorth -= 3


                for x in range(1, right + 1):
                    for y in range(1, currPos[1] + 1):
                        if (currFood[x][y]):
                            foodSouth += 1
                        elif ((x,y) in currentGameState.getCapsules()):
                            foodSouth += 4
                        elif ((x,y) in currentGameState.getGhostPositions()):
                            if (foodSouth > 3):
                                foodSouth -= 3


                if (action == 'North') & (foodNorth >= foodSouth):
                    #check that if you move North, you will not move south next
                    newSouthFood = 0
                    for x in range(left, right + 1):
                        if (currFood[x][currPos[1]]):
                            newSouthFood += 1
                    if ((foodSouth + newSouthFood) > (foodNorth - newSouthFood)):
                        return random.choice([0.1, 0.2])
                    else:
                        return foodNorth

                elif (action == 'South') & (foodSouth >= foodNorth):

                    newNorthFood = 0
                    for x in range(left, right + 1):
                        if (currFood[x][currPos[1]]):
                            newNorthFood += 1
                    if ((foodNorth + newNorthFood) > (foodSouth - newNorthFood)):
                        return random.choice([0.1, 0.2])
                    else:
                        return foodSouth
                else:
                    return 0

            elif (action == 'East') | (action == 'West'):
                #choose between E and W
                foodEast = 0
                foodWest = 0
                for x in range(currPos[0], right + 1):
                    for y in range(bottom, top + 1):
                        if (currFood[x][y]):
                            foodEast += 1
                        elif ((x,y) in currentGameState.getCapsules()):
                            foodEast += 4
                        elif ((x,y) in currentGameState.getGhostPositions()):
                            if (foodEast > 3):
                                foodEast -= 3

                for x in range(left, currPos[0] + 1):
                    for y in range(bottom, top + 1):
                        if (currFood[x][y]):
                            foodWest += 1
                        elif ((x,y) in currentGameState.getCapsules()):
                            foodWest += 4
                        elif ((x,y) in currentGameState.getGhostPositions()):
                            if (foodWest > 3):
                                foodWest -= 3

                if (action == 'East') & (foodEast >= foodWest):


                    newWestFood = 0
                    for y in range(bottom, top + 1):
                        if (currFood[currPos[0]][y]):
                            newWestFood += 1
                    if ((foodWest + newWestFood) > (foodEast - newWestFood)):
                        return random.choice([0.1, 0.2])
                    else:
                        return foodEast
                elif (action == 'West') & (foodWest >= foodEast):


                    newEastFood = 0
                    for y in range(bottom, top + 1):
                        if (currFood[currPos[0]][y]):
                            newEastFood += 1

                    if ((foodEast + newEastFood) > (foodWest - newEastFood)):
                        return random.choice([0.1, 0.2])
                    else:
                        return foodWest
                else:
                    return 0

            elif (action == 'Stop'):
                return 0



        


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        #global depth
        #global numAgents



        numAgents = gameState.getNumAgents()

        #print("depth: ", self.depth)
        #print("number of agents: ", numAgents)
        #print('')


        def value(state, index, currDepth):
            if (currDepth/numAgents == self.depth):
                score = self.evaluationFunction(state)
                return (score, 'terminal state')
            elif (index%numAgents == 0):
                #MAX
                v, action = max_value(state, index%numAgents, currDepth)
                return (v, action)
            else:
                #MIN
                v, action = min_value(state, index%numAgents, currDepth)
                return (v, action)


        def max_value(state, index, currDepth):
            bestValue = -1 * float('inf')
            bestAction = 'Stop'
            #print("type (pacman) ", index)
            if (state.isLose() | state.isWin()):
                return (state.getScore(), bestAction)
            for action in state.getLegalActions(index):
                successorState = state.generateSuccessor(index, action)
                #print('pacman action: ', action)
                successorValue, prevAction = value(successorState, index + 1, currDepth + 1)

                if (successorValue > bestValue):
                    bestValue = successorValue
                    bestAction = action

            return (bestValue, bestAction)
                


        def min_value(state, index, currDepth):
            bestValue = float('inf')
            bestAction = 'Stop'
            
            if (state.isLose() | state.isWin()):
                return (state.getScore(), bestAction)
                    
            for action in state.getLegalActions(index):
                successorState = state.generateSuccessor(index, action)
                #print('ghost action: ', action)
                successorValue, prevAction = value(successorState, index + 1, currDepth + 1)
                
                if (successorValue < bestValue):
                    bestValue = successorValue
                    bestAction = action

            return (bestValue, bestAction)

        action = value(gameState, 0, 0)[1]
        #print('ACTION: ', action)
        return action

            


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        numAgents = gameState.getNumAgents()

        #print("depth: ", self.depth)
        #print("number of agents: ", numAgents)
        #print('')


        def value(state, index, currDepth, alpha, beta):
            if (currDepth/numAgents == self.depth):
                score = self.evaluationFunction(state)
                return (score, 'terminal state')
            elif (index%numAgents == 0):
                #MAX
                v, action = max_value(state, index%numAgents, currDepth, alpha, beta)
                return (v, action)
            else:
                #MIN
                v, action = min_value(state, index%numAgents, currDepth, alpha, beta)
                return (v, action)


        def max_value(state, index, currDepth, alpha, beta):
            bestValue = -1 * float('inf')
            bestAction = 'Stop'
            #print("type (pacman) ", index)
            if (state.isLose() | state.isWin()):
                return (state.getScore(), bestAction)
            for action in state.getLegalActions(index):
                successorState = state.generateSuccessor(index, action)
                #print('pacman action: ', action)
                successorValue, prevAction = value(successorState, index + 1, currDepth + 1, alpha, beta)

                if (successorValue > beta):
                    return (successorValue, prevAction)

                alpha = max(alpha, successorValue)


                if (successorValue > bestValue):
                    bestValue = successorValue
                    bestAction = action

            return (bestValue, bestAction)
                


        def min_value(state, index, currDepth, alpha, beta):
            bestValue = float('inf')
            bestAction = 'Stop'
            
            if (state.isLose() | state.isWin()):
                return (state.getScore(), bestAction)
                    
            for action in state.getLegalActions(index):
                successorState = state.generateSuccessor(index, action)
                #print('ghost action: ', action)
                successorValue, prevAction = value(successorState, index + 1, currDepth + 1, alpha, beta)
                
                if (successorValue < alpha):
                    return (successorValue, prevAction)

                beta = min(beta, successorValue)

                if (successorValue < bestValue):
                    bestValue = successorValue
                    bestAction = action

            return (bestValue, bestAction)


        action = value(gameState, 0, 0, -1 * float('inf'), float('inf'))[1]
        #print('ACTION: ', action)
        return action

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        numAgents = gameState.getNumAgents()

        #print("depth: ", self.depth)
        #print("number of agents: ", numAgents)
        #print('')


        def value(state, index, currDepth):
            if (currDepth/numAgents == self.depth):
                score = self.evaluationFunction(state)
                return (score, 'terminal state')
            elif (index%numAgents == 0):
                #MAX
                v, action = max_value(state, index%numAgents, currDepth)
                return (v, action)
            else:
                #MIN
                v = exp_value(state, index%numAgents, currDepth)
                action = 'None'
                return (v, action)


        def max_value(state, index, currDepth):
            bestValue = -1 * float('inf')
            bestAction = 'Stop'
            #print("type (pacman) ", index)
            if (state.isLose() | state.isWin()):
                return (state.getScore(), bestAction)
            for action in state.getLegalActions(index):
                successorState = state.generateSuccessor(index, action)
                #print('pacman action: ', action)
                successorValue, prevAction = value(successorState, index + 1, currDepth + 1)

                if (successorValue > bestValue):
                    bestValue = successorValue
                    bestAction = action

            return (bestValue, bestAction)
                
        def exp_value(state, index, currDepth):
            val = 0
            if (state.isLose() | state.isWin()):
                return state.getScore()
            for action in state.getLegalActions(index):
                p = 1/(len(state.getLegalActions(index)))
                successorState = state.generateSuccessor(index, action)
                successorValue, prevAction = value(successorState, index + 1, currDepth + 1)
                val += p * successorValue
            return val

        action = value(gameState, 0, 0)[1]
        #print('ACTION: ', action)
        return action

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <high value for states:
                                        with scared ghosts
                                        with little food
                                        with large distance from ghost
                   low value for states:
                                        with lots of food
                                        with small distance from ghost

                                        >
    """
    "*** YOUR CODE HERE ***"
    currPos = currentGameState.getPacmanPosition()
    currFood = currentGameState.getFood()
    currGhostStates = currentGameState.getGhostStates()
    currScaredTimes = [ghostState.scaredTimer for ghostState in currGhostStates]

    ghostIndex = 1
    if currScaredTimes[0] > 0:   #if ghosts are scared
        #find the current closest ghost
        distFromGhosts = []
        i = 1
        for ghostState in currGhostStates:
            distFromGhosts += [manhattanDistance(currentGameState.getGhostPosition(i), currentGameState.getPacmanPosition())]
            i += 1
        #find index of closest ghost
        minDist = min(distFromGhosts)
        closestGhostIndex = [index for index in range(0, len(distFromGhosts)) if distFromGhosts[index] == minDist]

        #check if pacman can reach ghost
        if (minDist < currScaredTimes[closestGhostIndex[0]]):     #pacman can reach ghost
            #check if action gets pacman closer to ghost
            currGhostPos = currentGameState.getGhostPosition(closestGhostIndex[0] + 1)
            #if(manhattanDistance(newPos, newGhostPos) < manhattanDistance(currPos, currGhostPos)): 
            ghostDistFactor = 2
        else:
            ghostDistFactor = 1/minDist
    else:
        #calculate distance from ghosts
        ghostDistFactor = 1
        minDist = float('inf')
        # distance to closest ghost
        for i in range(1, len(currGhostStates) + 1):
            distFromCurrGhost = manhattanDistance(currentGameState.getGhostPosition(i), currentGameState.getPacmanPosition())
            if (distFromCurrGhost < minDist):
                minDist = distFromCurrGhost
        if (minDist != 0):
            ghostDistFactor = 1/minDist
    

    top = currFood.height - 2
    right = currFood.width - 2
    left = 1
    bottom = 1

    #calculate ratio of food to space
    space = top * right
    foodCount = 0
    distToClosestFood = 1/max(top, right)
    foodSum = 0
    capSum = 0
    distToClosestCap = 1/max(top, right)
    capLocs = currentGameState.getCapsules()

    for x in range(left, right):
        for y in range(bottom, top):
            if (currFood[x][y]):
                foodCount += 1
                foodDist = manhattanDistance((x, y), currPos)
                foodSum += foodDist
                if (foodDist < 1/distToClosestFood):
                    distToClosestFood = 1/foodDist
            elif ((x, y) in capLocs):
                capSum += 1
                capDist = manhattanDistance((x,y), currPos)
                if (capDist < 1/distToClosestCap):
                    distToClosestCap = 1/capSum

    if (foodCount == 0):
        foodRatio = 1
        foodCount = 1
    else:
        foodRatio = foodCount/space

    """
    minimize food on the board
    maximize score
    minimize distance to food
        or minimize dist to closest food

    """
    return 0.2 * distToClosestFood + 0.1 * ghostDistFactor + 0.4 * currentGameState.getScore() + 0.3 * distToClosestCap + 0.2/foodCount

better = betterEvaluationFunction

