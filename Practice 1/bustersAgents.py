from __future__ import print_function
# bustersAgents.py
# ----------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley.=
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

from wekaI import Weka
from builtins import range
from builtins import object
import util
from game import Agent
from game import Directions
from game import Configuration
from keyboardAgents import KeyboardAgent
from distanceCalculator import manhattanDistance
import inference
import busters
import os


class NullGraphics(object):
    "Placeholder for graphics"

    def initialize(self, state, isBlue=False):
        pass

    def update(self, state):
        pass

    def pause(self):
        pass

    def draw(self, state):
        pass

    def updateDistributions(self, dist):
        pass

    def finish(self):
        pass


class KeyboardInference(inference.InferenceModule):
    """
    Basic inference module for use with the keyboard.
    """

    def initializeUniformly(self, gameState):
        "Begin with a uniform distribution over ghost positions."
        self.beliefs = util.Counter()
        for p in self.legalPositions: self.beliefs[p] = 1.0
        self.beliefs.normalize()

    def observe(self, observation, gameState):
        noisyDistance = observation
        emissionModel = busters.getObservationDistribution(noisyDistance)
        pacmanPosition = gameState.getPacmanPosition()
        allPossible = util.Counter()
        for p in self.legalPositions:
            trueDistance = util.manhattanDistance(p, pacmanPosition)
            if emissionModel[trueDistance] > 0:
                allPossible[p] = 1.0
        allPossible.normalize()
        self.beliefs = allPossible

    def elapseTime(self, gameState):
        pass

    def getBeliefDistribution(self):
        return self.beliefs


class BustersAgent(object):
    "An agent that tracks and displays its beliefs about ghost positions."

    def __init__(self, index=0, inference="ExactInference", ghostAgents=None, observeEnable=True,
                 elapseTimeEnable=True):
        inferenceType = util.lookup(inference, globals())
        self.inferenceModules = [inferenceType(a) for a in ghostAgents]
        self.observeEnable = observeEnable
        self.elapseTimeEnable = elapseTimeEnable
        self.weka = Weka()
        self.weka.start_jvm()

    def registerInitialState(self, gameState):
        "Initializes beliefs and inference modules"
        import __main__
        self.display = __main__._display
        for inference in self.inferenceModules:
            inference.initialize(gameState)
        self.ghostBeliefs = [inf.getBeliefDistribution() for inf in self.inferenceModules]
        self.firstMove = True

    def observationFunction(self, gameState):
        "Removes the ghost states from the gameState"
        agents = gameState.data.agentStates
        gameState.data.agentStates = [agents[0]] + [None for i in range(1, len(agents))]
        return gameState

    def getAction(self, gameState):
        "Updates beliefs, then chooses an action based on updated beliefs."
        # for index, inf in enumerate(self.inferenceModules):
        #    if not self.firstMove and self.elapseTimeEnable:
        #        inf.elapseTime(gameState)
        #    self.firstMove = False
        #    if self.observeEnable:
        #        inf.observeState(gameState)
        #    self.ghostBeliefs[index] = inf.getBeliefDistribution()
        # self.display.updateDistributions(self.ghostBeliefs)
        return self.chooseAction(gameState)

    def chooseAction(self, gameState):
        "By default, a BustersAgent just stops.  This should be overridden."
        x = self.getState(gameState)
        a = self.weka.predict("j48_categ2.model", x, "training_keyboard_categ.arff")
        legal = gameState.getLegalPacmanActions()
        legal.remove("Stop")
        if a not in legal:
            a = random.choice(legal)
        return a

    def getState(self, gameState):
        legal = ['North', 'South', 'West', 'East']
        data = [gameState.getPacmanPosition()[0], gameState.getPacmanPosition()[1]]
        for i in legal:
            if i in gameState.getLegalPacmanActions():
                data.append(str(1))
            else:
                data.append(str(0))
        for i in gameState.data.ghostDistances:
            if i == None:
                i = 0
            data.append(i)
        data.append(gameState.getScore())
        print(data)
        return data


class BustersKeyboardAgent(BustersAgent, KeyboardAgent):
    "An agent controlled by the keyboard that displays beliefs about ghost positions."

    def __init__(self, index=0, inference="KeyboardInference", ghostAgents=None):
        KeyboardAgent.__init__(self, index)
        BustersAgent.__init__(self, index, inference, ghostAgents)
        self.countActions = 0

    def getAction(self, gameState):
        return BustersAgent.getAction(self, gameState)

    def chooseAction(self, gameState):
        self.countActions = self.countActions + 1
        self.printLineData(gameState)
        return KeyboardAgent.getAction(self, gameState)

    def printLineData(self, gameState):
        # condition to check if the document exists, if not it should create it.
        if not os.path.exists('all_data_pacman.arff'):
            f = open('all_data_pacman.arff', 'w+')
        # condition to check that ,if it is empty, it should create the header line.
        if os.path.getsize('all_data_pacman.arff') == 0:
            f = open('all_data_pacman.arff', 'w')
            header = '@RELATION all_data_pacman.arff\n' \
                     '\t@ATTRIBUTE TICK NUMERIC\n' \
                     '\t@ATTRIBUTE "PACMAN POSITION x" NUMERIC"\n' \
                     '\t@ATTRIBUTE "PACMAN POSITION y" NUMERIC"\n' \
                     '\t@ATTRIBUTE NORTH {1, 0}\n' \
                     '\t@ATTRIBUTE SOUTH {1, 0}\n' \
                     '\t@ATTRIBUTE WEST {1, 0}\n' \
                     '\t@ATTRIBUTE EAST {1, 0}\n' \
                     '\t@ATTRIBUTE STOP {1, 0}\n' \
                     '\t@ATTRIBUTE "GHOST1 POSITION x" NUMERIC\n' \
                     '\t@ATTRIBUTE "GHOST1 POSITION y" NUMERIC\n' \
                     '\t@ATTRIBUTE "GHOST2 POSITION x" NUMERIC\n' \
                     '\t@ATTRIBUTE "GHOST2 POSITION y" NUMERIC\n' \
                     '\t@ATTRIBUTE "GHOST3 POSITION x" NUMERIC\n' \
                     '\t@ATTRIBUTE "GHOST3 POSITION y" NUMERIC\n' \
                     '\t@ATTRIBUTE "GHOST4 POSITION x" NUMERIC\n' \
                     '\t@ATTRIBUTE "GHOST4 POSITION y" NUMERIC\n' \
                     '\t@ATTRIBUTE "GHOST1 DISTANCE" NUMERIC\n' \
                     '\t@ATTRIBUTE "GHOST2 DISTANCE" NUMERIC\n' \
                     '\t@ATTRIBUTE "GHOST3 DISTANCE" NUMERIC\n' \
                     '\t@ATTRIBUTE "GHOST4 DISTANCE" NUMERIC\n' \
                     '\t@ATTRIBUTE "PAC DOTS" NUMERIC\n' \
                     '\t@ATTRIBUTE SCORE NUMERIC\n' \
                     '\t@ATTRIBUTE DIRECTION {"North", "West", "South", "East", "Stop"}\n' \
                     '@DATA\n'
            f.write(header)
            f.write('\n')
            f.close()

        legal = ['North', 'South', 'West', 'East', 'Stop']

        f = open('all_data_pacman.arff', 'a')
        data = str(self.countActions) + ', ' + \
               str(gameState.getPacmanPosition()[0]) + ', ' + \
               str(gameState.getPacmanPosition()[1]) + ', '

        for i in legal:
            if i in gameState.getLegalPacmanActions():
                data += '1' + ', '
            else:
                data += '0' + ', '

        for i in range(len(gameState.getGhostPositions())):
            for j in range(2):
                data += str(gameState.getGhostPositions()[i][j]) + ', '

        for i in gameState.data.ghostDistances:
            if i == None:
                i = 0
            data += str(i) + ', '

        data += str(gameState.getNumFood()) + ', ' + \
                str(gameState.getScore()) + ', ' + \
                '"' + str(gameState.data.agentStates[0].getDirection()) + '"'

        f.write(data)
        f.write('\n')
        f.close()


from distanceCalculator import Distancer
from game import Actions
from game import Directions
import random, sys

'''Random PacMan Agent'''


class RandomPAgent(BustersAgent):

    def registerInitialState(self, gameState):
        BustersAgent.registerInitialState(self, gameState)
        self.distancer = Distancer(gameState.data.layout, False)

    ''' Example of counting something'''

    def countFood(self, gameState):
        food = 0
        for width in gameState.data.food:
            for height in width:
                if (height == True):
                    food = food + 1
        return food

    ''' Print the layout'''

    def printGrid(self, gameState):
        table = ""
        ##print(gameState.data.layout) ## Print by terminal
        for x in range(gameState.data.layout.width):
            for y in range(gameState.data.layout.height):
                food, walls = gameState.data.food, gameState.data.layout.walls
                table = table + gameState.data._foodWallStr(food[x][y], walls[x][y]) + ","
        table = table[:-1]
        return table

    def chooseAction(self, gameState):
        move = Directions.STOP
        legal = gameState.getLegalActions(0)  ##Legal position from the pacman
        move_random = random.randint(0, 3)
        if (move_random == 0) and Directions.WEST in legal:  move = Directions.WEST
        if (move_random == 1) and Directions.EAST in legal: move = Directions.EAST
        if (move_random == 2) and Directions.NORTH in legal:   move = Directions.NORTH
        if (move_random == 3) and Directions.SOUTH in legal: move = Directions.SOUTH
        return move


class GreedyBustersAgent(BustersAgent):
    "An agent that charges the closest ghost."

    def registerInitialState(self, gameState):
        "Pre-computes the distance between every two points."
        BustersAgent.registerInitialState(self, gameState)
        self.distancer = Distancer(gameState.data.layout, False)

    def chooseAction(self, gameState):
        """
        First computes the most likely position of each ghost that has
        not yet been captured, then chooses an action that brings
        Pacman closer to the closest ghost (according to mazeDistance!).

        To find the mazeDistance between any two positions, use:
          self.distancer.getDistance(pos1, pos2)

        To find the successor position of a position after an action:
          successorPosition = Actions.getSuccessor(position, action)

        livingGhostPositionDistributions, defined below, is a list of
        util.Counter objects equal to the position belief
        distributions for each of the ghosts that are still alive.  It
        is defined based on (these are implementation details about
        which you need not be concerned):

          1) gameState.getLivingGhosts(), a list of booleans, one for each
             agent, indicating whether or not the agent is alive.  Note
             that pacman is always agent 0, so the ghosts are agents 1,
             onwards (just as before).

          2) self.ghostBeliefs, the list of belief distributions for each
             of the ghosts (including ghosts that are not alive).  The
             indices into this list should be 1 less than indices into the
             gameState.getLivingGhosts() list.
        """
        pacmanPosition = gameState.getPacmanPosition()
        legal = [a for a in gameState.getLegalPacmanActions()]
        livingGhosts = gameState.getLivingGhosts()
        livingGhostPositionDistributions = \
            [beliefs for i, beliefs in enumerate(self.ghostBeliefs)
             if livingGhosts[i + 1]]
        return Directions.EAST


class BasicAgentAA(BustersAgent):

    def registerInitialState(self, gameState):
        BustersAgent.registerInitialState(self, gameState)
        self.distancer = Distancer(gameState.data.layout, False)
        self.countActions = 0
        # condition to check if the document exists, if not it should create it.
        if not os.path.exists('all_data_pacman.arff'):
            f = open('all_data_pacman.arff', 'w+')
        # condition to check that ,if it is empty, it should create the header line.
        if os.path.getsize('all_data_pacman.arff') == 0:
            f = open('all_data_pacman.arff', 'w')
            f.write('TICK,PACMAN POSITION,LEGAL ACTIONS,GHOST POSITIONS,GHOST DISTANCES,PAC DOTS,SCORE, DIRECTION')
            f.write('\n')
            f.close()

    ''' Example of counting something'''

    def countFood(self, gameState):
        food = 0
        for width in gameState.data.food:
            for height in width:
                if (height == True):
                    food = food + 1
        return food

    ''' Print the layout'''

    def printGrid(self, gameState):
        table = ""
        # print(gameState.data.layout) ## Print by terminal
        for x in range(gameState.data.layout.width):
            for y in range(gameState.data.layout.height):
                food, walls = gameState.data.food, gameState.data.layout.walls
                table = table + gameState.data._foodWallStr(food[x][y], walls[x][y]) + ","
        table = table[:-1]
        return table

    def printInfo(self, gameState):
        print("---------------- TICK ", self.countActions, " --------------------------")
        # Map size
        width, height = gameState.data.layout.width, gameState.data.layout.height
        print("Width: ", width, " Height: ", height)
        # Pacman position
        print("Pacman position: ", gameState.getPacmanPosition())
        # Legal actions for Pacman in current position
        print("Legal actions: ", gameState.getLegalPacmanActions())
        # Pacman direction
        print("Pacman direction: ", gameState.data.agentStates[0].getDirection())
        # Number of ghosts
        print("Number of ghosts: ", gameState.getNumAgents() - 1)
        # Alive ghosts (index 0 corresponds to Pacman and is always false)
        print("Living ghosts: ", gameState.getLivingGhosts())
        # Ghosts positions
        print("Ghosts positions: ", gameState.getGhostPositions())
        # Ghosts directions
        print("Ghosts directions: ",
              [gameState.getGhostDirections().get(i) for i in range(0, gameState.getNumAgents() - 1)])
        # Manhattan distance to ghosts
        print("Ghosts distances: ", gameState.data.ghostDistances)
        # Pending pac dots
        print("Pac dots: ", gameState.getNumFood())
        # Manhattan distance to the closest pac dot
        print("Distance nearest pac dots: ", gameState.getDistanceNearestFood())
        # Map walls
        print("Map:")
        print(gameState.getWalls())
        # Score
        print("Score: ", gameState.getScore())

    def chooseAction(self, gameState):
        self.countActions = self.countActions + 1
        #print(self.countActions)
        x = [2,11,0,0,1,1,10,4,8,16,2,-3]
        print(x)
        # Profe, aqui el move da error, por la funcion de predict. 
        move = self.weka.predict("./ibk.model", x, "./training_keyboard_modified.arff")
        #print(move)
        

        return Directions.move

    def printLineData(self, gameState):
        f = open('all_data_pacman.arff', 'a')
        data = str(self.countActions) + ', ' + \
               str(gameState.getPacmanPosition()) + ', ' + \
               str(gameState.getLegalPacmanActions()) + ', ' + \
               str(gameState.getGhostPositions()) + ', ' + \
               str(gameState.data.ghostDistances) + ', ' + \
               str(gameState.getNumFood()) + ', ' + \
               str(gameState.getScore()) + ', ' + \
               str(gameState.data.agentStates[0].getDirection())

        f.write(data)
        f.write('\n')
        f.close()
