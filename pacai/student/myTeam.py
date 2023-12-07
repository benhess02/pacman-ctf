from pacai.agents.capture.reflex import ReflexCaptureAgent
from pacai.util import Directions
import random

class OffensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that seeks food.
    This agent will give you an idea of what an offensive agent might look like,
    but it is by no means the best or only way to build an offensive agent.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index)

    def getFeatures(self, gameState, action):
        features = super().getFeatures(gameState, action)
        successor = self.getSuccessor(gameState, action)
        features['successorScore'] = self.getScore(successor)

        # Get positions and other relevant state info
        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()

        # Compute distance to the nearest food
        foodList = self.getFood(successor).asList()
        if foodList:  # This should always be True, but better safe than sorry
            # Strategic Food Targeting
            # Weigh food based on distance and safety
            foodScores = self.calculateFoodScores(foodList, myPos, features)
            bestFood = min(foodScores)[1]
            features['distanceToFood'] = self.getMazeDistance(myPos, bestFood)
        else:
            features['distanceToFood'] = 0

        # Risk Assessment from Ghosts
        ghostStates = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        ghosts = [ghost for ghost in ghostStates if not ghost.isPacman() and ghost.getPosition() is not None]
        features['ghostRisk'] = self.calculateGhostRisk(ghosts, myPos)
        if ghosts:
            distancesToGhosts = [self.getMazeDistance(myPos, ghost.getPosition()) for ghost in ghosts]
            # Risk increases as pacman gets closer to ghost
            features['ghostRisk'] = sum([1 / (d + 1) for d in distancesToGhosts])

        # Power Capsule Strategy
        capsules = self.getCapsules(successor)
        if capsules:
            features['distanceToCapsule'] = min([self.getMazeDistance(myPos, capsule) for capsule in capsules])
            # Higher value when ghosts are close
            features['capsuleValue'] = 1 / (1 + min(distancesToGhosts)) if ghosts else 0
        else:
            features['distanceToCapsule'] = 0
            features['capsuleValue'] = 0

        return features
    
    def calculateFoodScores(self, foodList, myPos, gameState):
        foodScores = []
        ghostPositions = [gameState.getAgentState(i).getPosition() for i in self.getOpponents(gameState) 
                        if not gameState.getAgentState(i).isPacman() and gameState.getAgentState(i).getPosition() is not None]

        for food in foodList:
            distanceToFood = self.getMazeDistance(myPos, food)
            safetyScore = sum([1 / (self.getMazeDistance(food, ghostPos) + 1) for ghostPos in ghostPositions if ghostPos is not None])

            # Calculate food density around this food item
            neighbors = [f for f in foodList if self.getMazeDistance(food, f) < 5]
            foodDensity = len(neighbors)

            # Score based on distance, safety, and density
            score = (1 / (distanceToFood + 1)) * (1 - safetyScore) * foodDensity
            foodScores.append((score, food))

        return foodScores
    
    def calculateGhostRisk(self, ghosts, myPos, gameState):
        ghostRisk = 0
        for ghost in ghosts:
            distance = self.getMazeDistance(myPos, ghost.getPosition())
            scared = gameState.getAgentState(ghost.index).scaredTimer > 0

            # Increase risk based on proximity and decrease if the ghost is scared
            riskFactor = (1 / (distance + 1)) * (0.5 if scared else 1)

            # Consider the direction of the ghost
            if self.isGhostApproaching(myPos, ghost, gameState):
                riskFactor *= 1.5  # Increase risk if the ghost is approaching

            ghostRisk += riskFactor

        return ghostRisk

    def isGhostApproaching(self, myPos, ghost, gameState):
        ghostPos = ghost.getPosition()
        ghostDirection = gameState.getAgentState(ghost.index).getDirection()
        nextGhostPos = self.getNextPosition(ghostPos, ghostDirection)

        return self.getMazeDistance(myPos, nextGhostPos) < self.getMazeDistance(myPos, ghostPos)
    
    def getWeights(self, gameState, action):
        return {
            'successorScore': 100,
            # Adjusted to encourage strategic movement
            'distanceToFood': -2,
            # Big negative weight to avoid ghosts
            'ghostRisk': -200,
            # Encourage going for capsule when strategic
            'distanceToCapsule': -10,
            'capsuleValue': 100
        }

 
class DefensiveAgent(ReflexCaptureAgent):

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def chooseAction(self, gameState):
        action = self.chooseDefensiveAction(gameState)
        return action

    def chooseDefensiveAction(self, gameState):
        # Choose the best defensive action
        if self.shouldChaseOpponent(gameState):
            return self.chaseOpponent(gameState)
        elif self.shouldSetAmbush(gameState):
            return self.setAmbush(gameState)
        elif gameState.getAgentState(self.index).isScared():
            return self.handleScaredState(gameState)
        else:
            return self.patrol(gameState)

    def patrol(self, gameState):
        foodList = self.getFoodYouAreDefending(gameState).asList()
        lateGame = len(foodList) < 10
        myPos = gameState.getAgentState(self.index).getPosition()

        if lateGame:
            # Focus on protecting remaining food clusters
            clusterCenter = self.findFoodClusterCenter(foodList)
            # Move towards the cluster center if it's identified
            return self.moveToPosition(gameState, clusterCenter) if clusterCenter else self.moveToRandomPosition(gameState)
        else:
            # Early game: focus on areas with recent opponent activity
            opponentPositions = [gameState.getAgentState(i).getPosition() for i in self.getOpponents(gameState) if gameState.getAgentState(i).getPosition() is not None]
            # Prioritize food closest to recent opponent activity and closest to the agent
            targetPoint = min(foodList, key=lambda food: (min(self.getMazeDistance(food, opp) for opp in opponentPositions if opp is not None), self.getMazeDistance(myPos, food)))
            return self.moveToPosition(gameState, targetPoint)
        
    def moveToRandomPosition(self, gameState):
        # Move to a random position if no specific target is identified
        actions = gameState.getLegalActions(self.index)
        return random.choice(actions) if actions else Directions.STOP
    
    def findFoodClusterCenter(self, foodList):
        if not foodList:
            return None

        # Clustering based on proximity
        clusters = {}
        for food in foodList:
            foundCluster = False
            for center in clusters:
                if self.getMazeDistance(food, center) < 5:  # Threshold for being in the same cluster
                    clusters[center].append(food)
                    foundCluster = True
                    break
            if not foundCluster:
                clusters[food] = [food]

        # Find the largest cluster
        largestCluster = max(clusters.values(), key=len)

        # Calculate the average position of the largest cluster
        x_avg = sum(food[0] for food in largestCluster) / len(largestCluster)
        y_avg = sum(food[1] for food in largestCluster) / len(largestCluster)
        return (x_avg, y_avg)

    def getPatrolPoints(self, foodList, gameState):
        # Identify the most vulnerable food based on opponent positions and paths
        opponentIndices = self.getOpponents(gameState)
        opponentPositions = [gameState.getAgentState(i).getPosition() for i in opponentIndices if gameState.getAgentState(i).getPosition() is not None]

        # Calculate vulnerability score for each food item
        foodVulnerability = {}
        for food in foodList:
            foodVulnerability[food] = sum(self.getMazeDistance(food, oppPos) for oppPos in opponentPositions if oppPos is not None)

        # Sort food by vulnerability and return the most vulnerable positions
        return sorted(foodList, key=lambda food: foodVulnerability.get(food, 0))[:2]

    def moveToPosition(self, gameState, position):
        # Choose the action that brings the agent closest to the target position
        actions = gameState.getLegalActions(self.index)
        bestAction = None
        minDistance = float('inf')
        for action in actions:
            successor = self.getSuccessor(gameState, action)
            pos = successor.getAgentState(self.index).getPosition()
            distance = self.getMazeDistance(pos, position)
            if distance < minDistance:
                minDistance = distance
                bestAction = action
        return bestAction

    def chaseOpponent(self, gameState):
        # Predicting the opponent's path and intercepting
        opponents = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        visibleOpponents = [opp for opp in opponents if opp.getPosition() is not None and opp.isPacman()]

        if visibleOpponents:
            myPos = gameState.getAgentState(self.index).getPosition()
            # Choose the opponent with the shortest path to intercept
            targetOpponent = min(visibleOpponents, key=lambda opp: self.getMazeDistance(myPos, opp.getPosition()))
            return self.moveToPosition(gameState, targetOpponent.getPosition())
        
    def predictOpponentPath(self, opponent, gameState):
        path = []
        position = opponent.getPosition()
        direction = opponent.getDirection()
        for _ in range(5):  # Predict next 5 moves
            nextPosition = self.getNextPosition(position, direction)
            path.append(nextPosition)
            # Update direction based on layout and opponent behavior
            direction = self.chooseNextDirection(nextPosition, gameState)
        return path

    def getNextPosition(self, position, direction):
        x, y = position
        if direction == Directions.NORTH:
            return (x, y + 1)
        elif direction == Directions.SOUTH:
            return (x, y - 1)
        elif direction == Directions.EAST:
            return (x + 1, y)
        elif direction == Directions.WEST:
            return (x - 1, y)
        return position
    
    def chooseNextDirection(self, position, gameState):
        directions = [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]
        walls = gameState.getWalls()
        possibleDirections = []

        for direction in directions:
            nextPos = self.getNextPosition(position, direction)
            if not walls[nextPos[0]][nextPos[1]]:
                possibleDirections.append(direction)

        # Choose a direction that leads to open space, avoiding dead ends
        return random.choice(possibleDirections) if possibleDirections else Directions.STOP

    def setAmbush(self, gameState):
        ambushPoints = self.getAmbushPoints(gameState)
        myPos = gameState.getAgentState(self.index).getPosition()

        # Calculate scores for each ambush point
        ambushScores = {}
        for point in ambushPoints:
            score = self.calculateAmbushPointScore(point, gameState)
            distanceScore = -self.getMazeDistance(myPos, point)
            ambushScores[point] = score + distanceScore

        # Select the ambush point with the highest score
        targetPoint = max(ambushScores, key=ambushScores.get)
        return self.moveToPosition(gameState, targetPoint)

    def calculateAmbushPointScore(self, point, opponentIndices, gameState):
        score = 0
        for oppIndex in opponentIndices:
            oppState = gameState.getAgentState(oppIndex)
            if oppState.isPacman() and oppState.getPosition() is not None:
                # Higher score for points closer to Pacman opponents
                score += 10 - self.getMazeDistance(point, oppState.getPosition())

        # Add score for proximity to choke points
        if point in self.getChokePoints(gameState):
            score += 15

        # Add score for being near power capsules
        if point in self.getCapsulesYouAreDefending(gameState):
            score += 20

        # Consider escape routes from the ambush point
        escapeRoutes = self.getEscapeRoutes(point, gameState)
        score += len(escapeRoutes) * 5

        return score
    
    def selectStrategicAmbushPointConsideringPosition(self, ambushPoints, myPos, gameState):
        bestPoint = None
        bestScore = float('-inf')
        for point in ambushPoints:
            strategicScore = self.calculateAmbushScore(point, gameState)
            distanceScore = -self.getMazeDistance(myPos, point)  # Closer points are preferred
            totalScore = strategicScore + distanceScore

            if totalScore > bestScore:
                bestScore = totalScore
                bestPoint = point

        return bestPoint
    
    def selectStrategicAmbushPoint(self, ambushPoints, gameState):
        # Analyze opponent patterns, map layout, and power capsules
        opponentIndices = self.getOpponents(gameState)
        opponentPositions = [gameState.getAgentState(i).getPosition() for i in opponentIndices if gameState.getAgentState(i).getPosition() is not None]

        # Score each ambush point based on strategic value
        ambushScores = {}
        for point in ambushPoints:
            ambushScores[point] = self.calculateAmbushScore(point, opponentPositions, gameState)

        # Select the point with the highest score
        return max(ambushScores, key=ambushScores.get)

    def calculateAmbushScore(self, point, gameState):
        score = 0
        opponents = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        visibleOpponents = [opp for opp in opponents if opp.getPosition() is not None]

        # Score based on proximity to visible opponents
        for opp in visibleOpponents:
            distance = self.getMazeDistance(point, opp.getPosition())
            # Closer to opponents means higher score, but avoid too close
            if distance < 5:
                score += (5 - distance)

        # Consider proximity to power capsules with a higher weight
        capsules = self.getCapsulesYouAreDefending(gameState)
        if point in capsules:
            score += 20

        # Factor in choke points
        chokePoints = self.getChokePoints(gameState)
        if point in chokePoints:
            score += 15

        # Consider ease of escape
        escapeRoutes = self.getEscapeRoutes(point, gameState)
        score += len(escapeRoutes) * 5

        return score
    
    def getEscapeRoutes(self, point, gameState):
        escapeRoutes = []
        directions = [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]
        walls = gameState.getWalls()

        for direction in directions:
            nextPos = self.getNextPosition(point, direction)
            if not walls[nextPos[0]][nextPos[1]]:
                escapeRoutes.append(direction)

        return escapeRoutes

    def getAmbushPoints(self, gameState):
        # Points near power capsules or in narrow corridors
        powerCapsules = self.getCapsulesYouAreDefending(gameState)
        chokePoints = self.getChokePoints(gameState)
        return powerCapsules + chokePoints
    
    def getChokePoints(self, gameState):
        chokePoints = []
        walls = gameState.getWalls()
        for x in range(walls.width):
            for y in range(walls.height):
                if not walls[x][y]:
                    # A choke point is identified by having multiple adjacent walls
                    adjacentWalls = sum([walls[x + dx][y + dy] for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]])
                    if adjacentWalls >= 3:
                        chokePoints.append((x, y))

        return chokePoints

    def handleScaredState(self, gameState):
        safeAreas = self.getSafeAreas(gameState)
        myPos = gameState.getAgentState(self.index).getPosition()

        # Choose the nearest safe area with escape routes
        targetArea = min(safeAreas, key=lambda area: (self.getMazeDistance(myPos, area), -len(self.getEscapeRoutes(area, gameState))))
        return self.moveToPosition(gameState, targetArea)
    
    def findSafestAreaConsideringPosition(self, safeAreas, myPos, gameState):
        # Evaluate safe areas based on safety and proximity to the agent's current position
        bestArea = None
        bestScore = float('-inf')
        for area in safeAreas:
            safetyScore = self.calculateSafetyScore(area, gameState)
            distanceScore = -self.getMazeDistance(myPos, area)  # Closer areas are preferred
            totalScore = safetyScore + distanceScore

            if totalScore > bestScore:
                bestScore = totalScore
                bestArea = area

        return bestArea

    def findSafestArea(self, safeAreas, gameState):
        myPos = gameState.getAgentState(self.index).getPosition()
        teammateIndices = self.getTeam(gameState)
        teammatePositions = [gameState.getAgentState(i).getPosition() for i in teammateIndices if i != self.index]

        # Score each safe area based on safety criteria
        safetyScores = {}
        for area in safeAreas:
            safetyScores[area] = self.calculateSafetyScore(area, myPos, teammatePositions, gameState)

        # Select the area with the highest safety score
        return max(safetyScores, key=safetyScores.get)

    def calculateSafetyScore(self, area, myPos, teammatePositions, gameState):
        # Calculate the safety score of an area based on opponents' positions
        opponents = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        visibleOpponents = [opp.getPosition() for opp in opponents if opp.getPosition() is not None]

        score = 0
        for oppPos in visibleOpponents:
            score -= self.getMazeDistance(area, oppPos)  # Farther from opponents is better

        return score
    
    def getSafeAreas(self, gameState):
        safeAreas = []
        myPos = gameState.getAgentState(self.index).getPosition()
        opponents = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        visibleOpponents = [opp.getPosition() for opp in opponents if opp.getPosition() is not None]

        # Consider areas far from visible opponents as safe
        for x in range(gameState.getWalls().width):
            for y in range(gameState.getWalls().height):
                if not gameState.hasWall(x, y):
                    position = (x, y)
                    if all(self.getMazeDistance(position, oppPos) > 5 for oppPos in visibleOpponents):
                        # Factor in the distance from the agent's current position
                        distanceFromMyPos = self.getMazeDistance(myPos, position)
                        # Prefer areas closer to the agent but still safe
                        if distanceFromMyPos < 10:
                            safeAreas.append(position)

        return safeAreas

    def cooperateWithTeammate(self, gameState):
        teammateIndex = [i for i in self.getTeam(gameState) if i != self.index][0]
        teammatePos = gameState.getAgentState(teammateIndex).getPosition()
        myPos = gameState.getAgentState(self.index).getPosition()

        # Determine the division of the map based on both agents' positions
        if self.shouldPatrolLeftZone(myPos, teammatePos, gameState):
            return self.patrolLeftZone(gameState)
        else:
            return self.patrolRightZone(gameState)

    def shouldPatrolLeftZone(self, myPos, teammatePos, gameState):
        # Decide whether to patrol left or right zone based on positions
        mapWidth = gameState.getWalls().width
        if myPos[0] < mapWidth / 2 and (teammatePos is None or teammatePos[0] >= mapWidth / 2):
            return True
        elif teammatePos is not None and teammatePos[0] < mapWidth / 2 and myPos[0] >= mapWidth / 2:
            return False
        else:
            # If both agents are in the same half, choose based on proximity to the center
            return myPos[0] < teammatePos[0]
        
    def patrolLeftZone(self, gameState):
        foodList = self.getFoodYouAreDefending(gameState).asList()
        leftZoneFood = [food for food in foodList if food[0] < gameState.getWalls().width / 2]
        chokePoints = self.getChokePoints(gameState)
        leftZoneChokePoints = [point for point in chokePoints if point[0] < gameState.getWalls().width / 2]

        patrolPoints = leftZoneFood + leftZoneChokePoints
        myPos = gameState.getAgentState(self.index).getPosition()

        # Choose the patrol point based on strategic importance
        targetPoint = min(patrolPoints, key=lambda point: self.getMazeDistance(myPos, point))
        return self.moveToPosition(gameState, targetPoint)
    
    def patrolRightZone(self, gameState):
        foodList = self.getFoodYouAreDefending(gameState).asList()
        rightZoneFood = [food for food in foodList if food[0] >= gameState.getWalls().width / 2]
        chokePoints = self.getChokePoints(gameState)
        rightZoneChokePoints = [point for point in chokePoints if point[0] >= gameState.getWalls().width / 2]

        patrolPoints = rightZoneFood + rightZoneChokePoints
        myPos = gameState.getAgentState(self.index).getPosition()

        # Choose the patrol point based on strategic importance
        targetPoint = min(patrolPoints, key=lambda point: self.getMazeDistance(myPos, point))
        return self.moveToPosition(gameState, targetPoint)

    def shouldChaseOpponent(self, gameState):
        # Consider visibility of opponent but also prximity to territory and whether has food or not
        opponents = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        visibleOpponents = [opp for opp in opponents if opp.getPosition() is not None and opp.isPacman()]

        for opp in visibleOpponents:
            if self.isOpponentNearTerritory(opp, gameState):
                return True

        return False

    def isOpponentNearTerritory(self, opponent, gameState):
        myPos = gameState.getAgentState(self.index).getPosition()
        oppPos = opponent.getPosition()
        return self.getMazeDistance(myPos, oppPos) < 5

    def shouldSetAmbush(self, gameState):
        # Decide when to set an ambush
        if self.shouldChaseOpponent(gameState):
            return False

        myState = gameState.getAgentState(self.index)
        myPos = myState.getPosition()
        scared = myState.isScared()

        # Set an ambush if not scared and near a strategic point
        return not scared and self.isNearStrategicPoint(myPos, gameState)

    def isNearStrategicPoint(self, myPos, gameState):
        # Find whether or not near a strategic point for ambush function
        powerCapsules = self.getCapsulesYouAreDefending(gameState)
        chokePoints = self.getChokePoints(gameState)
        strategicPoints = powerCapsules + chokePoints

        for point in strategicPoints:
            if self.getMazeDistance(myPos, point) < 5:
                return True

        return False

def createTeam(firstIndex, secondIndex, isRed,
        first = 'pacai.agents.capture.dummy.DummyAgent',
        second = 'pacai.agents.capture.dummy.DummyAgent'):
    """
    This function should return a list of two agents that will form the capture team,
    initialized using firstIndex and secondIndex as their agent indexed.
    isRed is True if the red team is being created,
    and will be False if the blue team is being created.
    """

    return [
        OffensiveReflexAgent(firstIndex),
        DefensiveAgent(secondIndex),
    ]
