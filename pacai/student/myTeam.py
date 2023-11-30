from pacai.agents.capture.reflex import ReflexCaptureAgent

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
            foodScores = [(self.getMazeDistance(myPos, food) / (1 + features.get('ghostRisk', 0)), food) for food in foodList]
            bestFood = min(foodScores)[1]
            features['distanceToFood'] = self.getMazeDistance(myPos, bestFood)
        else:
            features['distanceToFood'] = 0

        # Risk Assessment from Ghosts
        ghostStates = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        ghosts = [ghost for ghost in ghostStates if not ghost.isPacman() and ghost.getPosition() is not None]
        features['ghostRisk'] = 0
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
        pos = gameState.getAgentPosition(self.index)
        scared = gameState.getAgentState(self.index).isScared()

        opponents = self.getOpponents(gameState)
        food_list = self.getFoodYouAreDefending(gameState).asList()

        target_pos = None
        min_dist = float("inf")
        for opponent in opponents:
            opponent_pos = gameState.getAgentPosition(opponent)
            if scared:
                dist = self.getMazeDistance(opponent_pos, pos)
                if dist < min_dist:
                    min_dist = dist
                    target_pos = opponent_pos
            else:
                for food in food_list:
                    dist = self.getMazeDistance(opponent_pos, food)
                    if dist < min_dist:
                        min_dist = dist
                        target_pos = food

        legal_actions = gameState.getLegalActions(self.index)
        min_dist = float("inf")
        best_action = None
        for action in legal_actions:
            successor = self.getSuccessor(gameState, action)
            successor_pos = successor.getAgentPosition(self.index)
            dist = self.getMazeDistance(successor_pos, target_pos)
            if dist < min_dist:
                min_dist = dist
                best_action = action

        return best_action

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
