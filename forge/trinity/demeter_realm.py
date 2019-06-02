import numpy as np


class DemeterRealm:
    def __init__(self, sword, nPop):
        self.sword = sword
        self.prevReward = 0
        self.nPop = nPop
        self.featureSize = 4
        self.curReward = np.zeros(self.nPop)
        self.states = np.zeros((self.nPop, self.featureSize))
        self.overallState = np.zeros(self.featureSize)
        self.lengths = np.zeros(self.nPop)

    def collectState(self, desciples):
        lengths = np.zeros(self.curReward.shape[0])
        states = np.zeros((self.nPop, self.featureSize))
        overallState = np.zeros(self.featureSize)
        for ent in desciples.values():
            states[ent.annID][0] += self.sword.getUniqueGrass(ent.entID)
            states[ent.annID][1] += self.sword.getUniqueScrub(ent.entID)
            states[ent.annID][2] += self.sword.getCountGrass(ent.entID)
            states[ent.annID][3] += self.sword.getCountScrub(ent.entID)

            overallState[0] += self.sword.getUniqueGrass(ent.entID)
            overallState[1] += self.sword.getUniqueScrub(ent.entID)
            overallState[2] += self.sword.getCountGrass(ent.entID)
            overallState[3] += self.sword.getCountScrub(ent.entID)
            lengths[ent.annID] += 1

        if sum(lengths) != 0:
            self.overallState += overallState / sum(lengths)

        for i in range(len(lengths)):
            if lengths[i] != 0:
                self.states[i] += states[i] / lengths[i]

    def updateStates(self):
        state = self.states

        self.states = np.zeros((self.nPop, self.featureSize))
        self.overallState = np.zeros(self.featureSize)
        self.lengths = np.zeros(self.nPop)

        return state

    def collectReward(self, desciples):
        reward = np.zeros(self.curReward.shape[0])
        lengths = np.zeros(self.curReward.shape[0])
        for ent in desciples.values():
            reward[ent.annID] += ent.__getattribute__('timeAlive')
            lengths[ent.annID] += 1

        for i in range(len(reward)):
            if lengths[i] != 0:
                reward[i] /= lengths[i]

        self.curReward += reward

    def updateReward(self):
        #   r = (self.curReward - self.prevReward) / 1000
        self.prevReward = self.curReward
        self.curReward = np.zeros(8)
        return self.prevReward
