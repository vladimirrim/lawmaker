import os

import numpy as np
from matplotlib import cm
from matplotlib.colors import LogNorm

import seaborn as sns; sns.set()
from forge.blade.lib.enums import Material
import matplotlib.pyplot as plt


# Statistic Collector
class Demeter:
    def __init__(self, nPop, nRealm):
        self.nPop = nPop
        self.nRealm = nRealm
        self.featureSize = 4
        self.expMaps = np.zeros((self.nRealm, self.nPop, 80, 80))
        self.avgReward = 0
        self.states = np.zeros((self.nPop, self.featureSize))
        self.avgState = np.zeros(self.featureSize * 9)
        self.period = 10000
        self.avgRewards = np.zeros(self.nPop)

    def collectState(self, logs):
        state = np.zeros((self.nRealm, self.nPop, self.featureSize))
        overallState = np.zeros(self.featureSize)
        lengths = np.zeros((self.nRealm, self.nPop))
        length = 0
        for i in range(len(logs)):
            for blob in logs[i]:
                state[i][blob.annID][0] += blob.unique[Material.GRASS.value]
                state[i][blob.annID][1] += blob.counts[Material.GRASS.value]
                state[i][blob.annID][2] += blob.unique[Material.SCRUB.value]
                state[i][blob.annID][3] += blob.counts[Material.SCRUB.value]
                overallState += state[i][blob.annID]
                length += 1
                lengths[i][blob.annID] += 1
        states = np.zeros(self.nPop * self.featureSize)
        for i in range(len(logs)):
            for nn in range(self.nPop):
                if lengths[i][nn] != 0:
                    for j in range(self.featureSize):
                        states[nn * self.featureSize + j] += state[i][nn][j] / lengths[i][nn]

        if length != 0:
            overallState /= length

        return list(overallState / 8) + list(states / 8)

    def collectStates(self, logs):
        states = np.zeros((self.nPop, self.featureSize))
        overallState = np.zeros(self.featureSize)
        lengths = np.zeros(self.nPop)
        length = 0
        for i in range(len(logs)):
            for blob in logs[i]:
                states[blob.annID][0] += blob.unique[Material.GRASS.value]
                states[blob.annID][1] += blob.counts[Material.GRASS.value]
                states[blob.annID][2] += blob.unique[Material.SCRUB.value]
                states[blob.annID][3] += blob.counts[Material.SCRUB.value]
                overallState += states[blob.annID]
                length += 1
                lengths[blob.annID] += 1
        for nn in range(self.nPop):
            if lengths[nn] != 0:
                states[nn] /= lengths[nn]

        return states

    def updateExpMaps(self, logs):
        for i in range(len(logs)):
            for blob in logs[i]:
                self.expMaps[i][blob.annID] += blob.expMap

    def plotExpMaps(self):
        for i in range(self.nRealm):
            for nation in range(self.nPop):
                dir = 'plots/era' + str(1) + '/map' + str(i)
                try:
                    os.makedirs(dir)
                except FileExistsError:
                    pass

                self.plotExpMap(self.expMaps[i][nation], dir + '/nation' + str(nation) + '.png')

    def resetStatistics(self):
        self.avgReward = 0
        self.states = np.zeros((self.nPop, self.featureSize))
        self.avgRewards = np.zeros(self.nPop)
        self.avgState = np.zeros(self.featureSize * 9)


    def plotExpMap(self, expMap, title):
        plt.imshow(expMap + 10, cmap=cm.hot, norm=LogNorm())
        plt.colorbar()
        plt.savefig(title)
        plt.close()

    def collectReward(self, logs):
        reward = np.zeros((self.nRealm, self.nPop))
        rewards = np.zeros(self.nPop)
        lengths = np.zeros((self.nRealm, self.nPop))
        totalReward = 0
        for i in range(len(logs)):
            for blob in logs[i]:
                reward[i][blob.annID] += blob.lifetime
                lengths[i][blob.annID] += 1

        for i in range(len(logs)):
            for nn in range(self.nPop):
                if lengths[i][nn] == 0:
                    return 0, np.zeros(self.nPop)
                rewards[nn] += reward[i][nn] / lengths[i][nn]
                totalReward += reward[i][nn] / lengths[i][nn]
        return totalReward / self.nPop / self.nRealm, rewards / self.nRealm

    def plotDistribution(self, dist):
        for i in range(self.nPop):
            points = [dist.sample().data[i] for _ in range(int(1e5))]
            sns.distplot(points, hist=False, kde=True,
                         kde_kws={'shade': True, 'linewidth': 3})
            plt.title('Reward Distribution')
            plt.xlabel('Reward')
            plt.ylabel('Density')
            plt.savefig('plots/distribution' + (str(i)) + '.png')
            plt.close()
