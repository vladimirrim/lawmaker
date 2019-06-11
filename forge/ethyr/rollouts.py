from pdb import set_trace as T
from itertools import chain
import numpy as np

from forge.blade.lib.log import Blob


# Untested function
def discountRewards(rewards, gamma=0.99):
    rets, N = [], len(rewards)
    discounts = np.array([gamma ** i for i in range(N)])
    rewards = np.array(rewards)
    for idx in range(N): rets.append(sum(rewards[idx:] * discounts[:N - idx]))
    return rets


def sumReturn(rewards):
    return [sum(rewards) for e in rewards]


def mergeRollouts(rollouts):
    atnArgs = [rollout.atnArgs for rollout in rollouts]
    vals = [rollout.vals for rollout in rollouts]
    rets = [rollout.returns for rollout in rollouts]

    atnArgs = list(chain(*atnArgs))
    atnArgs = list(zip(*atnArgs))
    vals = list(chain(*vals))
    rets = list(chain(*rets))

    return atnArgs, vals, rets


class Rollout:
    def __init__(self, returnf=discountRewards):
        self.atnArgs = []
        self.vals = []
        self.rewards = []
        self.pop_rewards = []
        self.returnf = returnf
        self.feather = Feather()

    def step(self, atnArgs, val, reward):
        self.atnArgs.append(atnArgs)
        self.vals.append(val)
        self.rewards.append(reward)

    def finish(self):
        self.rewards[-1] = -10  ###
        self.returns = self.returnf(self.rewards)
        self.lifespan = len(self.rewards)
        self.feather.finish()


# Rollout logger
class Feather:
    def __init__(self):
        self.expMap = set()
        self.blob = Blob()

    def scrawl(self, stim, ent, val, reward, lmVal, lmPunishment):
        self.blob.annID = ent.annID
        tile = self.tile(stim)
        self.move(tile, ent.pos)
        # self.action(arguments, atnArgs)
        self.stats(val, reward, lmVal, lmPunishment)

    def tile(self, stim):
        R, C = stim.shape
        rCent, cCent = R // 2, C // 2
        tile = stim[rCent, cCent]
        return tile

    def action(self, arguments, atnArgs):
        move, attk = arguments
        moveArgs, attkArgs, _ = atnArgs
        moveLogits, moveIdx = moveArgs
        attkLogits, attkIdx = attkArgs

    def move(self, tile, pos):
        tile = type(tile.state)
        r, c = pos
        self.blob.expMap[r][c] += 1
        if pos not in self.expMap:
            self.expMap.add(pos)
            self.blob.unique[tile] += 1
        self.blob.counts[tile] += 1

    def stats(self, value, reward, lmVal, lmPunishment):
        self.blob.reward.append(reward)
        self.blob.value.append(float(value))
        self.blob.lmValue.append(float(lmVal))
        self.blob.lmPunishment.append(float(lmPunishment))

    def finish(self):
        self.blob.finish()
