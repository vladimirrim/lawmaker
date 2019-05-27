import os
import pickle
import ray

from forge.blade import core, lib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
import numpy as np
import seaborn as sns;

sns.set()

# Wrapper for remote async multi environments (realms)
# Supports both the native and vecenv per-env api
from forge.blade.entity.atalanta.atalanta import Atalanta
from forge.blade.lib.enums import Material
from forge.trinity.themis import Themis


class VecEnvServer:
    def __init__(self, config, args):
        self.envs = [core.VecEnvRealm.remote(config, args, i)
                     for i in range(args.nRealm)]

    # Reset the environments (only for vecenv api. This only returns
    # initial empty buffers to avoid special-case first iteration
    # code. Environments are persistent--attempting to reset them
    # will result in undefined behavior. Don't do it after setup.
    def reset(self):
        recvs = [e.reset.remote() for e in self.envs]
        return ray.get(recvs)

    def step(self, actions):
        recvs = ray.get([e.step.remote(pickle.dumps(a)) for e, a in
                         zip(self.envs, actions)])
        recvs = [pickle.loads(e) for e in recvs]
        return zip(*recvs)


class NativeServer:
    def __init__(self, config, args, trinity):
        self.envs = [core.NativeRealm.remote(trinity, config, args, i)
                     for i in range(args.nRealm)]

    def step(self, actions=None):
        recvs = [e.step.remote() for e in self.envs]
        return ray.get(recvs)

    # Use native api (runs full trajectories)
    def run(self, currentAction, swordUpdate=None):
        recvs = [e.run.remote(currentAction, swordUpdate) for e in self.envs]
        recvs = np.array(ray.get(recvs))
        return [(recvs[i][0], recvs[i][1]) for i in range(len(self.envs))], \
               [np.mean(x) for x in zip(*recvs[:, 2])], \
               np.mean([recvs[i][3] for i in range(len(self.envs))])

    def send(self, swordUpdate):
        [e.recvSwordUpdate.remote(swordUpdate) for e in self.envs]


# Example base runner class
class Blacksmith:
    def __init__(self, config, args):
        if args.render:
            print('Enabling local test mode for render')
            args.ray = 'local'
            args.nRealm = 1

        lib.ray.init(args.ray)

    def render(self):
        from forge.embyr.twistedserver import Application
        Application(self.env, self.renderStep)


# Example runner using the (slower) vecenv api
# The actual vecenv spec was not designed for
# multiagent, so this is a best-effort facsimile
class VecEnv(Blacksmith):
    def __init__(self, config, args, renderStep):
        super().__init__(config, args)
        self.env = VecEnvServer(config, args)
        self.renderStep = renderStep

    def step(self, actions):
        return self.env.step(actions)

    def reset(self):
        return self.env.reset()


# Example runner using the (faster) native api
# Use the /forge/trinity/ spec for model code
class Native(Blacksmith):
    def __init__(self, config, args, trinity):
        super().__init__(config, args)
        self.pantheon = trinity.pantheon(config, args)
        #  self.themis = Themis()
        self.trinity = trinity
        self.nPop = config.NPOP
        self.nRealm = args.nRealm
        self.featureSize = 4
        self.stepCount = 0
        self.avgReward = 0
        self.states = np.zeros((self.nPop, self.featureSize))
        self.avgState = np.zeros(self.featureSize * 9)
        self.period = 10000
        self.avgRewards = np.zeros(self.nPop)
        self.env = NativeServer(config, args, trinity)
        self.env.send(self.pantheon.model)
        self.expMaps = np.zeros((self.nRealm, self.nPop, 80, 80))
        self.atalanta = Atalanta(self.featureSize)
        self.atalanta.agent.load('checkpoints/atalanta')

        self.renderStep = self.step
        self.idx = 0

    # Runs full trajectories on each environment
    # With no communication -- all on the env cores.
    def run(self):
        self.stepCount += 1
        self.atalanta.agent.batch_act_and_train(list(self.states.astype('float32')))
        dist = self.atalanta.agent.act_prob(self.states.astype('float32'))
        recvs, _, _ = self.env.run(dist, self.pantheon.model)
        _, logs = list(zip(*recvs))
        state = self.collectState(logs)
        self.states += self.collectStates(logs)
        reward, rewards = self.collectReward(logs)
        self.avgReward += reward
        self.avgRewards += rewards
        self.avgState += state

        for i in range(len(logs)):
            for blob in logs[i]:
                self.expMaps[i][blob.annID] += blob.expMap

        # if self.stepCount % self.period == 0:
        #      self.updateModel()

        #     if self.stepCount % 1 == 0:
        #        self.plotDistribution(dist)

        #   if self.stepCount % 100 == 0:
        #      self.atalanta.agent.save('checkpoints/atalanta')

        self.pantheon.step(recvs)
        self.rayBuffers()

    def updateModel(self):
        #     isNewEra = self.themis.voteForBest(self.avgRewards)
        print(self.avgRewards)
        # self.themis.stepLawmakerZero(list(self.avgState), self.avgReward, self.avgRewards)
        self.atalanta.agent.batch_observe_and_train(list(self.states.astype('float32')),
                                                    list(self.avgRewards.astype('float32')), [False] * 8, [False] * 8)
        self.avgReward = 0
        self.states = np.zeros((self.nPop, self.featureSize))
        self.avgRewards = np.zeros(self.nPop)
        self.avgState = np.zeros(self.featureSize * 9)

    #  if isNewEra:
    #     self.plotExpMaps()
    #    self.expMaps = np.zeros((self.nRealm, self.nPop, 80, 80))

    def plotExpMaps(self):
        for i in range(self.nRealm):
            for nation in range(self.nPop):
                dir = 'plots/era' + str(1) + '/map' + str(i)
                try:
                    os.makedirs(dir)
                except FileExistsError:
                    pass

                self.plotExpMap(self.expMaps[i][nation], dir + '/nation' + str(nation) + '.png')

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

    # Only for render -- steps are run per core
    def step(self):
        self.env.step()

    # In early versions of ray, freeing memory was
    # an issue. It is possible this has been patched.
    def rayBuffers(self):
        self.idx += 1
        if self.idx % 32 == 0:
            lib.ray.clearbuffers()
