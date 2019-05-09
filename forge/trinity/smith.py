import os
import pickle
import ray

from forge.blade import core, lib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
import numpy as np

# Wrapper for remote async multi environments (realms)
# Supports both the native and vecenv per-env api
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
        self.themis = Themis()
        self.trinity = trinity
        self.nPop = config.NPOP
        self.nRealm = args.nRealm
        self.stepCount = 0

        self.env = NativeServer(config, args, trinity)
        self.env.send(self.pantheon.model)
        self.expMaps = np.zeros((self.nRealm, self.nPop, 80, 80))

        self.renderStep = self.step
        self.idx = 0

    # Runs full trajectories on each environment
    # With no communication -- all on the env cores.
    def run(self):
        recvs, state, reward = self.env.run(self.themis.currentAction, self.pantheon.model)
        isNewEra = self.themis.voteForMax(reward)
        self.themis.stepLawmakerZero(state, reward)
        self.stepCount += 1
        if self.stepCount % 100 == 0:
            self.themis.save_model()
        _, logs = list(zip(*recvs))
        for i in range(len(logs)):
            for blob in logs[i]:
                self.expMaps[i][blob.annID] += blob.expMap
        if isNewEra:
            self.plotExpMaps()
            self.expMaps = np.zeros((self.nRealm, self.nPop, 80, 80))
        self.pantheon.step(recvs)
        self.rayBuffers()

    def plotExpMaps(self):
        for i in range(self.nRealm):
            for nation in range(self.nPop):
                dir = 'plots/era' + str(self.themis.era) + '/map' + str(i)
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

    # Only for render -- steps are run per core
    def step(self):
        self.env.step()

    # In early versions of ray, freeing memory was
    # an issue. It is possible this has been patched.
    def rayBuffers(self):
        self.idx += 1
        if self.idx % 32 == 0:
            lib.ray.clearbuffers()
