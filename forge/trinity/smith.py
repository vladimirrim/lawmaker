# import numpy as np
import ray

from forge.blade import core, lib
from forge.trinity.ann import Lawmaker
# from forge.blade.entity.lawmaker.atalanta.atalanta import Atalanta

# Wrapper for remote async multi environments (realms)
# from forge.trinity.demeter import Demeter


class NativeServer:
    def __init__(self, config, args, trinity):
        self.lawmaker = Lawmaker(args, config)  ###
        self.envs = [core.NativeRealm.remote(trinity, config, args, i, self.lawmaker)  ###
                     for i in range(args.nRealm)]

    def step(self, actions=None):
        recvs = [e.step.remote() for e in self.envs]
        return ray.get(recvs)

    # Use native api (runs full trajectories)
    def run(self, swordUpdate=None):
        recvs = [e.run.remote(swordUpdate) for e in self.envs]
        recvs = ray.get(recvs)
        return recvs
        # recvs = np.array(ray.get(recvs))
        # return [(recvs[i][0], recvs[i][1]) for i in range(len(self.envs))],\
              # np.mean(recvs[:, 2]), np.mean(recvs[:, 3])

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


# Example runner using the (faster) native api
# Use the /forge/trinity/ spec for model code
class Native(Blacksmith):
    def __init__(self, config, args, trinity):
        super().__init__(config, args)
        self.pantheon = trinity.pantheon(config, args)
        self.trinity = trinity
        self.stepCount = 0
        self.period = 1

        self.env = NativeServer(config, args, trinity)
        self.env.send(self.pantheon.model)

        #self.statsCollector = Demeter(config.NPOP, args.nRealm)
        #self.lawmaker = Atalanta(self.statsCollector.featureSize)
        # self.lawmaker.load('checkpoints/atalanta')

        self.renderStep = self.step
        self.idx = 0

    # Runs full trajectories on each environment
    # With no communication -- all on the env cores.
    def run(self):
        self.stepCount += 1
        recvs = self.env.run(self.pantheon.model)  # , states, rewards

        ### sometimes update lawmaker here
        if self.env.lawmaker.count > self.env.lawmaker.update_period:
            self.env.lawmaker.backward()

        self.pantheon.step(recvs)
        self.rayBuffers()

    # def updateModel(self):
    #     print(self.statsCollector.avgRewards)
    #     self.lawmaker.batch_observe_and_train(list(self.statsCollector.states.astype('float32')),
    #                                           list(self.statsCollector.avgRewards.astype('float32')),
    #                                           [False] * 8, [False] * 8)
    #     self.statsCollector.resetStatistics()

    # Only for render -- steps are run per core
    def step(self):
        self.env.step()
        self.stepCount += 1

        ### sometimes update lawmaker here
        if self.env.lawmaker.count > self.env.lawmaker.update_period:
            self.env.lawmaker.backward()

    # In early versions of ray, freeing memory was
    # an issue. It is possible this has been patched.
    def rayBuffers(self):
        self.idx += 1
        if self.idx % 32 == 0:
            lib.ray.clearbuffers()
