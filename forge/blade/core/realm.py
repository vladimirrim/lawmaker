import pickle
import random
from copy import deepcopy
from itertools import chain

import numpy as np
import ray
import tensorflow as tf

from forge.blade import entity, core
from forge.blade.entity.lawmaker_zero import LawmakerZero
from forge.blade.entity.lawmaker_zero.config import get_config


class ActionArgs:
    def __init__(self, action, args):
        self.action = action
        self.args = args


class Realm:
    def __init__(self, config, args, idx):
        # Random samples
        if config.SAMPLE:
            config = deepcopy(config)
            nent = np.random.randint(0, config.NENT)
            config.NENT = config.NPOP * (1 + nent // config.NPOP)
        self.world, self.desciples = core.Env(config, idx), {}
        self.config, self.args, self.tick = config, args, 0
        self.npop = config.NPOP

        self.env = self.world.env
        self.values = None

    def clientData(self):
        if self.values is None and hasattr(self, 'sword'):
            self.values = self.sword.anns[0].visVals()

        ret = {
            'environment': self.world.env,
            'entities': dict((k, v.packet()) for k, v in self.desciples.items()),
            'values': self.values
        }
        return pickle.dumps(ret)

    def spawn(self):
        if len(self.desciples) >= self.config.NENT:
            return

        entID, color = self.god.spawn()
        ent = entity.Player(entID, color, self.config)
        self.desciples[ent.entID] = ent

        r, c = ent.pos
        self.world.env.tiles[r, c].addEnt(entID, ent)
        self.world.env.tiles[r, c].counts[ent.colorInd] += 1

    def cullDead(self, dead):
        for entID in dead:
            ent = self.desciples[entID]
            r, c = ent.pos
            self.world.env.tiles[r, c].delEnt(entID)
            self.god.cull(ent.annID)
            del self.desciples[entID]

    def stepWorld(self):
        ents = list(chain(self.desciples.values()))
        self.world.step(ents, [])

    def stepEnv(self):
        self.world.env.step()
        self.env = self.world.env.np()

    def stepEnt(self, ent, action, arguments):
        move, attack = action
        moveArgs, attackArgs = arguments

        ent.move = ActionArgs(move, moveArgs)
        ent.attack = ActionArgs(attack, attackArgs[0])

    def getStim(self, ent):
        return self.world.env.stim(ent.pos, self.config.STIM)


@ray.remote
class NativeRealm(Realm):
    def __init__(self, trinity, config, args, idx):
        super().__init__(config, args, idx)
        self.god = trinity.god(config, args)
        self.sword = trinity.sword(config, args)
        self.sword.anns[0].world = self.world
        self.lawmaker = entity.Lawmaker()
        self.logs = []
        self.stepCount = 0
        self.currentAction = [0] * 8
        self.prevReward = 0
        self.curReward = 0
        self.setupLawmakerZero()

    def collectState(self):
        states = [[0., 0., 0.]] * 8
        lengths = [0] * 8

        for ent in self.desciples.values():
            states[ent.annID][0] += self.sword.getUniqueGrass(ent.entID)
            states[ent.annID][1] += self.sword.getUniqueGrass(ent.entID)
            states[ent.annID][2] += ent.__getattribute__('timeAlive')
            lengths[ent.annID] += 1

        state = []
        for i in range(8):
            for param in states[i]:
                if lengths[i] != 0:
                    state.append(param / lengths[i])
                else:
                    state.append(param)

        ans = np.array(state)
        ans.shape = (1, 24)
        return ans

    def collectReward(self):
        reward = 0
        for ent in self.desciples.values():
            reward += ent.__getattribute__('timeAlive')
        self.curReward += reward / len(self.desciples.values())

    def updateReward(self):
        r = (self.curReward - self.prevReward) / 1000
        self.prevReward = self.curReward
        self.curReward = 0
        return r

    def stepLawmaker(self, state, reward):
        if self.stepCount != 0:
            for i in range(8):
                state[0] = i + 1
                self.lawmaker.updateLaw(state, reward)

        for i in range(8):
            state[0] = i + 1
            self.currentAction[i] = self.lawmaker.step(state)
        self.lawmaker.save()
        self.save()

    def stepLawmakerZero(self, state, reward):
        if self.stepCount == 0:
            self.lawmakerZero.initStep(np.repeat(state, 24, axis=0), reward, np.random.randint(0, 10), True)
        else:
            self.lawmakerZero.updateModel(np.repeat(state, 24, axis=0), reward, True)
            for i in range(8):
                self.currentAction[i] = self.lawmakerZero.stepEnv()
        self.save()

    def save(self):
        ROOT = 'resource/exps/laws/'
        with open(ROOT + 'actions.txt', 'a') as f:
            for action in self.currentAction:
                f.write("%s " % action)
            f.write('\n')

    def stepEnts(self):
        dead = []
        self.collectReward()

        if self.stepCount % 1000 == 0:
            self.stepLawmakerZero(self.collectState(), self.updateReward())

        for ent in self.desciples.values():
            ent.step(self.world)

            ent.applyDamage(self.currentAction[ent.annID])

            if self.postmortem(ent, dead):
                continue

            stim = self.getStim(ent)
            action, arguments, val = self.sword.decide(ent, stim)
            ent.act(self.world, action, arguments, val)

            self.stepEnt(ent, action, arguments)

        self.cullDead(dead)

    def postmortem(self, ent, dead):
        entID = ent.entID
        if not ent.alive or ent.kill:
            dead.append(entID)
            if not self.config.TEST:
                self.sword.collectRollout(entID, ent)
            return True
        return False

    def step(self):
        self.spawn()
        self.stepEnv()
        self.stepEnts()
        self.stepWorld()

    def run(self, swordUpdate=None):
        self.recvSwordUpdate(swordUpdate)

        updates = None
        while updates is None:
            self.step()
            self.stepCount += 1
            updates, self.logs = self.sword.sendUpdate()
        return updates, self.logs

    def recvSwordUpdate(self, update):
        if update is None:
            return
        self.sword.recvUpdate(update)

    def recvGodUpdate(self, update):
        self.god.recv(update)

    def setupLawmakerZero(self):
        flags = dict()

        # Model
        flags['model'] = 'm1'
        flags['dueling'] = False
        flags['double_q'] = False

        # Etc
        flags['use_gpu'] = False
        flags['gpu_fraction'] = '1/1'
        flags['display'] = False
        flags['is_train'] = True
        flags['random_seed'] = 123
        random_seed = 1337

        # Set random seed
        tf.set_random_seed(random_seed)
        random.seed(random_seed)

        def calc_gpu_fraction(fraction_string):
            idx, num = fraction_string.split('/')
            idx, num = float(idx), float(num)

            fraction = 1 / (num - idx + 1)
            print(" [*] GPU : %.4f" % fraction)
            return fraction

        gpu_options = tf.GPUOptions(
            per_process_gpu_memory_fraction=calc_gpu_fraction(flags['gpu_fraction']))

        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            config = get_config(flags) or flags

            if not flags['use_gpu']:
                config.cnn_format = 'NHWC'

            self.lawmakerZero = LawmakerZero(config, sess)
