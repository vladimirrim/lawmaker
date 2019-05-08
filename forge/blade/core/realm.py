import pickle
from copy import deepcopy
from itertools import chain

import numpy as np
import ray

from forge.blade import entity, core


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
        self.logs = []
        self.currentAction = [0] * 8
        self.prevReward = 0
        self.curReward = np.zeros(config.NPOP)
        self.states = np.zeros((config.NPOP, 3))
        self.overallState = [0., 0., 0.]
        self.lengths = [0] * 8
        self.stepCount = 0
        #  self.lawmaker.load()

    def collectState(self):
        lengths = np.zeros(self.curReward.shape[0])
        states = np.zeros((8, 3))
        overallState = np.zeros(3)
        for ent in self.desciples.values():
            states[ent.annID][0] += self.sword.getUniqueGrass(ent.entID)
            states[ent.annID][1] += self.sword.getUniqueScrub(ent.entID)
            states[ent.annID][2] += ent.__getattribute__('timeAlive')

            overallState[0] += self.sword.getUniqueGrass(ent.entID)
            overallState[1] += self.sword.getUniqueScrub(ent.entID)
            overallState[2] += ent.__getattribute__('timeAlive')
            lengths[ent.annID] += 1

        if sum(lengths) != 0:
            self.overallState += overallState / sum(lengths)

        for i in range(len(lengths)):
            if lengths[i] != 0:
                self.states[i] += states[i] / lengths[i]

    def updateState(self):
        overallState = self.overallState
        state = self.states

        self.states = np.zeros((8, 3))
        self.overallState = [0., 0., 0.]
        self.lengths = [0] * 8

        return list(overallState) + list(state.reshape(24))

    def collectReward(self):
        reward = np.zeros(self.curReward.shape[0])
        lengths = np.zeros(self.curReward.shape[0])
        for ent in self.desciples.values():
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
        return np.sum(self.prevReward) / self.stepCount

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

    def stepEnts(self):
        dead = []

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
        self.collectReward()
        self.collectState()

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

    def run(self, currentAction, swordUpdate=None):
        self.recvSwordUpdate(swordUpdate)
        self.currentAction = currentAction

        updates = None
        self.stepCount = 0
        while updates is None:
            self.stepCount += 1
            self.step()
            updates, self.logs = self.sword.sendUpdate()
        return updates, self.logs, self.updateState(), self.updateReward()

    def recvSwordUpdate(self, update):
        if update is None:
            return
        self.sword.recvUpdate(update)

    def recvGodUpdate(self, update):
        self.god.recv(update)
