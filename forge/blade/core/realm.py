import pickle
from copy import deepcopy
from itertools import chain

import numpy as np
import ray

from forge import trinity as Trinity
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
        self.lawmaker = entity.Lawmaker()
        self.logs = []
        self.stepCount = 0
        self.currentAction = [0] * 8
        self.prevReward = 0
        self.curReward = 0

    def collectState(self):
        states = [[0., 0., 0.]] * 8
        lengths = [0] * 8

        for ent in self.desciples.values():
            states[ent.annID][0] += self.sword.getUniqueGrass(ent.entID)
            states[ent.annID][1] += self.sword.getUniqueGrass(ent.entID)
            states[ent.annID][2] += ent.__getattribute__('timeAlive')
            lengths[ent.annID] += 1

        state = [0]
        for i in range(8):
            for param in states[i]:
                if lengths[i] != 0:
                    state.append(param / lengths[i])
                else:
                    state.append(param)

        ans = np.array(state)
        ans.shape = (1, 25)
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
            self.stepLawmaker(self.collectState(), self.updateReward())

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


@ray.remote
class VecEnvRealm(Realm):
    # Use the default God behind the scenes for spawning
    def __init__(self, config, args, idx):
        super().__init__(config, args, idx)
        self.god = Trinity.God(config, args)

    def stepEnts(self, decisions):
        dead = []
        for tup in decisions:
            entID, action, arguments, val = tup
            ent = self.desciples[entID]
            ent.step(self.world)

            if self.postmortem(ent, dead):
                continue

            ent.act(self.world, action, arguments, val)
            self.stepEnt(ent, action, arguments)
        self.cullDead(dead)

    def postmortem(self, ent, dead):
        entID = ent.entID
        if not ent.alive or ent.kill:
            dead.append(entID)
            return True
        return False

    def step(self, decisions):
        decisions = pickle.loads(decisions)
        self.stepEnts(decisions)
        self.stepWorld()
        self.spawn()
        self.stepEnv()

        stims, rews, dones = [], [], []
        for entID, ent in self.desciples.items():
            stim = self.getStim(ent)
            stims.append((ent, self.getStim(ent)))
            rews.append(1)
        return pickle.dumps((stims, rews, None, None))

    def reset(self):
        self.spawn()
        self.stepEnv()
        return [(e, self.getStim(e)) for e in self.desciples.values()]
