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
        self.idx = idx

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
        self.sword = trinity.sword(config, args, idx)
        self.sword.anns[0].world = self.world
        self.logs = []
        self.stepCount = 0

    def stepEnts(self):
        dead = []
        deadEnts = []

        for ent in self.desciples.values():
            ent.step(self.world)

            if not ent.alive or ent.kill:
                dead.append(ent.entID)
                deadEnts.append(ent)
                continue

            stim = self.getStim(ent)
            action, arguments, val = self.sword.decide(ent, stim) ###
            ent.act(self.world, action, arguments, val)

            self.stepEnt(ent, action, arguments)

        self.cullDead(dead)
        self.sword.lawmaker.collectRewards(-len(dead), self.desciples.keys())  ###

        for ent in deadEnts:
            if not self.config.TEST:
                self.sword.collectRollout(ent.entID, ent)

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
        self.stepCount = 0
        while updates is None:
            self.stepCount += 1
            self.step()
            if self.config.TEST:
                updates, updates_lm, logs = self.sword.sendLmLogUpdate(), [], []
            else:
                updates, updates_lm, logs = self.sword.sendUpdate()
        return updates, updates_lm, logs

    def recvSwordUpdate(self, update):
        if update is None:
            return
        self.sword.recvUpdate(update)

    def recvGodUpdate(self, update):
        self.god.recv(update)
