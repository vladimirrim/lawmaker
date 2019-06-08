from collections import defaultdict

from forge import trinity
from forge.blade.lib.enums import Material
from forge.ethyr.rollouts import Rollout, mergeRollouts
from forge.ethyr.torch import optim
from forge.ethyr.torch.param import setParameters, zeroGrads
import torch
from forge.trinity.ann import Lawmaker


class Sword:
    def __init__(self, config, args, idx):
        self.config, self.args = config, args
        self.nANN, self.h = config.NPOP, config.HIDDEN
        self.anns = [trinity.ANN(config)
                     for i in range(self.nANN)]

        self.init, self.nRollouts = True, 32
        self.networksUsed = set()
        self.updates, self.rollouts = defaultdict(Rollout), {}
        self.ents, self.rewards, self.grads = {}, [], None
        self.nGrads = 0
        self.idx = idx

        self.lawmaker = Lawmaker(args, config)

    def backward(self):
        ents = self.rollouts.keys()
        anns = [self.anns[idx] for idx in self.networksUsed]
        atns, vals, rets = mergeRollouts(self.rollouts.values())
        reward, val, grads, pg, valLoss, entropy = optim.backward(
            self.rollouts, anns, valWeight=0.25,
            entWeight=self.config.ENTROPY)
        self.grads = dict((idx, grad) for idx, grad in
                          zip(self.networksUsed, grads))

        self.blobs = [r.feather.blob for r in self.rollouts.values()]
        self.rollouts = {}
        self.nGrads = 0
        self.networksUsed = set()

        self.lawmaker.backward(entWeight=0.05)

    def sendGradUpdate(self):
        grads = self.grads
        grads_lm = self.lawmaker.grads
        self.grads = None
        self.lawmaker.grads = None
        return grads, grads_lm

    def sendLogUpdate(self):
        blobs = self.blobs
        self.blobs = []
        return blobs

    def getExploredTiles(self, entID):
        blob = self.updates[entID].feather.blob
        return blob.unique[Material.GRASS.value] + blob.unique[Material.SCRUB.value]

    def getUniqueGrass(self, entID):
        blob = self.updates[entID].feather.blob
        return blob.unique[Material.GRASS.value]

    def getCountGrass(self, entID):
        blob = self.updates[entID].feather.blob
        return blob.counts[Material.GRASS.value]

    def getUniqueScrub(self, entID):
        blob = self.updates[entID].feather.blob
        return blob.unique[Material.SCRUB.value]

    def getCountScrub(self, entID):
        blob = self.updates[entID].feather.blob
        return blob.counts[Material.SCRUB.value]

    def sendUpdate(self):
        if self.grads is None:
            return None, None, None
        recvs, recvs_lm = self.sendGradUpdate()
        return recvs, recvs_lm, self.sendLogUpdate()

    def recvUpdate(self, update):
        update, update_lm = update
        for idx, paramVec in enumerate(update):
            setParameters(self.anns[idx], paramVec)
            zeroGrads(self.anns[idx])

        ### update lawmaker
        setParameters(self.lawmaker, update_lm)
        zeroGrads(self.lawmaker)

    def collectStep(self, entID, atnArgs, val, reward):
        if self.config.TEST:
            return
        self.updates[entID].step(atnArgs, val, reward)

    def collectRollout(self, entID, ent):
        assert entID not in self.rollouts
        rollout = self.updates[entID]
        rollout.finish()
        self.nGrads += rollout.lifespan
        self.rollouts[entID] = rollout
        del self.updates[entID]

        # assert ent.annID == (hash(entID) % self.nANN)
        self.networksUsed.add(ent.annID)

        # Two options: fixed number of gradients or rollouts
        # if len(self.rollouts) >= self.nRollouts:
        if self.nGrads >= 10 * 32:
            self.backward()

    def decide(self, ent, stim):
        reward, entID, annID = 1, ent.entID, ent.annID  ###
        action, arguments, atnArgs, val = self.anns[annID](ent, stim)

        ### subtract reward with lawmaker here
        policy = torch.tensor(atnArgs[0][0].tolist(), requires_grad=True)
        punishment, val_lawmaker = self.lawmaker(ent, stim, policy)
        reward -= float(punishment)

        self.collectStep(entID, atnArgs, val, reward)
        self.updates[entID].feather.scrawl(
            stim, ent, val, reward)
        return action, arguments, float(val)
