from pdb import set_trace as T
import numpy as np
import torch

from torch import nn
from torch.nn import functional as F
from torch.distributions import Categorical
from forge.ethyr.torch import param

from forge.blade.action.tree import ActionTree
from forge.blade.action.v2 import ActionV2
from forge.blade.lib.enums import Neon
from forge.blade.lib import enums
from forge.ethyr import torch as torchlib
from forge.blade import entity

from forge.ethyr.torch import loss
from forge.ethyr.rollouts import discountRewards


def classify(logits):
    if len(logits.shape) == 1:
        logits = logits.view(1, -1)
    distribution = Categorical(1e-3 + F.softmax(logits, dim=1))
    atn = distribution.sample()
    return atn


####### Network Modules
class ConstDiscrete(nn.Module):
    def __init__(self, config, h, nattn):
        super().__init__()
        self.fc1 = torch.nn.Linear(h, nattn)
        self.config = config

    def forward(self, env, ent, action, stim):
        leaves = action.args(env, ent, self.config)
        x = self.fc1(stim)
        xIdx = classify(x)
        leaf = leaves[int(xIdx)]
        return leaf, x, xIdx


class VariableDiscrete(nn.Module):
    def __init__(self, config, xdim, h):
        super().__init__()
        self.attn = AttnCat(xdim, h)
        self.config = config

    # Arguments: stim, action/argument embedding
    def forward(self, env, ent, action, key, vals):
        leaves = action.args(env, ent, self.config)
        x = self.attn(key, vals)
        xIdx = classify(x)
        leaf = leaves[int(xIdx)]
        return leaf, x, xIdx


class AttnCat(nn.Module):
    def __init__(self, xdim, h):
        super().__init__()
        # self.fc1 = torch.nn.Linear(xdim, h)
        # self.fc2 = torch.nn.Linear(h, 1)
        self.fc = torch.nn.Linear(xdim, 1)
        self.h = h

    def forward(self, x, args):
        n = args.shape[0]
        x = x.expand(n, self.h)
        xargs = torch.cat((x, args), dim=1)

        x = self.fc(xargs)
        # x = F.relu(self.fc1(xargs))
        # x = self.fc2(x)
        return x.view(1, -1)


####### End network modules

class ValNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc = torch.nn.Linear(config.HIDDEN, 1)
        self.envNet = Env(config)

    def forward(self, conv, flat, ent):
        stim = self.envNet(conv, flat, ent)
        x = self.fc(stim)
        x = x.view(1, -1)
        return x


class Ent(nn.Module):
    def __init__(self, entDim, h):
        super().__init__()
        self.ent = torch.nn.Linear(entDim, h)

    def forward(self, ents):
        ents = self.ent(ents)
        ents, _ = torch.max(ents, 0)
        return ents


class Env(nn.Module):
    def __init__(self, config):
        super().__init__()
        h = config.HIDDEN
        entDim = 11  # + 225

        self.fc1 = torch.nn.Linear(3 * h, h)
        self.embed = torch.nn.Embedding(7, 7)

        self.conv = torch.nn.Linear(1800, h)
        self.flat = torch.nn.Linear(entDim, h)
        self.ents = Ent(entDim, h)

    def forward(self, conv, flat, ents):
        tiles, nents = conv[0], conv[1]
        nents = nents.view(-1)

        tiles = self.embed(tiles.view(-1).long()).view(-1)
        conv = torch.cat((tiles, nents))

        conv = self.conv(conv)
        ents = self.ents(ents)
        flat = self.flat(flat)

        x = torch.cat((conv, flat, ents)).view(1, -1)
        x = self.fc1(x)
        # Removed relu (easier training, lower policy cap)
        # x = torch.nn.functional.relu(self.fc1(x))
        return x


class MoveNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.moveNet = ConstDiscrete(config, config.HIDDEN, 5)
        self.envNet = Env(config)

    def forward(self, env, ent, action, s):
        stim = self.envNet(s.conv, s.flat, s.ents)
        action, arg, argIdx = self.moveNet(env, ent, action, stim)
        return action, (arg, argIdx)


# Network that selects an attack style
class StyleAttackNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config, h = config, config.HIDDEN
        self.h = h

        self.envNet = Env(config)
        self.targNet = ConstDiscrete(config, h, 3)

    def target(self, ent, arguments):
        if len(arguments) == 1:
            return arguments[0]
        arguments = [e for e in arguments if e.entID != ent.entID]
        arguments = sorted(arguments, key=lambda a: a.health.val)
        return arguments[0]

    def forward(self, env, ent, action, s):
        stim = self.envNet(s.conv, s.flat, s.ents)
        action, atn, atnIdx = self.targNet(env, ent, action, stim)

        # Hardcoded targeting
        arguments = action.args(env, ent, self.config)
        argument = self.target(ent, arguments)

        attkOuts = [(atn, atnIdx)]
        return action, [argument], attkOuts


# Network that selects an attack and target (In progress,
# for learned targeting experiments)
class AttackNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config, h = config, config.HIDDEN
        entDim = 11
        self.styleEmbed = torch.nn.Embedding(3, h)
        self.targEmbed = Ent(entDim, h)
        self.h = h

        self.envNet = Env(config)
        self.styleNet = ConstDiscrete(config, h, 3)
        self.targNet = VariableDiscrete(config, 3 * h, h)

    def forward(self, env, ent, action, s):
        stim = self.envNet(s.conv, s.flat, s.ents)
        action, atn, atnIdx = self.styleNet(env, ent, action, stim)

        # Embed targets
        targets = action.args(env, ent, self.config)
        targets = torch.tensor([e.stim for e in targets]).float()
        targets = self.targEmbed(targets).unsqueeze(0)
        nTargs = len(targets)

        atns = self.styleEmbed(atnIdx).expand(nTargs, self.h)
        vals = torch.cat((atns, targets), 1)

        argument, arg, argIdx = self.targNet(
            env, ent, action, stim, vals)

        attkOuts = ((atn, atnIdx), (arg, argIdx))
        return action, [argument], attkOuts


class ANN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.valNet = ValNet(config)

        self.config = config
        self.moveNet = MoveNet(config)
        self.attackNet = (StyleAttackNet(config) if
                          config.AUTO_TARGET else AttackNet(config))

    def forward(self, ent, env):
        s = torchlib.Stim(ent, env, self.config)
        val = self.valNet(s.conv, s.flat, s.ents)

        actions = ActionTree(env, ent, ActionV2).actions()
        _, move, attk = actions

        # Actions
        moveArg, moveOuts = self.moveNet(
            env, ent, move, s)
        attk, attkArg, attkOuts = self.attackNet(
            env, ent, attk, s)

        action = (move, attk)
        arguments = (moveArg, attkArg)
        outs = (moveOuts, *attkOuts)
        return action, arguments, outs, val

    # Messy hooks for visualizers
    def visDeps(self):
        from forge.blade.core import realm
        from forge.blade.core.tile import Tile
        colorInd = int(12 * np.random.rand())
        color = Neon.color12()[colorInd]
        color = (colorInd, color)
        ent = realm.Desciple(-1, self.config, color).server
        targ = realm.Desciple(-1, self.config, color).server

        sz = 15
        tiles = np.zeros((sz, sz), dtype=object)
        for r in range(sz):
            for c in range(sz):
                tiles[r, c] = Tile(enums.Grass, r, c, 1, None)

        targ.pos = (7, 7)
        tiles[7, 7].addEnt(0, targ)
        posList, vals = [], []
        for r in range(sz):
            for c in range(sz):
                ent.pos = (r, c)
                tiles[r, c].addEnt(1, ent)
                s = torchlib.Stim(ent, tiles, self.config)
                conv, flat, ents = s.conv, s.flat, s.ents
                val = self.valNet(conv, s.flat, s.ents)
                vals.append(float(val))
                tiles[r, c].delEnt(1)
                posList.append((r, c))
        vals = list(zip(posList, vals))
        return vals

    def visVals(self, food='max', water='max'):
        posList, vals = [], []
        R, C = self.world.shape
        for r in range(self.config.BORDER, R - self.config.BORDER):
            for c in range(self.config.BORDER, C - self.config.BORDER):
                colorInd = int(12 * np.random.rand())
                color = Neon.color12()[colorInd]
                color = (colorInd, color)
                ent = entity.Player(-1, color, self.config)
                ent._pos = (r, c)

                if food != 'max':
                    ent._food = food
                if water != 'max':
                    ent._water = water
                posList.append(ent.pos)

                self.world.env.tiles[r, c].addEnt(ent.entID, ent)
                stim = self.world.env.stim(ent.pos, self.config.STIM)
                s = torchlib.Stim(ent, stim, self.config)
                val = self.valNet(s.conv, s.flat, s.ents).detach()
                self.world.env.tiles[r, c].delEnt(ent.entID)
                vals.append(float(val))

        vals = list(zip(posList, vals))
        return vals


class PunishNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.repr = torch.nn.Linear(config.HIDDEN + 5, config.HIDDEN)
        self.relu = torch.nn.ReLU()
        self.fc = torch.nn.Linear(config.HIDDEN, 1)
        self.activation = torch.nn.Sigmoid()
        self.envNet = Env(config)

    def forward(self, conv, flat, ent, policy):
        stim = self.envNet(conv, flat, ent)
        feat = torch.cat((stim, policy), 1)
        repr = self.repr(feat)
        x = self.relu(repr)
        x = self.fc(x)
        x_sigmoid = self.activation(x)
        x = x.view(1, -1)
        return x, x_sigmoid, repr


class LawmakerAbstract(nn.Module):
    def __init__(self, args, config):
        super().__init__()
        self.valNet = None
        self.PunishNet = None

        self.config = config
        self.nRealm = args.nRealm

        self.punishments = {}
        self.values = {}
        self.rewards = {}

        self.states = {}
        self.representations = {}
        self.count = 0

        self.grads = None
        self.grad_clip = 5

    def forward(self, ent, env, policy):
        return torch.zeros([1]), torch.zeros([1])

    def collectStates(self, entID, punishment, val, s, stim):
        self.count += 1

    def collectRewards(self, reward, desciples):
        pass

    def updateStates(self):  # should only dead be here?
        self.count = 0
        punishments = self.punishments
        values = self.values
        self.punishments = {}
        self.values = {}
        if self.config.TEST:
            states = self.states
            representations = self.representations
            self.states = {}
            self.representations = {}
            return punishments, values, states, representations
        return punishments, values, None, None

    def mergeUpdateStates(self):
        punishments, values, states, representations = self.updateStates()
        punishments = merge_dct(punishments)
        values = merge_dct(values)
        if states is not None:
            states = merge_dct(states)
        if representations is not None:
            representations = merge_dct(representations)
        return punishments, values, states, representations

    def updateRewards(self):  # same concerns
        rewards = self.rewards
        self.rewards = {}
        return rewards

    def update(self):
        punishments, values, _, _ = self.updateStates()
        rewards = self.updateRewards()
        return punishments, values, rewards

    def mergeUpdate(self):
        punishments, values, rewards = self.update()
        punishments_lst = merge_dct(punishments)
        values_lst = merge_dct(values)
        rewards_lst = []
        returns_lst = []
        for entID in rewards.keys():
            rewards_lst += rewards[entID]
            returns_lst += discountRewards(rewards[entID])

        return punishments_lst, values_lst, rewards_lst, returns_lst

    def backward(self, valWeight=0.25, entWeight=None):
        self.grads = param.getGrads(self)


def merge_dct(dct):
    lst = []
    for v in dct.values():
        lst += v
    return lst


class Lawmaker(LawmakerAbstract):
    def __init__(self, args, config):
        super().__init__(args, config)
        self.valNet = ValNet(config)
        self.PunishNet = PunishNet(config)

    def forward(self, ent, env, policy):
        s = torchlib.Stim(ent, env, self.config)
        s.policy = policy
        val = self.valNet(s.conv, s.flat, s.ents)

        punishment, punishment_sigm, stim = self.PunishNet(s.conv, s.flat, s.ents, s.policy)

        self.collectStates(ent.entID, punishment, val, s, stim)

        return punishment_sigm, val

    def collectStates(self, entID, punishment, val, s, stim):
        self.count += 1
        if entID not in self.punishments.keys():
            self.punishments[entID] = []
            self.values[entID] = []
            self.rewards[entID] = []

        if not self.config.TEST:
            self.punishments[entID].append(punishment)
            self.values[entID].append(val)
        else:
            self.punishments[entID].append(punishment.detach().numpy()[0][0])
            self.values[entID].append(val.detach().numpy()[0][0])
            if entID not in self.states.keys():
                self.states[entID] = []
                self.representations[entID] = []
            self.states[entID].append(s)
            self.representations[entID].append(stim.detach().numpy()[0])

    def collectRewards(self, reward, desciples):
        if self.config.TEST:
            return
        # shared_reward = reward / len(desciples)  # may be share only with those within certain radius from death?
        for entID in desciples:
            self.rewards[entID].append(reward)  # should this be shared_reward or reward?

    def backward(self, valWeight=0.25, entWeight=None):
        if entWeight is None:
            entWeight = self.config.ENTROPY

        punishments, values, rewards, rets = self.mergeUpdate()
        returns = torch.tensor(rets).view(-1, 1).float()
        punishments = torch.cat(punishments)
        values = torch.cat(values)

        pg, entropy = loss.PG_lawmaker(punishments, values, returns)
        valLoss = loss.valueLoss(values, returns)
        totLoss = pg + valWeight * valLoss + entWeight * entropy

        totLoss.backward()
        self.grads = param.getGrads(self)


def gatherStatistics(lawmakers):
    values = [lawmakers[i].values for i in range(len(lawmakers))]
    punishments = [lawmakers[i].punishments for i in range(len(lawmakers))]
    rewards = [lawmakers[i].rewards for i in range(len(lawmakers))]
    return values, punishments, rewards
