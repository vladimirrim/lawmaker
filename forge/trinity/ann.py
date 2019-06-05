from pdb import set_trace as T
import numpy as np
import torch

from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.distributions import Categorical
from torch.optim import Adam

from forge.blade.action.tree import ActionTree
from forge.blade.action.v2 import ActionV2
from forge.blade.lib.enums import Neon
from forge.blade.lib import enums
from forge.ethyr import torch as torchlib
from forge.blade import entity

from copy import deepcopy
from forge.ethyr.torch import loss
from forge.ethyr.rollouts import discountRewards
from torch.nn.utils import clip_grad_norm_


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
        from forge.blade.core import realm
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
        self.fc = torch.nn.Linear(config.HIDDEN + 5, 1)
        self.activation = torch.nn.Sigmoid()
        self.envNet = Env(config)

    def forward(self, conv, flat, ent, policy):
        stim = self.envNet(conv, flat, ent)
        feat = torch.cat((stim, policy), 1)
        x = self.fc(feat)
        x_sigmoid = self.activation(x)
        x = x.view(1, -1)
        return x, x_sigmoid


class Lawmaker(nn.Module):
    def __init__(self, args, config):
        super().__init__()
        self.valNet = ValNet(config)

        self.config = config
        self.PunishNet = PunishNet(config)
        self.nRealm = args.nRealm

        self.update_period = 2 ** 12 * args.nRealm

        self.punishments = [{} for _ in range(args.nRealm)]
        self.values = [{} for _ in range(args.nRealm)]
        self.rewards = [{} for _ in range(args.nRealm)]
        self.count = 0

        self.opt = Adam(self.parameters(), lr=1e-3)
        self.grad_clip = 5

    def gatherStatistics(self, lawmakers):
        self.count = np.sum([lm.count for lm in lawmakers])
        self.values = [lawmakers[i].values[i] for i in range(len(lawmakers))]
        self.punishments = [lawmakers[i].punishments[i] for i in range(len(lawmakers))]
        self.rewards = [lawmakers[i].rewards[i] for i in range(len(lawmakers))]

    def forward(self, ent, env, policy, idx):
        s = torchlib.Stim(ent, env, self.config)
        val = self.valNet(s.conv, s.flat, s.ents)

        punishment, punishment_sigm = self.PunishNet(s.conv, s.flat, s.ents, policy)

        self.collectStates(ent.entID, punishment, val, idx)

        return punishment_sigm, val

    def collectStates(self, entID, punishment, val, idx):
        if entID not in self.punishments[idx].keys():
            self.punishments[idx][entID] = []
            self.values[idx][entID] = []
            self.rewards[idx][entID] = []
        self.punishments[idx][entID].append(punishment)
        self.values[idx][entID].append(val)
        self.count += 1

    def updateStates(self):  # bad? should only dead be here?
        punishments = deepcopy(self.punishments)  # do we need deepcopy?
        values = deepcopy(self.values)
        self.punishments = [{} for _ in range(self.nRealm)]
        self.values = [{} for _ in range(self.nRealm)]
        self.count = 0
        return punishments, values

    def collectRewards(self, reward, idx, desciples):
        shared_reward = reward / len(
            self.rewards[idx])  # may be share only with those within certain radius from death?
        for entID in desciples:
            self.rewards[idx][entID].append(reward)  # should this be shared_reward or reward?

    def updateRewards(self):  # same concerns
        rewards = deepcopy(self.rewards)
        self.rewards = [{} for _ in range(self.nRealm)]
        return rewards

    def update(self):
        punishments, values = self.updateStates()
        rewards = self.updateRewards()
        return punishments, values, rewards

    def mergeUpdate(self):
        punishments, values, rewards = self.update()
        punishments_lst = []
        values_lst = []
        rewards_lst = []
        returns_lst = []
        for idx in range(len(punishments)):
            for entID in punishments[idx].keys():
                punishments_lst += punishments[idx][entID]
                values_lst += values[idx][entID]
                rewards_lst += rewards[idx][entID]
                returns_lst += discountRewards(rewards[idx][entID])

        return punishments_lst, values_lst, rewards_lst, returns_lst

    def backward(self, valWeight=0.25, entWeight=None):
        # print('Doing backward')
        if entWeight is None:
            entWeight = self.config.ENTROPY

        self.opt.zero_grad()

        punishments, values, rewards, rets = self.mergeUpdate()
        returns = torch.tensor(rets).view(-1, 1).float()
        punishments = torch.tensor(punishments).view(-1, 1).float()
        values = torch.cat(values)
        # print('punishments:', punishments[:5], '; len:', len(punishments))
        # print('values:', values[:5], '; len:', len(values))
        # print('returns:', returns[:5], '; len:', len(returns))

        pg, entropy = loss.PG_lawmaker(punishments, values, returns)
        valLoss = loss.valueLoss(values, returns)
        totLoss = Variable(pg + valWeight * valLoss + entWeight * entropy, requires_grad=True)
        # print("totLoss:", totLoss)

        totLoss.backward()
        if self.grad_clip is not None:
            clip_grad_norm_(self.parameters(), self.grad_clip)
        self.opt.step()
