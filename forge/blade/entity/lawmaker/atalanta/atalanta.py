"""An example of training A2C against OpenAI Gym Envs.
This script is an example of training a A2C agent against OpenAI Gym envs.
Both discrete and continuous action spaces are supported.
To solve CartPole-v0, run:
    python train_a2c_gym.py 8 --env CartPole-v0
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from builtins import *  # NOQA

from chainerrl.policies import FCBNDeterministicPolicy, FCLSTMDeterministicPolicy
from future import standard_library

from forge.blade.entity.lawmaker.atalanta.agent import A2C
from forge.blade.entity.lawmaker.lawmaker import BatchLawmaker, Lawmaker

standard_library.install_aliases()  # NOQA

import chainer
from chainer import functions as F
import numpy as np

from chainerrl.agents import a2c
from chainerrl import misc
from chainerrl.optimizers.nonbias_weight_decay import NonbiasWeightDecay
from chainerrl import policies
from chainerrl import v_function


class A2CGaussian(chainer.ChainList, a2c.A2CModel):

    def __init__(self, obs_size, action_size):
        self.pi = policies.FCGaussianPolicyWithFixedCovariance(
            obs_size,
            action_size,
            np.log(np.e - 1),
            n_hidden_layers=2,
            n_hidden_channels=64)
        self.v = v_function.FCVFunction(obs_size, n_hidden_layers=2,
                                        n_hidden_channels=64,
                                        nonlinearity=F.tanh)
        super().__init__(self.pi, self.v)

    def pi_and_v(self, state):
        return self.pi(state), self.v(state)


class A2CDeterministic(chainer.ChainList, a2c.A2CModel):

    def __init__(self, obs_size, action_size):
        self.pi = FCLSTMDeterministicPolicy(n_input_channels=obs_size, action_size=action_size, n_hidden_layers=2,
                                            n_hidden_channels=64, min_action=0, max_action=1)
        self.v = v_function.FCVFunction(obs_size, n_hidden_layers=2,
                                        n_hidden_channels=64,
                                        nonlinearity=F.tanh)
        super().__init__(self.pi, self.v)

    def pi_and_v(self, state):
        return self.pi(state), self.v(state)


class Atalanta(BatchLawmaker, Lawmaker):

    def save(self, dirname):
        self.agent.save(dirname)

    def get_statistics(self):
        pass

    def batch_act(self, batch_obs):
        return self.agent.batch_act(batch_obs)

    def batch_act_and_train(self, batch_obs):
        return self.agent.batch_act_and_train(batch_obs)

    def batch_observe(self, batch_obs, batch_reward, batch_done, batch_reset):
        return self.agent.batch_observe(batch_obs, batch_reward, batch_done, batch_reset)

    def batch_observe_and_train(self, batch_obs, batch_reward, batch_done, batch_reset):
        return self.agent.batch_observe_and_train(batch_obs, batch_reward, batch_done, batch_reset)

    def load(self, dirname):
        self.agent.load(dirname)

    def __init__(self, obs_space):
        self.seed = 0
        self.rmsprop_epsilon = 1e-5
        self.gamma = 0.99
        self.use_gae = False
        self.tau = 0.95
        self.num_envs = 8
        self.lr = 7e-4
        self.weight_decay = 0.0
        self.max_grad_norm = 0.5
        self.alpha = 0.99
        self.update_steps = 5

        # Set a random seed used in ChainerRL.
        # If you use more than one processes, the results will be no longer
        # deterministic even with the same random seed.
        misc.set_random_seed(self.seed)

        # Set different random seeds for different subprocesses.
        # If seed=0 and processes=4, subprocess seeds are [0, 1, 2, 3].
        # If seed=1 and processes=4, subprocess seeds are [4, 5, 6, 7].
        process_seeds = np.arange(self.num_envs) + self.seed * self.num_envs
        assert process_seeds.max() < 2 ** 3

        action_space = 1

        # Switch policy types accordingly to action space types
        model = A2CGaussian(obs_space, action_space)

        optimizer = chainer.optimizers.RMSprop(self.lr,
                                               eps=self.rmsprop_epsilon,
                                               alpha=self.alpha)
        optimizer.setup(model)
        optimizer.add_hook(chainer.optimizer.GradientClipping(self.max_grad_norm))
        if self.weight_decay > 0:
            optimizer.add_hook(NonbiasWeightDecay(self.weight_decay))

        self.agent = A2C(model, optimizer, gamma=self.gamma,
                         num_processes=self.num_envs,
                         update_steps=self.update_steps,
                         use_gae=self.use_gae,
                         tau=self.tau)

    def act_and_train(self, obs, reward):
        pass

    def act(self, obs):
        pass

    def stop_episode_and_train(self, state, reward, done=False):
        pass

    def stop_episode(self):
        pass
