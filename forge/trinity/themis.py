import random

import tensorflow as tf
import numpy as np
import gc

from forge.blade.entity.lawmaker_zero import LawmakerZero
from forge.blade.entity.lawmaker_zero.config import get_config


class Themis:
    def __init__(self):
        self.stepCount = 0
        self.currentAction = [0] * 8
        self.prevReward = 0
        self.curReward = 0
        self.prevMax = 0
        self.curMax = 0
        self.era = 6
        self.testPeriod = 120
        self.lawmakerZero = []
        self.setupLawmakerZero()

    def stepLawmakerZero(self, state, reward):
        if self.stepCount == 0:
            for i in range(8):
                with tf.variable_scope('lawmaker' + str(i)):
                    self.lawmakerZero[i].initStep(state[3 * i + 3:3 * i + 6] + state, reward, np.random.randint(0, 10),
                                                  False)
        else:
            for i in range(8):
                with tf.variable_scope('lawmaker' + str(i)):
                    self.lawmakerZero[i].updateModel(np.array(state[3 * i + 3:3 * i + 6] + state).reshape((1, 30)),
                                                     reward, False)
                self.currentAction[i] = self.lawmakerZero[i].stepEnv()
        self.stepCount += 1
        self.save()

    def save_model(self):
        for lawmaker in self.lawmakerZero:
            lawmaker.save()

    def save(self):
        ROOT = 'resource/exps/laws/'
        with open(ROOT + 'actions.txt', 'a') as f:
            for action in self.currentAction:
                f.write("%s " % action)
            f.write('\n')

    def getAction(self, annId):
        return self.currentAction[annId]

    def election(self):
        self.lawmakerZero.clear()
        self.session.close()
        tf.reset_default_graph()
        gc.collect()
        self.era += 1
        self.curMax = 0
        self.prevMax = 0
        self.stepCount = 0
        self.setupLawmakerZero()

    def voteForMax(self, reward):
        self.curMax += reward
        if self.stepCount % self.testPeriod == 0 and self.stepCount != 0:
            if self.curMax <= self.prevMax:
                self.election()
                return True
            else:
                self.prevMax = self.curMax
                self.curMax = 0
        return False

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
        self.session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

        for i in range(8):
            flags['era'] = 'era' + str(self.era)
            flags['env_name'] = 'lawmaker' + str(i)
            config = get_config(flags) or flags

            with tf.variable_scope('lawmaker' + str(i)):
                self.lawmakerZero.append(LawmakerZero(config, self.session))
