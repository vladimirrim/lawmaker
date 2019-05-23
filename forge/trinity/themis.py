import os
import random

import tensorflow as tf
import numpy as np
import gc

from forge.blade.entity.lawmaker_zero import LawmakerZero
from forge.blade.entity.lawmaker_zero.config import get_config


class Themis:
    def __init__(self):
        self.stepCount = 0
        self.currentAction = [2] * 8
        self.rewards = np.zeros(8)
        self.prevReward = 0
        self.curReward = 0
        self.prevMax = 0
        self.curMax = 0
        self.era = 1
        self.testPeriod = 100
        self.lawmakerZero = []
        self.featureSize = 5
        self.setupLawmakerZero()

    def stepLawmakerZero(self, state, reward, rewards):
        if self.stepCount == 0:
            for i in range(8):
                with tf.variable_scope('lawmaker' + str(i)):
                    self.lawmakerZero[i].initStep(state[self.featureSize * i + self.featureSize
                                                        :self.featureSize * i + self.featureSize * 2]
                                                  + state, reward + rewards[i], 1,
                                                  False)
        else:
            for i in range(8):
                with tf.variable_scope('lawmaker' + str(i)):
                    self.lawmakerZero[i].updateModel(np.array(state[self.featureSize * i + self.featureSize
                                                                    :self.featureSize * i + self.featureSize * 2]
                                                              + state).reshape((1, self.featureSize * 10)),
                                                     reward + rewards[i], False)
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

    def heritage(self):
        self.era += 1
        king = np.argmax(self.rewards)
        self.lawmakerZero[king].era = 'era' + str(self.era)
        self.saveKing(king)
        self.lawmakerZero.clear()
        self.session.close()
        tf.reset_default_graph()
        gc.collect()
        self.setupLawmakerZero()
        self.rewards = np.zeros(8)

    def saveKing(self, king):
        self.lawmakerZero[king].save()
        kingargs = tf.contrib.framework.list_variables(self.lawmakerZero[king].checkpoint_dir)
        print(king)
        for i in range(8):
            if i == king:
                continue
            with tf.Graph().as_default(), tf.Session().as_default() as sess:
                new_vars = []
                for name, shape in kingargs:
                    v = tf.contrib.framework.load_variable(self.lawmakerZero[king].checkpoint_dir, name)
                    new_vars.append(tf.Variable(v, name=name.replace('lawmaker' + str(king), 'lawmaker' + str(i))))
                    saveDir = self.lawmakerZero[king].checkpoint_dir.replace('lawmaker' + str(king),
                                                                             'lawmaker' + str(i))
                    if not os.path.exists(saveDir):
                        os.makedirs(saveDir)
                    saver = tf.train.Saver(new_vars)
                    sess.run(tf.global_variables_initializer())
                    saver.save(sess, saveDir, global_step=self.stepCount)

    def voteForBest(self, rewards):
        self.rewards += rewards
        if self.stepCount % self.testPeriod == 0 and self.stepCount != 0:
            self.heritage()
            return True
        return False

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
        flags['dueling'] = True
        flags['double_q'] = True

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
