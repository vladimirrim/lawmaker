from __future__ import print_function

import os
import random
from functools import reduce

import numpy as np
import tensorflow as tf

from .base import BaseModel
from .history import History
from .ops import linear, clipped_error
from .replay_memory import ReplayMemory
from .utils import save_pkl, load_pkl


class LawmakerZero(BaseModel):
    def __init__(self, config, sess):
        super(LawmakerZero, self).__init__(config)
        self.sess = sess
        self.weight_dir = 'weights'

        self.step = 0
        self.history = History(self.config)
        self.action_size = 11
        self.memory = ReplayMemory(self.config, self.model_dir)

        with tf.variable_scope('step'):
            self.step_op = tf.Variable(0, trainable=False, name='step')
            self.step_input = tf.placeholder('int32', name='step_input')
            self.step_assign_op = self.step_op.assign(self.step_input)

        self.build_dqn()
        self.start_step = self.step_op.eval(session=self.sess)

        self.num_game, self.update_count, self.ep_reward = 0, 0, 0.
        self.max_avg_ep_reward = 0
        self.currentAction = 0
        self.ep_rewards, self.actions = [], []

    def initStep(self, screen, reward, action, terminal):
        self.screen, self.reward, self.currentAction, self.terminal = screen, reward, action, terminal
        for _ in range(self.history_length):
            self.history.add(self.screen)

    def stepEnv(self):
        self.currentAction = self.predict(self.history.get())
        return self.currentAction

    def updateModel(self, screen, reward, terminal):
        self.observe(screen, reward, self.currentAction, terminal)

        self.num_game += 1
        self.ep_rewards.append(self.ep_reward)
        self.ep_reward = 0.
        self.step += 1

        self.actions.append(self.currentAction)

    def predict(self, s_t, test_ep=None):
        ep = test_ep or (self.ep_end +
                         max(0., (self.ep_start - self.ep_end)
                             * (self.ep_end_t - max(0., self.step - self.learn_start)) / self.ep_end_t))

        if random.random() < ep:
            action = random.randrange(self.action_size)
        else:
            action = self.q_action.eval({self.s_t: [s_t]}, session=self.sess)[0]

        return action

    def observe(self, screen, reward, action, terminal):

        self.history.add(screen)
        self.memory.add(screen, reward, action, terminal)

        if self.step > self.learn_start:
            if self.step % self.train_frequency == 0:
                self.q_learning_mini_batch()

            if self.step % self.target_q_update_step == self.target_q_update_step - 1:
                self.update_target_q_network()

    def q_learning_mini_batch(self):
        if self.memory.count < self.history_length:
            return
        else:
            s_t, action, reward, s_t_plus_1, terminal = self.memory.sample()

        if self.double_q:
            # Double Q-learning
            pred_action = self.q_action.eval({self.s_t: s_t_plus_1}, session=self.sess)

            q_t_plus_1_with_pred_action = self.target_q_with_idx.eval({
                self.target_s_t: s_t_plus_1,
                self.target_q_idx: [[idx, pred_a] for idx, pred_a in enumerate(pred_action)]
            }, session=self.sess)
            target_q_t = (1. - terminal) * self.discount * q_t_plus_1_with_pred_action + reward
        else:
            q_t_plus_1 = self.target_q.eval({self.target_s_t: s_t_plus_1}, session=self.sess)

            terminal = np.array(terminal) + 0.
            max_q_t_plus_1 = np.max(q_t_plus_1, axis=1)
            target_q_t = (1. - terminal) * self.discount * max_q_t_plus_1 + reward

        _, q_t, loss = self.sess.run([self.optim, self.q, self.loss], {
            self.target_q_t: target_q_t,
            self.action: action,
            self.s_t: s_t,
            self.learning_rate_step: self.step,
        })

        self.update_count += 1

    def build_dqn(self):
        self.w = {}
        self.t_w = {}

        activation_fn = tf.nn.relu

        # training network
        with tf.variable_scope('prediction'):
            self.s_t = tf.placeholder('float32',
                                      [None, self.history_length, self.screen_height, self.screen_width],
                                      name='s_t')
            shape = self.s_t.get_shape().as_list()
            self.s_t_flat = tf.reshape(self.s_t, [-1, reduce(lambda x, y: x * y, shape[1:])])

            self.l1, self.w['l1_w'], self.w['l1_b'] = linear(self.s_t_flat, 512,
                                                             name='l1')
            self.l2, self.w['l2_w'], self.w['l2_b'] = linear(self.l1, 512,
                                                             name='l2')
            self.l3, self.w['l3_w'], self.w['l3_b'] = linear(self.l2, 512,
                                                             name='l3')

            shape = self.l3.get_shape().as_list()
            self.l3_flat = tf.reshape(self.l3, [-1, reduce(lambda x, y: x * y, shape[1:])])

            if self.dueling:
                self.value_hid, self.w['l4_val_w'], self.w['l4_val_b'] = \
                    linear(self.l3_flat, 512, activation_fn=activation_fn, name='value_hid')

                self.adv_hid, self.w['l4_adv_w'], self.w['l4_adv_b'] = \
                    linear(self.l3_flat, 512, activation_fn=activation_fn, name='adv_hid')

                self.value, self.w['val_w_out'], self.w['val_w_b'] = \
                    linear(self.value_hid, 1, name='value_out')

                self.advantage, self.w['adv_w_out'], self.w['adv_w_b'] = \
                    linear(self.adv_hid, self.action_size, name='adv_out')

                # Average Dueling
                self.q = self.value + (self.advantage -
                                       tf.reduce_mean(self.advantage, reduction_indices=1, keep_dims=True))
            else:
                self.l4, self.w['l4_w'], self.w['l4_b'] = linear(self.l3_flat, 512, activation_fn=activation_fn,
                                                                 name='l4')
                self.q, self.w['q_w'], self.w['q_b'] = linear(self.l4, self.action_size,
                                                              name='q')

            self.q_action = tf.argmax(self.q, axis=1)

        # target network
        with tf.variable_scope('target'):
            self.target_s_t = tf.placeholder('float32',
                                             [None, self.history_length, self.screen_height, self.screen_width],
                                             name='target_s_t')

            shape = self.target_s_t.get_shape().as_list()
            self.target_s_t_flat = tf.reshape(self.target_s_t, [-1, reduce(lambda x, y: x * y, shape[1:])])

            self.target_l1, self.t_w['l1_w'], self.t_w['l1_b'] = linear(self.target_s_t_flat, 512
                                                                        , name='target_l1')
            self.target_l2, self.t_w['l2_w'], self.t_w['l2_b'] = linear(self.target_l1, 512,
                                                                        name='target_l2')
            self.target_l3, self.t_w['l3_w'], self.t_w['l3_b'] = linear(self.target_l2, 512,
                                                                        name='target_l3')

            shape = self.target_l3.get_shape().as_list()
            self.target_l3_flat = tf.reshape(self.target_l3, [-1, reduce(lambda x, y: x * y, shape[1:])])

            if self.dueling:
                self.t_value_hid, self.t_w['l4_val_w'], self.t_w['l4_val_b'] = \
                    linear(self.target_l3_flat, 512, activation_fn=activation_fn, name='target_value_hid')

                self.t_adv_hid, self.t_w['l4_adv_w'], self.t_w['l4_adv_b'] = \
                    linear(self.target_l3_flat, 512, activation_fn=activation_fn, name='target_adv_hid')

                self.t_value, self.t_w['val_w_out'], self.t_w['val_w_b'] = \
                    linear(self.t_value_hid, 1, name='target_value_out')

                self.t_advantage, self.t_w['adv_w_out'], self.t_w['adv_w_b'] = \
                    linear(self.t_adv_hid, self.action_size, name='target_adv_out')

                # Average Dueling
                self.target_q = self.t_value + (self.t_advantage -
                                                tf.reduce_mean(self.t_advantage, reduction_indices=1, keep_dims=True))
            else:
                self.target_l4, self.t_w['l4_w'], self.t_w['l4_b'] = \
                    linear(self.target_l3_flat, 512, activation_fn=activation_fn, name='target_l4')
                self.target_q, self.t_w['q_w'], self.t_w['q_b'] = \
                    linear(self.target_l4, self.action_size, name='target_q')

            self.target_q_idx = tf.placeholder('int32', [None, None], 'outputs_idx')
            self.target_q_with_idx = tf.gather_nd(self.target_q, self.target_q_idx)

        with tf.variable_scope('pred_to_target'):
            self.t_w_input = {}
            self.t_w_assign_op = {}

            for name in self.w.keys():
                self.t_w_input[name] = tf.placeholder('float32', self.t_w[name].get_shape().as_list(), name=name)
                self.t_w_assign_op[name] = self.t_w[name].assign(self.t_w_input[name])

        # optimizer
        with tf.variable_scope('optimizer'):
            self.target_q_t = tf.placeholder('float32', name='target_q_t')
            self.action = tf.placeholder('int64', name='action')

            action_one_hot = tf.one_hot(self.action, self.action_size, 1.0, 0.0, name='action_one_hot')
            q_acted = tf.reduce_sum(self.q * action_one_hot, reduction_indices=1, name='q_acted')

            self.delta = self.target_q_t - q_acted

            self.global_step = tf.Variable(0, trainable=False)

            self.loss = tf.reduce_mean(clipped_error(self.delta), name='loss')
            self.learning_rate_step = tf.placeholder('int64', name='learning_rate_step')
            self.learning_rate_op = tf.maximum(self.learning_rate_minimum,
                                               tf.train.exponential_decay(
                                                   self.learning_rate,
                                                   self.learning_rate_step,
                                                   self.learning_rate_decay_step,
                                                   self.learning_rate_decay,
                                                   staircase=True))
            self.optim = tf.train.RMSPropOptimizer(
                self.learning_rate_op, momentum=0.95, epsilon=0.01).minimize(self.loss)

        with tf.variable_scope('summary'):
            scalar_summary_tags = ['average.reward', 'average.loss', 'average.q',
                                   'episode.max reward', 'episode.min reward', 'episode.avg reward',
                                   'episode.num of game', 'training.learning_rate']

            self.summary_placeholders = {}
            self.summary_ops = {}

            for tag in scalar_summary_tags:
                self.summary_placeholders[tag] = tf.placeholder('float32', name=tag.replace(' ', '_'))
                self.summary_ops[tag] = tf.summary.scalar("%s-%s/%s" % ('law', 'maker', tag),
                                                          self.summary_placeholders[tag])

            histogram_summary_tags = ['episode.rewards', 'episode.actions']

            for tag in histogram_summary_tags:
                self.summary_placeholders[tag] = tf.placeholder('float32', name=tag.replace(' ', '_'))
                self.summary_ops[tag] = tf.summary.histogram(tag, self.summary_placeholders[tag])

            self.writer = tf.summary.FileWriter('./logs/%s' % self.model_dir, self.sess.graph)

        tf.global_variables_initializer().run(session=self.sess)

        self._saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.env_name),
                                     max_to_keep=1)

        self.load_model()
        self.update_target_q_network()

    def save(self):
        self.step_assign_op.eval({self.step_input: self.step + 1}, session=self.sess)
        self.save_model(self.step + 1)

    def update_target_q_network(self):
        for name in self.w.keys():
            self.t_w_assign_op[name].eval({self.t_w_input[name]: self.w[name].eval(session=self.sess)},
                                          session=self.sess)

    def save_weight_to_pkl(self):
        if not os.path.exists(self.weight_dir):
            os.makedirs(self.weight_dir)

        for name in self.w.keys():
            save_pkl(self.w[name].eval(), os.path.join(self.weight_dir, "%s.pkl" % name))

    def load_weight_from_pkl(self, cpu_mode=False):
        with tf.variable_scope('load_pred_from_pkl'):
            self.w_input = {}
            self.w_assign_op = {}

            for name in self.w.keys():
                self.w_input[name] = tf.placeholder('float32', self.w[name].get_shape().as_list(), name=name)
                self.w_assign_op[name] = self.w[name].assign(self.w_input[name])

        for name in self.w.keys():
            self.w_assign_op[name].eval({self.w_input[name]: load_pkl(os.path.join(self.weight_dir, "%s.pkl" % name))})

        self.update_target_q_network()

    def inject_summary(self, tag_dict):
        summary_str_lists = self.sess.run([self.summary_ops[tag] for tag in tag_dict.keys()], {
            self.summary_placeholders[tag]: value for tag, value in tag_dict.items()
        })
        for summary_str in summary_str_lists:
            self.writer.add_summary(summary_str, self.step)
