import tensorflow as tf
import numpy as np


class Lawmaker:
    def __init__(self):
        tf.reset_default_graph()
        # These lines establish the feed-forward part of the network used to choose actions
        self.stateVector = tf.placeholder(shape=[1, 25], dtype=tf.float32)
        self.W = tf.Variable(tf.random_uniform([25, 11], 0, 0.01))
        self.Qout = tf.matmul(self.stateVector, self.W)
        self.predict = tf.argmax(self.Qout, 1)

        # Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        self.nextQ = tf.placeholder(shape=[1, 11], dtype=tf.float32)
        loss = tf.reduce_sum(tf.square(self.nextQ - self.Qout))
        self.trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
        self.session = tf.Session()
        self.actions = []
        self.rewards = []
        self.init = tf.global_variables_initializer()
        self.e = 0.1
        self.y = .99

        self.session.run(self.init)
        self.updateModel = self.trainer.minimize(loss)
        self.saver = tf.train.Saver()

    def step(self, s):
        # Set learning parameters
        self.prevState = s

        # Choose an action by greedily (with e chance of random action) from the Q-network
        self.a, self.allQ = self.session.run([self.predict, self.Qout],
                                             feed_dict={self.stateVector: s})
        if np.random.rand(1) < self.e:
            self.a[0] = np.random.randint(0, 10)

        self.actions.append(self.a[0])

        return self.a[0]

    def save(self):
        ROOT = 'resource/exps/laws/'
        self.saver.save(self.session, ROOT + 'model.ckpt')

    def load(self):
        ROOT = 'resource/exps/laws/'
        self.saver.restore(self.session, ROOT + 'model.ckpt')

    def updateLaw(self, s, r):
        # Obtain the Q' values by feeding the new state through our network
        Q1 = self.session.run(self.Qout, feed_dict={self.stateVector: s})
        # Obtain maxQ' and set our target value for chosen action.
        rAll = 0

        maxQ1 = np.max(Q1)
        targetQ = self.allQ
        targetQ[0, self.a[0]] = r + self.y * maxQ1
        # Train our network using target and predicted Q values
        _, W1 = self.session.run([self.updateModel, self.W],
                                 feed_dict={self.stateVector: self.prevState,
                                            self.nextQ: self.allQ})
        rAll += r
        self.prevState = s

        self.e = 1. / ((len(self.rewards) / 50) + 10)
        self.rewards.append(rAll)
