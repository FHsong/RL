import 	gym,os
import  numpy as np
import  matplotlib
from matplotlib import pyplot as plt
# Default parameters for plots
matplotlib.rcParams['font.size'] = 18
matplotlib.rcParams['figure.titlesize'] = 18
matplotlib.rcParams['figure.figsize'] = [9, 7]
matplotlib.rcParams['font.family'] = ['KaiTi']
matplotlib.rcParams['axes.unicode_minus']=False

import 	tensorflow as tf
import tensorlayer as tl
from    tensorflow import keras
from    tensorflow.keras import layers,optimizers,losses

RANDOMSEED = 1              # 设置随机种子。建议大家都设置，这样试验可以重现。

# 定义策略网络，生成动作的概率分布
class Network(keras.Model):
    def __init__(self):
        super(Network, self).__init__()
        # 输入为长度为4的向量，输出为左、右2个动作
        self.fc1 = layers.Dense(30, kernel_initializer='he_normal', activation='relu')
        self.fc2 = layers.Dense(16, kernel_initializer='he_normal', activation='relu')
        self.fc3 = layers.Dense(2, kernel_initializer='he_normal', activation='softmax')


    def call(self, x):
        # 状态输入s的shape为向量：[4]
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


class PolicyGradient():
    def __init__(self, env, learning_rate, n_episode):
        self.env = env

        # 定义相关参数
        self.learning_rate = learning_rate  # 学习率
        self.gamma = 0.95 # 折扣
        self.n_episode = n_episode

        # 创建策略网络
        self.pi = Network()
        # self.pi = self.get_model([None, env.observation_space.shape[0]])
        # self.pi.train()

        self.optimizer = tf.optimizers.Adam(lr=learning_rate)  # 网络优化器
        # 用于保存每个ep的数据。
        self.ep_states, self.ep_actions, self.ep_rewards = [], [], []


    def get_model(self, inputs_shape):
        """
        创建一个神经网络
        输入: state
        输出: act
        """
        self.tf_obs = tl.layers.Input(inputs_shape, tf.float32, name="observations")
        layer = tl.layers.Dense(
            n_units=30, act=tf.nn.tanh, W_init=tf.random_normal_initializer(mean=0, stddev=0.3),
            b_init=tf.constant_initializer(0.1), name='fc1'
        )(self.tf_obs)
        # fc2
        all_act = tl.layers.Dense(
            n_units=self.env.action_space.n, act=None, W_init=tf.random_normal_initializer(mean=0, stddev=0.3),
            b_init=tf.constant_initializer(0.1), name='all_act'
        )(layer)
        return tl.models.Model(inputs=self.tf_obs, outputs=all_act, name='PG model')


    def choose_action(self, state):
        """
        用神经网络输出的**策略pi**，选择动作。
        输入: state
        输出: act
        """
        _logits = self.pi(np.array([state], np.float32)).numpy()
        # 从类别分布中采样1个动作, shape: [1], 动作的概率越高，该动作就有更高的概率被采样到
        a = np.random.choice(len(_logits[0]), p=_logits[0])
        return int(a)


    def store_transition(self, state, action, reward):
        """
        保存数据到buffer中
        """
        self.ep_states.append(np.array([state], np.float32))
        self.ep_actions.append(action)
        self.ep_rewards.append(reward)


    def train(self):
        """
        通过带权重更新方法更新神经网络
        """
        # _discount_and_norm_rewards中存储的就是这一ep中，每个状态的G值。
        discounted_ep_rs_norm = self._discount_and_norm_rewards()

        with tf.GradientTape() as tape:
            # 把s放入神经网络，计算_logits
            _logits = self.pi(np.vstack(self.ep_states))

            # 敲黑板
            ## _logits和真正的动作的差距
            # 差距也可以这样算,和sparse_softmax_cross_entropy_with_logits等价的:
            # neg_log_prob = tf.reduce_sum(-tf.log(self.all_act_prob)*tf.one_hot(self.tf_acts, self.n_actions), axis=1)
            neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=_logits, labels=np.array(self.ep_actions))

            # 在原来的差距乘以G值，也就是以G值作为更新
            loss = tf.reduce_mean(neg_log_prob * discounted_ep_rs_norm)

        grad = tape.gradient(loss, self.pi.trainable_weights)
        self.optimizer.apply_gradients(zip(grad, self.pi.trainable_weights))

        self.ep_states, self.ep_actions, self.ep_rewards = [], [], []  # empty episode data
        return discounted_ep_rs_norm


    def _discount_and_norm_rewards(self):
        """
        通过回溯计算G值
        """
        # 先创建一个数组，大小和ep_rs一样。ep_rs记录的是每个状态的收获r。
        discounted_ep_rs = np.zeros_like(self.ep_rewards)
        running_add = 0
        # 从ep_rs的最后往前，逐个计算G
        for t in reversed(range(0, len(self.ep_rewards))):
            running_add = running_add * self.gamma + self.ep_rewards[t]
            discounted_ep_rs[t] = running_add

        # 归一化G值。
        # 我们希望G值有正有负，这样比较容易学习。
        discounted_ep_rs -= np.mean(discounted_ep_rs)
        discounted_ep_rs /= np.std(discounted_ep_rs)
        return discounted_ep_rs



    def run(self):
        print('----------------- Policy Gradient -----------------')
        total_reward = 0
        returns = []

        np.random.seed(RANDOMSEED)
        tf.random.set_seed(RANDOMSEED)
        self.env.seed(RANDOMSEED)  # 不加这一句优化效果极差，也不知道为什么

        for episode in range(self.n_episode):
            state = self.env.reset()
            for step in range(200):
                action = self.choose_action(state)

                next_state, reward, done, _ = self.env.step(action)

                total_reward += reward

                self.store_transition(next_state, action, reward)

                if done:
                    self.train()
                    break
                state = next_state

            if episode % 20 == 0:
                returns.append(total_reward / 20)
                total_reward = 0
                print('Episode: {}/{}  | Episode Average Reward: {:.4f}'
                      .format(episode, self.n_episode, returns[-1]))






