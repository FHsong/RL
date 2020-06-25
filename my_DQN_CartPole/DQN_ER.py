import  gym,os, matplotlib, random
import  numpy as np
import  tensorflow as tf
from    tensorflow.keras import layers,optimizers,losses, Model
from    collections import deque
from matplotlib import pyplot as plt

matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['figure.titlesize'] = 14
matplotlib.rcParams['figure.figsize'] = [8, 5]
matplotlib.rcParams['font.family'] = ['KaiTi']
matplotlib.rcParams['axes.unicode_minus']=False

#####################  hyper parameters  ####################
lambd = .9             # 折扣率(decay factor)
epsilon = 0.1                 # epsilon-greedy算法参数，越大随机性越大，越倾向于探索行为。
is_render = False          # 是否渲染游戏


class DQN(Model):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc1 = layers.Dense(32, kernel_initializer='he_normal', activation='relu')
        self.fc2 = layers.Dense(16, kernel_initializer='he_normal', activation='relu')
        self.out = layers.Dense(2, kernel_initializer='he_normal', activation=None)

    def call(self, inputs):
        x = self.fc1(inputs)
        x = self.fc2(x)
        x = self.out(x)
        return x



class DQN_ER():
    def __init__(self, env, learning_rate, n_episode, batch):
        super(DQN_ER, self).__init__()
        self.env = env

        # 建立两个网络
        self.Q_network = DQN()
        self.target_Q_network = DQN()

        ## epsilon-greedy相关参数
        self.epsilon = 1.0  # epsilon大小，随机数大于epsilon，则进行开发；否则，进行探索。
        self.epsilon_decay = 0.995  # 减少率：epsilon会随着迭代而更新，每次会乘以0.995
        self.epsilon_min = 0.01  # 小于最小epsilon就不再减少了。

        # 其余超参数
        self.memory = deque(maxlen=2000)  # 队列，最大值是2000
        self.batch = batch
        self.gamma = 0.95  # 折扣率
        self.learning_rate = learning_rate  # 学习率
        self.optimizer = tf.optimizers.Adam(learning_rate)  # 优化器
        self.is_rend = False  # 默认不渲染，当达到一定次数后，开始渲染。

        self.n_episode = n_episode  # 迭代多少轮

    def update_target_Q(self):
        for i, target in zip(self.Q_network.trainable_weights, self.target_Q_network.trainable_weights):
            target.assign(i)


    def update_epsilon(self):
        '''
                用于更新epsilon
                    除非已经epsilon_min还小，否则比每次都乘以减少率epsilon_decay。
        '''
        if self.epsilon >= self.epsilon_min:
            self.epsilon *= self.epsilon_decay


    def remember(self, state, action, next_state, r, done):
        data = (state, action, next_state, r, done)
        self.memory.append(data)


    def get_action(self, state):
        '''
         用epsilon-greedy的方式求动作。
        '''
        # 先随机一个数，如果比epsilon大，那么，就输出最大Q值的动作。
        if np.random.rand() >= self.epsilon:
            state = tf.constant(state, dtype=tf.float32)
            state = tf.expand_dims(state, axis=0)
            q = self.Q_network(state).numpy()
            action = np.argmax(q)
            return action
        else:
            action = random.randint(0, 1)
            return action


    def process_data(self):
        # 随机从队列中取出一个batch大小的数据
        data = random.sample(self.memory, self.batch)
        state = tf.constant([d[0] for d in data], dtype=tf.float32)
        action = tf.constant([d[1] for d in data], dtype=tf.float32)
        next_state = tf.constant([d[2] for d in data], dtype=tf.float32)
        r = [d[3] for d in data]
        done = [d[4] for d in data]

        y = self.Q_network(state).numpy()
        Q1 = self.target_Q_network(next_state).numpy()

        for i, (_, a, _, r, done) in enumerate(data):
            if done:
                target = r
            else:
                target = r + self.gamma * np.max(Q1[i])
            target = np.array(target, dtype='float32')
            y[i][a] = target

        return state, y


    def update_Q_network(self):
        '''
        更新Q_network，最小化target和Q的距离
        '''
        state, y = self.process_data()
        with tf.GradientTape() as tape:
            Q = self.Q_network(state)
            loss = tf.losses.mean_squared_error(Q, y)
        grads = tape.gradient(loss, self.Q_network.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.Q_network.trainable_weights))
        return loss


    def train(self, result_path):
        step = 0
        rend = 0
        total_reward = 0
        returns = []
        for episode in range(self.n_episode):
            state = self.env.reset()

            total_loss = []
            loss = 0

            for i in range(500):
                # if is_render == True: env.render() # 渲染
                action = self.get_action(state)
                next_state, r, done, _ = self.env.step(action)

                total_reward += r
                step += 1

                self.remember(state, action, next_state, r, done)
                state = next_state

                if len(self.memory) >= self.batch:
                    loss = self.update_Q_network()
                    total_loss.append(loss)
                    if (step + 1) % 5 == 0:
                        self.update_epsilon()
                        self.update_target_Q()

                # 如果有5个episode成绩大于200，就开始渲染游戏
                if total_reward >= 200:
                    rend += 1
                    if rend == 5:
                        is_render = True

                if done:
                    break

            if episode % 20 == 0:
                returns.append(total_reward / 20)
                total_reward = 0
                print('Episode: {}/{}  | Episode Average Reward: {:.4f}'
                      .format(episode, self.n_episode, returns[-1]))

        self.env.close()
        print(np.array(returns))
        plt.figure()
        plt.plot(np.arange(len(returns)) * 20, np.array(returns))
        plt.plot(np.arange(len(returns)) * 20, np.array(returns), 's')
        plt.xlabel('回合数')
        plt.ylabel('总回报')
        plt.savefig(result_path)
        plt.show()
