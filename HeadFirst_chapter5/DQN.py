import time
import gym
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from HeadFirst_chapter5.maze_env import Maze
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten


#####################  hyper parameters  ####################
lambd = .9             # 折扣率(decay factor)
epsilon = 0.1                 # epsilon-greedy算法参数，越大随机性越大，越倾向于探索行为。
num_episodes = 1000    # 迭代次数
render = False          # 是否渲染游戏
running_reward = None

##################### DQN ##########################

class DQNModel(Model):
    def __init__(self):
        super(DQNModel, self).__init__()
        self.fc1 = Dense(10, kernel_initializer='he_normal', activation='relu')
        self.fc2 = Dense(4, kernel_initializer='he_normal', activation=None)

    def call(self,inputs):
        x = self.fc1(inputs)
        x = self.fc2(x)
        return x

## 把分类的数字表示，变成onehot表示。
# 例如有4类，那么第三类变为：[0,0,1,0]的表示。
def to_one_hot(i, n_classes=None):
    a = np.zeros(n_classes, 'uint8')    # 这里先按照分类数量构建一个全0向量
    a[i] = 1                            # 然后点亮需要onehot的位数。
    return a


## Define Q-network q(a,s) that ouput the rewards of 4 actions by given state, i.e. Action-Value Function.
# encoding for state: 4x4 grid can be represented by one-hot vector with 16 integers.
def get_model(inputs_shape):
    '''
    定义Q网络模型：
    1. 注意输入的shape和输出的shape
    2. W_init和b_init是模型在初始化的时候，控制初始化参数的随机。该代码中用正态分布，均值0，方差0.01的方式初始化参数。
    '''
    ni = tl.layers.Input(inputs_shape, name='observation')
    # nn1 = tl.layers.Dense(4, act=None, W_init=tf.random_uniform_initializer(0, 0.01), b_init=None, name='q_a_s1')(ni)
    nn = tl.layers.Dense(4, act=None, W_init=tf.random_uniform_initializer(0, 0.01), b_init=None, name='q_a_s')(ni)

    return tl.models.Model(inputs=ni, outputs=nn, name="Q-Network")


def save_ckpt(model):  # save trained weights
    '''
    保存参数
    '''
    tl.files.save_npz(model.trainable_weights, name='dqn_model.npz')


def load_ckpt(model):  # load trained weights
    '''
    加载参数
    '''
    tl.files.load_and_assign_npz(name='dqn_model.npz', network=model)


if __name__ == '__main__':
    env = Maze()

    qnetwork = get_model([None, env.n_state])  # 定义inputshape[None,16]。16是state数量
    qnetwork.train()  # 调用tensorlayer的时候，需要标注这个模型是否可以训练。(再次吐槽tenorlayers...)
    train_weights = qnetwork.trainable_weights  # 模型的参数
    optimizer = tf.optimizers.SGD(learning_rate=0.1)  # 定义优化器

    t0 = time.time()
    for episode in range(num_episodes):
        env.current_state = env.reset()
        env.render()
        rAll = 0
        for step in range(200):
            allQ = qnetwork(np.asarray([to_one_hot(env.current_state, env.n_state)], dtype=np.float32)).numpy()

            action = np.argmax(allQ, axis=1)[0]

            if np.random.rand(1) < epsilon:
                action = np.random.choice(env.action_space)

            next_state, r, done = env.step(action)

            # 把new-state 放入，预测下一个state的**所有动作**的Q值。
            Q1 = qnetwork(np.asarray([to_one_hot(next_state, env.n_state)], dtype=np.float32)).numpy()

            maxQ1 = np.max(Q1)
            targetQ = allQ
            targetQ[0, action] = r + lambd * maxQ1

            with tf.GradientTape() as tape:
                _qvalues = qnetwork(np.asarray([to_one_hot(env.current_state, env.n_state)], dtype=np.float32))
                _loss = tl.cost.mean_squared_error(targetQ, _qvalues, is_mean=False)

            grad = tape.gradient(_loss, train_weights)
            optimizer.apply_gradients(zip(grad, train_weights))

            env.current_state = next_state
            rAll += r

            env.render()

            # 更新epsilon，让epsilon随着迭代次数增加而减少。
            # 目的就是智能体越来越少进行“探索”
            if done == True:
                epsilon = 1. / ((episode / 50) + 10)
                break
        running_reward = rAll if running_reward is None else running_reward * 0.99 + rAll * 0.01
        print('Episode: {}/{}  | Episode Reward: {:.4f} | Running Average Reward: {:.4f}  | Running Time: {:.4f}' \
              .format(episode, num_episodes, rAll, running_reward, time.time() - t0))


