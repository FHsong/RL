import tensorflow as tf
import numpy as np
from sklearn import datasets
from pandas import DataFrame
import pandas as pd

with tf.GradientTape() as tape:
    w = tf.Variable(tf.constant(3.0))
    loss = tf.pow(w, 2)
grad = tape.gradient(loss, w)
# print(grad)

classes = 3
labels = tf.constant([1,0,2])
output = tf.one_hot(labels, depth=classes)
# print(output)

y = tf.constant([1.01, 2.01, -0.66])
y_pro = tf.nn.softmax(y)
# print(y_pro)

test = np.array([[1,2,3], [2,3,4], [5,4,3], [8,7,2]])
# print(test)
# print(tf.argmax(test, axis=0))
# print(tf.argmax(test, axis=1))

#鸢尾花数据集（Iris）
x_data = datasets.load_iris().data
y_data = datasets.load_iris().target
# print('x_data: ', x_data.shape, '\n', x_data)
# print('y_data: ', y_data)

x_data = DataFrame(x_data, columns=['花萼长度', '花萼宽度', '花瓣长度', '花瓣宽度'])
pd.set_option('display.unicode.east_asian_width', True)
x_data['类别'] = y_data
# print(x_data)



