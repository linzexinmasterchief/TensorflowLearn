import tensorflow as tf

import tensorflow.examples.tutorials.mnist.input_data as input_data
mnist = input_data.read_data_sets("./MNIST_data/", one_hot=True)

# x不是一个特定的值，而是一个占位符placeholder，
# 我们在TensorFlow运行计算时输入这个值。
# 我们希望能够输入任意数量的MNIST图像，
# 每一张图展平成784维的向量。我们用2维的浮点数张量来表示这些图，
# 这个张量的形状是[None，784 ]。
# （这里的None表示此张量的第一个维度可以是任何长度的。）
x = tf.placeholder(tf.float32, [None, 784])

# 我们赋予tf.Variable不同的初值来创建不同的Variable：
# 在这里，我们都用全为零的张量来初始化W和b。因为我们要学习W和b的值，
# 它们的初值可以随意设置。

# 注意，W的维度是[784，10]，
# 因为我们想要用784维的图片向量乘以它以得到一个10维的证据值向量，
# 每一位对应不同数字类。b的形状是[10]，所以我们可以直接把它加到输出上面。
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

# 用tf.matmul(​​X，W)表示x乘以W，对应之前等式里面的，
# 这里x是一个2维张量拥有多个输入。然后再加上b，
# 把和输入到tf.nn.softmax函数里面。
y = tf.nn.softmax(tf.matmul(x,W) + b)

# cross entropy place holder
y_ = tf.placeholder("float", [None,10])

# 首先，用 tf.log 计算 y 的每个元素的对数。
# 接下来，我们把 y_ 的每一个元素和 tf.log(y) 的对应元素相乘。
# 最后，用 tf.reduce_sum 计算张量的所有元素的总和。
# （注意，这里的交叉熵不仅仅用来衡量单一的一对预测和真实值，
# 而是所有100幅图片的交叉熵的总和。
# 对于100个数据点的预测表现比单一数据点的表现能更好地描述我们的模型的性能。
cross_entropy = -tf.reduce_sum(y_*tf.log(y))

# TensorFlow拥有一张描述你各个计算单元的图，
# 它可以自动地使用反向传播算法(backpropagation algorithm)
# 来有效地确定你的变量是如何影响你想要最小化的那个成本值的。
# 然后，TensorFlow会用你选择的优化算法来不断地修改变量以降低成本。
# 梯度下降算法
# 0.01的学习速率最小化交叉熵
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for i in range(1000):
	batch_xs, batch_ys = mnist.train.next_batch(100)
	sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

from matplotlib import pyplot as plt
from random import randint
num = randint(0, mnist.test.images.shape[0])
img = mnist.test.images[num]

classification = sess.run(tf.argmax(y, 1), feed_dict={x: [img]})
plt.imshow(img.reshape(28, 28), cmap=plt.cm.binary)
plt.show()
print('NN predicted', classification[0])

sess.close()