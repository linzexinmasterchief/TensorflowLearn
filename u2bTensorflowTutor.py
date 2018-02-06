import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("./MNIST_data/", one_hot=True)

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 10

def neural_network_model(data):

    hidden_1_layer = {
        'weights':tf.Variable(tf.random_normal([784, n_nodes_hl1])),
        'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))
    }

    hidden_2_layer = {
        'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
        'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))
    }

    hidden_3_layer = {
        'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
        'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))
    }

    output_layer = {
        'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
        'biases':tf.Variable(tf.random_normal([n_classes]))
    }

    # (input_data * weights) + bias

    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.add(tf.matmul(l3, output_layer['weights']), output_layer['biases'])
    return output

def train_neural_network():
    batch_size = 100

    x = tf.placeholder('float', [None, 784])
    y = tf.placeholder('float')

    prediction = neural_network_model(x)
    # prediction =
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))

    optimizer = tf.train.AdamOptimizer().minimize(cost)
    tf.add_to_collection('prediction', prediction)
    tf.add_to_collection('x', x)

    hm_epochs = 10

    with tf.Session() as sess:
        writer = tf.summary.FileWriter('./graphs', sess.graph)
        sess.run(tf.initialize_all_variables())

        for epoch in range(hm_epochs):
            epoch_loss = 0

            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c
            print('Epoch ', epoch, ' completed out of ', hm_epochs, " | lost: ", epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy', accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))

        saver = tf.train.Saver()
        saver.save(sess, 'model/model.ckpt')

    writer.close()

def appli():
    sess = tf.Session()
    # sess.run(tf.initialize_all_variables())
    new_saver = tf.train.import_meta_graph('model/model.ckpt.meta')
    new_saver.restore(sess, tf.train.latest_checkpoint('model/'))
    prediction = tf.get_collection('prediction')[0]
    x = tf.get_collection('x')[0]

    from matplotlib import pyplot as plt

    from PIL import Image
    import numpy as np
    img = np.array(Image.open('test4.bmp'))

    # from random import randint
    # num = randint(0, mnist.test.images.shape[0])
    # img = mnist.test.images[num]

    img = img.reshape(1, 784)
    feed_dict = {x: img}
    y_pred = sess.run(tf.argmax(prediction, 1), feed_dict=feed_dict)

    plt.imshow(img.reshape(28, 28), cmap=plt.cm.binary)
    plt.show()
    print('NN predicted', y_pred[0])


train_neural_network()
appli()