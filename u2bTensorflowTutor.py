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
    # define default graph
    # set the size of one batch to 100
    batch_size = 100
    # input images
    x = tf.placeholder('float', [None, 784])
    # labels of the images
    y = tf.placeholder('float')

    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    # the amount of time training the network
    hm_epochs = 10

    with tf.Session() as sess:
        with tf.device("/cpu:0"):
            # create writer for tensorboard
            writer = tf.summary.FileWriter('./graphs', sess.graph)
            sess.run(tf.initialize_all_variables())

            for epoch in range(hm_epochs):
                epoch_loss = 0

                for _ in range(int(mnist.train.num_examples/batch_size)):
                    # extract batch_size amount of data (image) and label from the data size
                    epoch_x, epoch_y = mnist.train.next_batch(batch_size)

                    # runs the default graph above, x and y are the pre-defined placeholders
                    _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                    epoch_loss += c
                print('Epoch ', epoch, ' completed out of ', hm_epochs, " | lost: ", epoch_loss)

            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
            print('Accuracy', accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))

            saver = tf.train.Saver()
            saver.save(sess, 'model/model.ckpt')

    writer.close()

def apply():
    # creates a new session
    sess = tf.Session()
    # locate the saved session
    new_saver = tf.train.import_meta_graph('model/model.ckpt.meta')
    # restore the saved session (the session contains all the data we need to restore the trained model
    new_saver.restore(sess, tf.train.latest_checkpoint('model/'))

    # creates another default graph for predicting
    # creates a input placeholder (since we no longer has the label y)
    x = tf.placeholder('float', [None, 784])
    # use the neural network model structure
    prediction = neural_network_model(x)
    # initialize x
    sess.run(tf.initialize_all_variables())

    # code to use your own picture as input x
    from PIL import Image
    import numpy as np
    img = np.array(Image.open('test4.bmp'))

    ## code to use random pictures from mnist as input x
    # from random import randint
    # num = randint(0, mnist.test.images.shape[0])
    # img = mnist.test.images[num]

    # code to display the picture
    from matplotlib import pyplot as plt
    img = img.reshape(1, 784)
    plt.imshow(img.reshape(28, 28), cmap=plt.cm.binary)
    plt.show()

    # finally feed the image and get the prediction
    feed_dict = {x: img}
    y_pred = sess.run(tf.argmax(prediction, 1), feed_dict=feed_dict)
    print('NN predicted', y_pred[0])


train_neural_network()
apply()