# 20173256 Kangmin Bae
# Python 3.5.2
# TensorFlow 1.4.0
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import os

def main():
    # Version check
    if not tf.__version__=='1.4.0':
        print('Tensorflow version mismatch. This code is based on tensorflow 1.4.0!!')
        raise AssertionError()

    # Parameters
    num_class = 10
    display_step = 100
    max_iter = 10000
    batch_size = 512#256
    starter_learning_rate = 0.1
    dropout_rate = 0.5
    weight_decay = 0.001
    batch_norm = True

    # Import data
    mnist = input_data.read_data_sets('./', one_hot=True)

    # Placeholder
    x = tf.placeholder(tf.float32, [None, 784],'x')
    y_label = tf.placeholder(tf.float32, [None, num_class],'y_label')
    dropout = tf.placeholder_with_default(False, shape=(), name='Dropout')

    # Build network
    input_x = tf.reshape(x, [-1, 28, 28, 1])
    tf.summary.image('Image', input_x, max_outputs=16)
    with tf.variable_scope('ResNet1'):
        with tf.variable_scope('conv1'):
            output11 = tf.layers.conv2d(input_x, 6, 3, 1, padding='same', activation=tf.nn.relu)
            if batch_norm==True:
                output11 = tf.layers.batch_normalization(output11)
        with tf.variable_scope('conv2'):
            output11 = tf.layers.conv2d(output11, 20, 3, 1, padding='same')
        with tf.variable_scope('conv3'):
            input_x = tf.layers.conv2d(input_x, 20, 3, 1, padding='same')
        output1 = output11 + input_x
        if batch_norm == True:
            output1 = tf.layers.batch_normalization(output1)

    with tf.variable_scope('ResNet2'):
        with tf.variable_scope('conv1'):
            output11 = tf.layers.conv2d(output1, 20, 3, 1, padding='same', activation=tf.nn.relu)
            if batch_norm==True:
                output11 = tf.layers.batch_normalization(output11)
        with tf.variable_scope('conv2'):
            output11 = tf.layers.conv2d(output11, 20, 3, 1, padding='same')
        '''with tf.variable_scope('conv3'):
            output1 = tf.layers.conv2d(output1, 20, 3, 1, padding='same')'''
        output1 = output11 + output1
        if batch_norm == True:
            output1 = tf.layers.batch_normalization(output1)

    with tf.variable_scope('Maxpooling1'):
        output2 = tf.layers.max_pooling2d(output1, 2, 2)
    with tf.variable_scope('conv2'):
        output3 = tf.layers.conv2d(output2, 16, 5, 1, padding='same', activation=tf.nn.relu)
        if batch_norm == True:
            output3 = tf.layers.batch_normalization(output3)
    with tf.variable_scope('Maxpooling2'):
        output4 = tf.layers.max_pooling2d(output3, 2, 2)
    with tf.variable_scope('fc1'):
        input5 = tf.layers.flatten(output4)
        output5 = tf.layers.dense(input5, 128, activation=tf.nn.relu)
        if batch_norm == True:
            output5 = tf.layers.batch_normalization(output5)
        output5 = tf.layers.dropout(output5, dropout_rate, training=dropout)
    with tf.variable_scope('fc2'):
        output6 = tf.layers.dense(output5, 84, activation=tf.nn.relu)
        if batch_norm == True:
            output6 = tf.layers.batch_normalization(output6)
        output6 = tf.layers.dropout(output6, dropout_rate, training=dropout)
    with tf.variable_scope('fc3'):
        output7 = tf.layers.dense(output6, num_class)
        if batch_norm == True:
            output7 = tf.layers.batch_normalization(output7)
    output = tf.nn.softmax(output7)

    # Define loss function(Cross entropy)
    with tf.variable_scope('Cross_entropy'):
        #softmax_layer = tf.exp(output)/tf.reshape(tf.reduce_sum(tf.exp(output), 1),[-1, 1])
        #cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_label, logits=output) #+ weight_decay * tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_label * tf.log(output + 0.0001), reduction_indices=[1])) + weight_decay*tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
        tf.summary.scalar('Cross-entropy loss', cross_entropy)

    # Define optimizer
    with tf.variable_scope('Optimizer'):
        global_step = tf.Variable(0, trainable=False, name='global_step')
        learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, max_iter, 0.97, staircase=True)
        optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.5, name='Optimizer')\
            .minimize(cross_entropy, global_step=global_step)

    # Define accuracy
    with tf.variable_scope('Accuracy'):
        correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y_label, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('Accuracy', accuracy)

    # Summary writer
    merged = tf.summary.merge_all()
    i = 1
    while 1:
        if not os.path.exists('MNIST/{}'.format(i)):
            break
        i += 1

    writer_train = tf.summary.FileWriter('MNIST/{}/train'.format(i))
    writer_test = tf.summary.FileWriter('MNIST/{}/validation'.format(i))

    # Define Session & initializer
    sess = tf.Session()
    init = tf.global_variables_initializer()
    writer_train.add_graph(sess.graph)

    # Train
    sess.run(init)
    for step in range(max_iter):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        sess.run(optimizer, feed_dict={x: batch_xs, y_label: batch_ys, dropout: True})
        if step%display_step == 0:
            acc, loss, result = sess.run([accuracy, cross_entropy, merged], feed_dict={x: batch_xs, y_label: batch_ys})
            writer_train.add_summary(result, step)
            print('Iteration: {}/{} Accuracy = {:.2f}% Loss = {:.2f}'.format(step, max_iter, 100 * acc, loss))
            acc, loss, result = sess.run([accuracy, cross_entropy, merged], feed_dict={x: mnist.validation.images, y_label: mnist.validation.labels})
            writer_test.add_summary(result, step)
            print('Validation accuracy = {:.2f}%, Validation loss = {:.2f}'.format(100 * acc, loss))

    print('Optimization finished')

    # Test network
    acc, loss = sess.run([accuracy, cross_entropy], feed_dict={x: mnist.test.images, y_label: mnist.test.labels})
    print('Test accuracy = {:.2f}%, Test loss = {:.2f}'.format(100*acc, loss))

if __name__ == '__main__':
    main()