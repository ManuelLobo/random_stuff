import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST/", one_hot=True)


# Weight initialization function
def init_weights(shape):
    init_random_dist = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(init_random_dist)


# Bias initialization function
def init_bias(shape):
    init_bias_vals = tf.constant(0.1, shape=shape)
    return init_bias_vals


# 2D convulution Function
# Tensorflow already has a function that creates a 2D convulutional.
# It takes a tensor and a kernel (filter tensor) and performs a convulition
# on it, depending on the stride and padding
# We're going to create a wrapper around the existing funtion.

def conv2d(x, W):
    """
    x is the input tensor: x = [batch, Height, Width, Channels]
    w is the kernal: w = [filter height, filter width, channels in, channels out ]
    padding="SAME" means zero padding
    """

    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")


# Pooling - also a wrapper
def max_pool_2by2(x):
    """ Grabs the maximum value in that stride.
    x is the input tensor.
    ksize is the size of the window for each dimension in the input tensor.
        in this case, the ksize will be [1,2,2,1]:
            1 - one image?
            2,2 - the size of the pool
            1 - the number of channels
    strides is the stride of the window for each dimension. We use the same values
    for stride. [1,2,2,1]

    """

    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                          padding="SAME")


# Now we want to create the functions where we actually create the layers
# Convulutional layer
def convulutional_layer(input_x, shape):
    W = init_weights(shape)
    b = init_bias([shape[3]])
    return tf.nn.relu(conv2d(input_x, W) + b)


# Then we do a Normal Layer - or Fully Connected Layer
def normal_full_layer(input_layer, size):
    input_size = int(input_layer.get_shape()[1])
    W = init_weights([input_size, size])
    b = init_bias([size])

    return tf.matmul(input_layer, W) + b


# Create the network

### Placeholders ###
x = tf.placeholder(tf.float32, shape=[None, 784]) #None is the size of that batch
y_true = tf.placeholder(tf.float32, shape=[None, 10])

### Layers ###
# Image Layer (essentially our input image)
x_image = tf.reshape(x, [-1, 28, 28, 1])
# first convlulutional layer
convo_1 = convulutional_layer(x_image, shape=[5,5,1,32]) # [h,w,c, features] #it's going to compute 32 feature for each 5by5 (the 1 is the channels)
# first pooling layer
convo_1_pooling = max_pool_2by2(convo_1)

# second convlulutional layer
convo_2 = convulutional_layer(convo_1_pooling, shape=[5,5,32,64])
convo_2_pooling = max_pool_2by2(convo_2)

# flatten the resulting layer
convo_2_flat = tf.reshape(convo_2_pooling, [-1, 7*7*64])

full_layer_one = tf.nn.relu(normal_full_layer(convo_2_flat, 1024))


# Dropout
hold_prob = tf.placeholder(tf.float32)
full_one_dropout = tf.nn.dropout(full_layer_one, keep_prob=hold_prob)

y_pred = normal_full_layer(full_one_dropout, 10)


# Now we just need to craete the optimizer, loss function, initialize the variables and create the session.

# Loss function
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    labels=y_true, logits=y_pred))

# optimizier
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
train = optimizer.minimize(cross_entropy)

# initialize variables
init = tf.global_variables_initializer()

steps = 5000
with tf.Session() as sess:
    sess.run(init)
    for step in range(steps):
        x_batch, y_batch = mnist.train.next_batch(10)
        sess.run(train, feed_dict={x: x_batch, y_true: y_batch, hold_prob: 0.5})

        if step % 100 == 0:
            print("ON STEP: {}".format(step))
            print("ACCURACY: ")
            matches = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
            acc = tf.reduce_mean(tf.cast(matches, tf.float32))
            print(sess.run(acc, feed_dict={x: mnist.test.images,
                                           y_true: mnist.test.labels,
                                           hold_prob: 1.0}))  # dont drop
            print("\n")





#
