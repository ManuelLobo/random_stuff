import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets("MNIST/", one_hot=True)

#### Data details ####
# Each image is 28x28

# There are 55000 examples for training
print(mnist.train.num_examples)

# There are 10000 examples for testing
print(mnist.test.num_examples)

# The training set has shape (55000, 784)
print(mnist.train.images.shape)

# We can access a single image
one_image = mnist.train.images[1]

# We can flatten the image to 2D (they're in 1D: 1x784)
single_image = one_image.reshape(28, 28)
#plt.imshow(single_image)
#plt.show()

# Grayscale
#plt.imshow(single_image, cmap="gist_gray")
#plt.show()

# The data is already normalized (min = 0, max = 1)
single_image.min()
single_image.max()

#### End Data Details ####


### Create the model ###
# Softmax approach

# Create placeholders, variables, graph operations, loss function, optimizer
# and then create the session and run all of it.

# Placeholders
x = tf.placeholder(tf.float32, shape=[None, 784]) # Shape None because batch size?

# Weights
W = tf.Variable(tf.zeros([784, 10])) # 10 possible labels

# Bias
b = tf.Variable(tf.zeros([10]))

# Create graph operation
y = tf.matmul(x, W) + b

# Loss Function
y_true = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    labels=y_true, logits=y))


# optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5)
train = optimizer.minimize(cross_entropy)

# Create session
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for step in range(100000):
        batch_x, batch_y = mnist.train.next_batch(100)
        sess.run(train, feed_dict={x: batch_x, y_true: batch_y})


    # Evaluate the model
    # tf.argmax(y, 1) # label with the highest probability
    # compare with the true value tf.argmax(y_true, 1)

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_true, 1))
    # returns a list like [True, False....] and we transform to 0s and 1s

    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    print(sess.run(acc, feed_dict={x: mnist.test.images,
                                   y_true: mnist.test.labels}))






#
