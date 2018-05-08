import tensorflow as tf
import numpy as np
#import matplotlib.pyplot as plt
np.random.seed(101)
tf.set_random_seed(101)


def graphs():
    n1 = tf.constant(1)
    n2 = tf.constant(2)
    n3 = n1 + n2

    with tf.Session() as sess:
        result = sess.run(n3)

    print(result)
    print(tf.get_default_graph())


    g = tf.Graph()
    graph_one = tf.get_default_graph()
    graph_two = tf.Graph()

    with graph_two.as_default():
        print(graph_two is tf.get_default_graph())
    print(graph_two is tf.get_default_graph())




def placeholder():
    with tf.Session() as sess:
        my_tensor = tf.random_uniform((4, 4), 0, 1)  # min val 0 max 1
        variable = tf.Variable(initial_value=my_tensor)
        init = tf.global_variables_initializer()
        sess.run(init)
        array = sess.run(variable)
        print(array)


        ph = tf.placeholder(tf.float32, shape=(None, 5)) # the none represents the number of samples being fed in batches


def operations():


    rand_a = np.random.uniform(0, 100, (5, 5))
    rand_b = np.random.uniform(0, 100, (5, 1))

    a = tf.placeholder(tf.float32)
    b = tf.placeholder(tf.float32)

    add_op = a + b
    mul_op = a * b

    with tf.Session() as sess:
        add_result = sess.run(add_op, feed_dict={a: rand_a, b: rand_b})
        print(add_result)

        mult_result = sess.run(mul_op, feed_dict={a: rand_a, b: rand_b})
        print(mult_result)



def neural_network():
    n_features = 10
    n_dense_neurons = 3

    x = tf.placeholder(tf.float32, shape=(None, n_features))
    W = tf.Variable(tf.random_normal([n_features, n_dense_neurons]))
    b = tf.Variable(tf.ones([n_dense_neurons]))

    xW = tf.matmul(x, W)
    z = tf.add(xW, b)

    a = tf.sigmoid(z)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        output_layer = sess.run(a, feed_dict={x:np.random.random([1, n_features])})

        print(output_layer)

#neural_network()


def simple_regression():
    x_data = np.linspace(0, 10, 10) + np.random.uniform(-1.5, 1.5, 10)
    y_label = np.linspace(0, 10, 10) + np.random.uniform(-1.5, 1.5, 10)

    m = tf.Variable(0.44) # random value
    b = tf.Variable(0.87)

    error = 0
    for x, y in zip(x_data, y_label):
        y_hat = m*x + b
        error += (y - y_hat)**2

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train = optimizer.minimize(error)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        training_steps = 1000
        for i in range(training_steps):
            sess.run(train)

        final_slope, final_intercept = sess.run([m, b])

    # plot results
    x_test = np.linspace(-1, 11, 10)
    y_pred_plot = final_slope*x_test + final_intercept
    #plt.plot(x_test, y_pred_plot, "r")
    #plt.plot(x_data, y_label, "*")
    #plt.show()

simple_regression()

#
