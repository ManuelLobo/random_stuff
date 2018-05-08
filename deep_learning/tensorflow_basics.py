import tensorflow as tf

def hello_world():
    hello = tf.constant("Hello")
    world = tf.constant("World")
    print(hello)

    with tf.Session() as sess:
        result = sess.run(hello+world)

    print(result)

def constants():
    a = tf.constant(10)
    b = tf.constant(20)

    with tf.Session() as sess:
        result = sess.run(a + b)
    print(result)


def run_operations():
    constant = tf.constant(10)
    fill_matrix = tf.fill((4, 4), 10) # 4x4 with values 10
    zeros = tf.zeros((4, 4))
    ones = tf.ones((4, 4))
    randn = tf.random_normal((4, 4), mean=0, stddev=1.0)
    randu = tf.random_uniform((4, 4), minval=0, maxval=1)

    operations = [constant, fill_matrix, zeros, ones, randn, randu]

    with tf.Session() as sess:
        for operation in operations:
            result = sess.run(operation)
            print(result)


def matrix_multiplication():
    a = tf.constant([[1, 2], [3, 4]])  # 2x2
    a.get_shape()

    b = tf.constant([[10], [100]])  # 2x1
    b.get_shape()


    with tf.Session() as sess:
        mul = tf.matmul(a, b)
        result = sess.run(mul)
        print(result)
        print(mul.eval())
