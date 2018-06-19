import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import logging


logging.basicConfig(level=logging.DEBUG,
  format="[%(asctime)s]"
         "[%(levelname)s] %(message)s")


x_data = np.linspace(0.0, 10.0, 1000000)
noise = np.random.randn(len(x_data))

y_true = (0.5 * x_data) + 5 + noise # what we will try to figure out

x_dataframe = pd.DataFrame(data=x_data, columns=["X Data"])
y_dataframe = pd.DataFrame(data=y_true, columns=["Y"])
data = pd.concat([x_dataframe, y_dataframe], axis=1)

# sample of Data
data.sample(n=250).plot(kind='scatter', x="X Data", y="Y")
#plt.show()

batch_size = 8
m = tf.Variable(0.81) # random variables
b = tf.Variable(0.17)

x_placeholder = tf.placeholder(tf.float32, [batch_size])
y_placeholder = tf.placeholder(tf.float32, [batch_size])

y_model = m * x_placeholder + b

error = tf.reduce_sum(tf.square(y_placeholder - y_model))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train = optimizer.minimize(error)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    batches = 1000
    for i in range(batches):
        rand_ind = np.random.randint(len(x_data), size=batch_size)
        feed = {x_placeholder: x_data[rand_ind],
                y_placeholder: y_true[rand_ind]}
        sess.run(train, feed_dict=feed)

    model_m, model_b = sess.run([m, b])

    print(model_m, model_b)


# plotting the data and result
y_hat = x_data*model_m + model_b
my_data.sample(250).plot(kind="scatter", x="X Data", y="Y")
plt.plot(x_data, y_hat, "r").show()
plt.show()
