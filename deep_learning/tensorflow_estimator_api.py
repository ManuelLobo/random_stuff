from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import logging


logging.basicConfig(level=logging.DEBUG,
  format="[%(asctime)s]"
         "[%(levelname)s] %(message)s")

######################## tensorflow_regression.py ###########################
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

    #print(model_m, model_b)

############################################################################

feature_columns = [tf.feature_column.numeric_column("x", shape=[1])]
estimator = tf.estimator.LinearRegressor(feature_columns=feature_columns)
#print(estimator)

x_train, x_eval, y_train, y_eval = train_test_split(x_data, y_true,
                                            test_size=0.3, random_state=101)



# estimator inputs - input function that acts like a feed dictionary and batch
# size indicator all at once.
input_func = tf.estimator.inputs.numpy_input_fn({"x":x_train}, y_train,
                                                batch_size=8, num_epochs=None,
                                                shuffle=True)

train_input_func = tf.estimator.inputs.numpy_input_fn({"x":x_train}, y_train,
                                                batch_size=1000, num_epochs=None,
                                                shuffle=False)

eval_input_func = tf.estimator.inputs.numpy_input_fn({"x":x_eval}, y_eval,
                                                batch_size=1000, num_epochs=None,
                                                shuffle=False)


estimator.train(input_fn=input_func, steps=1000)
train_metrics = estimator.evaluate(input_fn=train_input_func, steps=1000)
eval_metrics = estimator.evauluate(input_fn=eval_input_func, steps=1000)





















#
