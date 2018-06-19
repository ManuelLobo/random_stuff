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

# shuffle is true is because were going to used this train input function
# for evaluation against the test input function.
train_input_func = tf.estimator.inputs.numpy_input_fn({"x":x_train}, y_train,
                                                batch_size=1000, num_epochs=None,
                                                shuffle=False)

eval_input_func = tf.estimator.inputs.numpy_input_fn({"x":x_eval}, y_eval,
                                                batch_size=1000, num_epochs=None,
                                                shuffle=False)

# First we train the estimator, and we'll do it for 1000 steps
estimator.train(input_fn=input_func, steps=1000)

# Get the metrics for the sets
train_metrics = estimator.evaluate(input_fn=train_input_func, steps=1000)
eval_metrics = estimator.evaluate(input_fn=eval_input_func, steps=1000)

print("** Training Data Metrics **")
print(train_metrics)

print("** Training Eval Metrics **")
print(eval_metrics)

# These metrics are a good indicator to see if our model is overfitting to the
# training data. A good indicator for the model overfitting the training data
# is when we have a really low loss on the training data but a really big loss
# on the eval data. We want them to be as close as possible to eachother.
# In general they should be somewhat similar.


# Now, to predict new values, we create a new input function and pass some new
# data to predict.
new_data = np.linspace(0, 10, 10)
input_fn_predict = tf.estimator.inputs.numpy_input_fn({'x': new_data},
                                                      shuffle=False)
predictions = []
for pred in estimator.predict(input_fn=input_fn_predict):
    predictions.append(pred["predictions"])

#data.sample(n=250).plot(kind="scatter", x="X Data", y="Y")
plt.plot(new_data, predictions, "r")
plt.show()




#
