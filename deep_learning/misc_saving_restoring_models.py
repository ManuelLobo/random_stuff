import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

np.random.seed(101)
tf.set_random_seed(101)

x_data = np.linspace(0,10,10) + np.random.uniform(-1.5,1.5,10)
y_label = np.linspace(0,10,10) + np.random.uniform(-1.5,1.5,10)
m = tf.Variable(0.39)
b = tf.Variable(0.2)
error = tf.reduce_mean(y_label - (m*x_data+b))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train = optimizer.minimize(error)

init = tf.global_variables_initializer()


# Save the model
saver = tf.train.Saver()

# Create session, run and save
with tf.Session() as sess:

    sess.run(init)

    epochs = 100

    for i in range(epochs):

        sess.run(train)

    # Fetch Back Results
    final_slope , final_intercept = sess.run([m,b])

    # ONCE YOU ARE DONE
    # GO AHEAD AND SAVE IT!
    # Make sure to provide a directory for it to make or go to. May get errors otherwise
    #saver.save(sess,'models/my_first_model.ckpt')
    saver.save(sess,'new_models/my_second_model.ckpt')



# Load the model
with tf.Session() as sess:

    # Restore the model
    saver.restore(sess,'new_models/my_second_model.ckpt')


    # Fetch Back Results
    restored_slope , restored_intercept = sess.run([m,b])
