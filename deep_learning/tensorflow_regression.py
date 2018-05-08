import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt


x_data = np.linspace(0.0, 10.0, 1000000)
noise = np.random.randn(len(x_data))

y_true = (0.5 * x_data) + 5 + noise # what we will try to figure out

x_dataframe = pd.DataFrame(data=x_data, columns=["X Data"])
y_dataframe = pd.DataFrame(data=y_true, columnes=["Y"])
data = pd.concat([x_dataframe, y_dataframe], axis=1)
print(data)
