import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%matplotlib inline

x = np.arange(0, 10)
print(x)

y = x**2

# PLOTS
# plt.plot(x, y)
# plt.show()
#
# plt.plot(x, y, "*")
# plt.show()
#
# plt.plot(x, y, "red")
# plt.show()


# plt.plot(x, y, "r--")
# plt.xlim(0, 4)
# plt.ylim(0, 10)
# plt.title("TITLE")
# plt.xlabel("X LABEL")
# plt.ylabel("Y LABEL")
# plt.show()


# Image
#mat = np.arange(0, 100).reshape(10, 10)
mat = np.random.randint(0, 1000, (10, 10))
plt.imshow(mat, cmap="RdYlGn")
plt.colorbar()
plt.show()


# Plot CSV file

df = pd.read_csv("csv_file")
df.plot(x="column_1", y="column_2", kind="scatter")




#
