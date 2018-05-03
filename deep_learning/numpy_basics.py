import numpy as np


# transform list into numpy array
my_list = [1, 2, 3]
arr = np.array(my_list)
arr_type = type(arr) # numpy.ndarray

print("numpy array", arr, arr_type)

# range numpy array
range_arr = np.arange(0, 10)
print("range array", range_arr)

range_arr = np.arange(0, 10, 2)
print("step size 2, range array", range_arr)


# Zero values
zeros_arr = np.zeros(5)
zeros_mat = np.zeros((3, 5))
print("Arrays of zeros", zeros_arr)
print("Matrix of zeros", zeros_mat)

# One values
ones_arr = np.ones(5)
ones_mat = np.ones((3, 5))
print("Arrays of ones", ones_arr)
print("Matrix of ones", ones_mat)


# linear spaced values
lin_space = np.linspace(0, 11, 10) # 10 evenly spaced points between 0 and 11
print("Linear spaced points", lin_space)


# random values
random_int = np.random.randint(0, 10)
print("Random integer", random_int)

random_int = np.random.randint(0, 10, (3, 3)) # matrix 3x3 random values between 0 and 10
print("Random integer matrix", random_int)


# Setting random seeds
np.random.seed(101)
a = np.random.randint(0, 100, 10)
b = np.random.randint(0, 100, 10)
print("random seed value comparison: ")



# Operations
arr = np.random.randint(0, 100, 10)
print(f"Array: {arr}")
print(f"Array max value: {arr.max()}")
print(f"Array min value: {arr.min()}")
print(f"Array mean value: {arr.mean()}")
print(f"Array max value index: {arr.argmax()}")
print(f"Reshape Array (2x5): {arr.reshape(2,5)}")

# create matrix from range array:
mat = np.arange(0, 100).reshape(10, 10) # 10x10 matrix of values between 0 and 100

# Get elements
print(f"Get second element of first row: {mat[0,1]}")
print(f"Get third element of fifth row: {mat[4,3]}")
print(f"Slice - All the rows from first row: {mat[:, 0]}")
print(f"Slice - All the columns from fifth row: {mat[5, :]}")
print(f"Slice - first 3 columns from first 3 rows: {mat[0:3, 0:3]}")
print(f"Return a matrix of bool values when the values are > 50: {mat > 50}")

my_filter = mat > 50
print(mat[my_filter])
print(mat[mat>50])




#
