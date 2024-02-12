'''
NumPy Exercises 

'''

# Import NumPy as np
import numpy as np

# Create an array of 10 zeros 
np_arr = np.zeros(10)
print(np_arr)

# Create an array of 10 ones
np_arr = np.ones(10)
print(np_arr)

# Create an array of 10 fives
np_arr = np.linspace(5, 5, 10)
print(np_arr)

# Create an array of the integers from 10 to 50
np_arr = np.arange(10, 51)
print(np_arr)

# Create an array of all the even integers from 10 to 50
np_arr = np.arange(10, 51, 2)
print(np_arr)

# Create a 3x3 matrix with values ranging from 0 to 8
np_mat = np.arange(9)
np_mat = np_mat.reshape(3, 3)
print(np_mat)

# Create a 3x3 identity matrix
np_mat = np.eye(3)
print(np_mat)

# Use NumPy to generate a random number between 0 and 1
np_rand = np.random.rand(1)
print(np_rand)

# Use NumPy to generate an array of 25 random numbers sampled from a standard normal distribution
np_arr = np.random.randn(25)
print(np_arr)
      
# Create an array of 20 linearly spaced points between 0 and 1:)
np_arr = np.linspace(0, 1, 20)
print(np_arr)


###
### Numpy Indexing and Selection
###
### you will be given a matrix, and be asked to replicate the resulting matrix outputs:
###

mat = np.arange(1,26).reshape(5,5)

# array([[12, 13, 14, 15],
#        [17, 18, 19, 20],
#        [22, 23, 24, 25]])
print(mat[2:, 1:])

# 20
print(mat[3, 4])

# array([[ 2],
#        [ 7],
#        [12]])
print(mat[:3, 1:2])

# array([21, 22, 23, 24, 25])
print(mat[4, :])

# array([[16, 17, 18, 19, 20],
#        [21, 22, 23, 24, 25]])
print(mat[3:5, :])

# Get the sum of all the values in mat
print(mat.sum())

# Get the standard deviation of the values in mat
print(mat.std())

# Get the sum of all the columns in mat
print(mat.sum(axis = 0))