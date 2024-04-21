# this is gonna cover some of the most fundamental concepts of tensors using tensorFlow
# this is gonna cover
# -intro to tensors
# -getting informations from tensors
# -manipulating tensors
# -tensors and Numpy
# -using @tf.functions ( a way to speed up your regular python functions
# -using gpu's with tensorFlow (or TPU)

# * intro to Tensors!!
# * import TensorFlow
import tensorflow as tf
# print(tf.__version__)

# * create tensors with tf.constant()
scalar = tf.constant(7)
# print(scalar)

# * check number of dimensions of a tensor (ndim stand for number of dimensions)
# print(scalar.ndim)

# * Create a vector
vector = tf.constant([10, 10])
# print(vector)
# print(vector.ndim)

# * create a matrix (has more than one dimension)
matrix = tf.constant([[10, 7], [7, 10]])
# print(matrix)
# print(matrix.ndim)

# * create another matrix
another_matrix = tf.constant([[3., 7.],
                              [9., 7.],
                              [1., 9.]], dtype=tf.float16)
# print(another_matrix)

# * what the number dimensions of another matrix
# print(another_matrix.ndim)

# * lets create a tensor
tensor = tf.constant([[[1, 2, 3, ], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]], [[13, 14, 15], [16, 17, 18]]])

# print(tensor)
# print(tensor.ndim)

# * what weve created so far
# * scalar: a single number
# * vector a number with directions ( wind speed and directions)
# * matrix a 2 dimensional array of numbers
# * tensor: an n dimensional array of number


# * create the same tenosr with tf.Variable() as above
# changeable _

changeable_tensor = tf.Variable([10, 7])
unchangeable_tensor = tf.constant([10, 7])
# print(changeable_tensor, unchangeable_tensor)


# * lets try to change one of the elements in our changeable tensor
# changeable_tensor[0] = 7
# * fails it needs to be assigned

# * how about we try .assign
# changeable_tensor[0].assign(7)
# print(changeable_tensor)

# * change out unchangeable
# unchangeable_tensor[0].assign(7)
# * fails it is a constant like const in javascript

# * Creating Random Tensors
# * random tensors ware tensors of arbitrary size which contains random numbers

# * create 2 random but the same tensors

random_tensor = tf.random.Generator.from_seed(42)  # set seed for reproducibility
random_tensor = random_tensor.normal(shape=(3, 3))
# print(random_tensor)
random_2 = tf.random.Generator.from_seed(42)
random_2 = random_2.normal(shape=(3, 3))
# * look at uniform distribution for yourself

# * are they equal to each other
# * print(random_tensor == random_2)

# *  Shuffling the order of the elements in a tensor
# * shuffle your data ut doesn't get stuck on the order

not_shuffled = tf.constant([[10, 7],
                            [3, 4],
                            [7, 5]])

shuffled = tf.random.shuffle(not_shuffled)
# print(shuffled)
tf.random.set_seed(42)
random_shuffled = tf.random.shuffle(not_shuffled, seed=42)

# print(random_shuffled)


# print(tf.random.set_seed(42)) # global level random seed
# print(tf.random.shuffle(not_shuffled, seed=42)) # operation level random seed

# * it looks like if we want our shuffled tensors to be in the same order we've got to use
# * global level random seed as well as the operational level random seed

# * other ways to make tensors
# * create a tensor of all ones
tf.ones([10,7])
# print(tf.ones([10,7]))

# * create a tensor of all zeros
tf.zeros(shape=(3,4))
# print(tf.zeros(shape=(3,4)))

# * you can also turn numPy arrays into tensors
# * the main difference between numPy arrays and tensorflow is that tensors can be run on a GPU ( which is much faster fo number computing

import numpy as np
numpy_A = np.arange(1,25, dtype=np.int32) # create a numpy array between 1 and 25
# print(numpy_A)

A = tf.constant(numpy_A, shape=(2,3,4)) # tensors because it has more than 1 dimension
# print(A)

B = tf.constant(numpy_A)
# print(B)

# X = tf.constant (some_matrix) # capital for matrix or tensor

# y = tf.constant(vector) # non - capital for vector

# * getting information from tensors

# important tensor terms
# * shape - the length of each of the dimensions
# *rank - the number of tensor dimensions A scalar has rank 0 a vector is rank 1
# * axis or dimension - a particular dimension of a tensor
# * size -  the total number of items in the tensor

# * create a rank 4 tensor  meaning 4 dimensions

rank_4_tensor = tf.zeros(shape=[2,3,4,5])
# print(rank_4_tensor)
# print(rank_4_tensor.shape[0]) #0 axis so 2 i believe
# print(rank_4_tensor.shape)
# print(rank_4_tensor)
# print(rank_4_tensor.ndim) # number of dimension rank
# print(tf.size(rank_4_tensor)) # size

# * get various attributes of our tensors
# print("Data type of every element: ", rank_4_tensor.dtype)
# print("number of dimensions (rank): ", rank_4_tensor.ndim)
# print("Shape of Tensor: ", rank_4_tensor.shape)
# print("Elemens along the 0 axis: ", rank_4_tensor.shape[0])
# print("Elements along the last axis: ", rank_4_tensor.shape[-1])
# print("Total number of elements in our tensor: ", tf.size(rank_4_tensor).numpy)
# print("Total number of elements in our tensor using numpy: ", tf.size(rank_4_tensor).numpy())

# * Tensors can be indexed just like python lists
# *  get the first 2 elements of each dimension
some_list = [1,2,3,4]
# print(some_list[:2])
# print(rank_4_tensor[:2,:2,:2,:2])

# * Get the first element from each dimension from each index for the final one
# print(rank_4_tensor[:1, :1, :, :1])
#  * create a rank 2 tensor (2dimensions)
rank_2_tensor = tf.ones(shape=[4,8])
rank_2_tensor_copy = tf.constant([[10, 7], [3, 4]])
# print(rank_2_tensor.shape)
# print(rank_2_tensor.ndim)

# * Get the last item of each of our rank 2 tensor
# print(rank_2_tensor[:-1,:-1])
# print(rank_2_tensor_copy[:, -1,])

# * add in extra dimension to our rank 2 tensor
rank_3_tensor = rank_2_tensor_copy[..., tf.newaxis]
# *  3 dots means on every axis before the last one and new axis then add a new on on the end
# print(rank_3_tensor)

# * alternative to tf.newaxis
# print(tf.expand_dims(rank_2_tensor_copy, axis=-1)) # -1 means expand the final axis

# print(tf.expand_dims(rank_2_tensor_copy, axis=0))

# * manipulating Tensors (tensors operation)

# * basic Operations
# * you can add values to a tensor using the addition operator
tensorMath = tf.constant([[10, 7], [3, 4]])
# print(tensorMath + 10)
# print(tensorMath)
# * multiplication also work
# print(tensorMath * 10)
# subtractions
# print(tensorMath -10)
# we can use the tensorflow built-in function too
# print(tf.multiply(tensorMath, 10))
# print(tf.add(tensorMath, 10))
# print(tf.subtract(tensorMath, 10))
# print(tf.divide(tensorMath, 10))

# * Matrix multiplication
# * in machine learning, matrix multiplication is one of the most common tensor operations
# * commonly reffered to as the element y operations  which means going through and applying it to all elements in the matrix

# * matrix multiplication in tensorflow
# print(tf.matmul(tensorMath, tensorMath))

# print(tensorMath * tensorMath)

tensorPractice = tf.constant([[1, 2, 5],
                              [7, 2, 1],
                              [3, 3, 3]])

tensorPractice2 = tf.constant([[3, 5],
                               [6, 7],
                               [1, 8]])

tensorPractice3 = tf.constant([[1,2],
                               [2, 3],
                               [4, 5]])

tensorPractice4 = tf.constant([[1,3,5,], [2,4,6], [2,56,9]])

# print(tf.matmul(tensorPractice, tensorPractice2))

# * matrix multiplication with Python operator '@'
# print(tensorMath @ tensorMath)

# * dosent work that well on tensors of different shapes

# * try to matrix multiply tensors of same shape
# print(tensorPractice2, tensorPractice3)

# print(tensorPractice2 @ tensorPractice3)
# print(tf.matmul(tensorPractice3, tensorPractice2))
# * neither of these works

# * rules of tensor multiplication - the 2 rules are  the tensors (or matrices) need to fulfill if were going to
#  * matrix multiply them:
# 1) the inner dimensions must match
# 2) the resulting matrix has the shape of the inner dimensions

# * the inner dimensions must match means that is they are both (3,2) (3,2) it means that the 2 -
# * and 3  are next to each other and they dont match so it wont compile
# print(tensorPractice4 @ tensorPractice)
# print(tf.matmul(tensorPractice, tensorPractice4))

# print(tf.reshape(tensorPractice3, shape=(2, 3)))
# print(tensorPractice3)
# print(tf.matmul(tensorPractice2, tf.reshape(tensorPractice3, shape=(2, 3))))
# * you can reshape a tensor when you call it and you can make it work with how you reshape it
# print(tf.matmul(tf.reshape(tensorPractice2, shape=(2, 3)), tensorPractice3))
# * trying to change the shape of tensorPractice 2 instead of tensor practice 3

# * when you multiply the inner dimensions must be the same,  but it will take the shape of the
# * outerDimensions so the outer one when we changed tensorpractice2 we got a (2,2) because those are the outer
# dimensions

#can do the same with transpose
# print(tf.transpose(tensorPractice2))
# print(tf.reshape(tensorPractice2, shape=(2,3)))

# * difference between transpose and reshape is that transpose grabs them by their axis's- also flips the axis
#  *and reshape grabs them by the order they are put in already
# * try matrix multiplication with transpose rather than reshape
# print(tf.matmul(tf.transpose(tensorPractice2), tensorPractice3))

# * The dot Product
# * matrix multiplciation is also referred to as the dot product
# * you can perform matrix multiplication useing:
# *tf.matmul() && ~ tf.tensordot()

# * Perform the dot product on X and Y (requires X or Y to be transposed)
# print(tf.tensordot(tf.transpost(tensorPractice2), tensorPractice3, axes=1))
# * perform matrix multiplication between x and y (transposed)
# print(tf.matmul(tensorPractice2, tf.transpose(tensorPractice3)))

# * perform matrix multiplication between X and Y (reshaped)
# print(tf.matmul(tensorPractice2, tf.reshape(tensorPractice3, shape=(2,3))))

# * check the values of Y, reshape Y and transposed Y

# print("normal Y: ")
# print(tensorPractice3, "\n")  # "\n is for newline
# print("Y  reshaped to (2, 3) : ")
# print(tf.reshape(tensorPractice3, shape=(2, 3)), "\n")
# print("Y transposed: ")
# print(tf.transpose(tensorPractice3))

# * generally, when performing matrix multiplication on two tensors and one of the axes doesn't line up, you will
# * transpose (rather than reshape) one of the tensors to satisfy the matrix multiplication

#Changing the datatype of a tensor
# create a new tensor with default datatype (float32)

C = tf.constant([1.7, 7.4])
# print(C.dtype)

D = tf.constant([7, 10])
# print(D.dtype)

# * Change from float32 to float16 this is called reduced precision

E = tf.cast(C, dtype=tf.float16)
# print(E, E.dtype)

# * change from int32 to float32
F = tf.cast(D, dtype=tf.float32)
# print(F)

F_float16 = tf.cast(F, dtype=tf.float16)
# print(F_float16)

# * Aggregating tensors
# * Aggregating tensors = condensing them from multiple values down to a smaller amount of values
# * get the absolute values

D = tf.constant([-7, -10])
# * get the absolute values
tf.abs(D)
# * changes negatives to positives

# * Lets go through the following forms of aggreagtions
# * get the minimum
# * get the maximum
# * get the mean of a tensor
# * get the sum of a tensor

# * create a random tesnosr with values 0 - 100
randomvalue_tensor = tf.constant(np.random.randint(0, 100, size=50))
# print(randomvalue_tensor)
# print(tf.size(randomvalue_tensor))
# print(randomvalue_tensor.shape)
# print(randomvalue_tensor.ndim)

# * find the minimum
# print(tf.reduce_min(randomvalue_tensor))
# * find the max
# print(tf.reduce_max(randomvalue_tensor))

# * find the mean
# print(tf.reduce_mean(randomvalue_tensor))
# * find the sum
# print(tf.reduce_sum(randomvalue_tensor))

# * find the standard deviation ( I believe it needs to be a float )
# print(tf.math.reduce_std(tf.cast(randomvalue_tensor, dtype=tf.float32)))


#this is to find the variance of a tensor
# print(tf.math.reduce_variance(tf.cast(randomvalue_tensor, dtype=tf.float32)))

# *** having trouble importing tensorflow_probability not sure how to
# *** remedy that
# import tensorflow_probability as tfp
# print(tfp.stats.variance(randomvalue_tensor))

# * find the positional maximum and minimum
#  * create a new tensor fro finding positional minimum and maxmimum

tf.random.set_seed(42)
G = tf.random.uniform(shape=[50])
# print(G)

# find the positional maximum
# print(tf.argmax(G))

# Index on our largest value position
# print(G[tf.argmax(G)])

# * find the max value of F
# print(tf.reduce_max(G))

# * Check for equality
# print(G[tf.argmax(G)] == tf.reduce_max(G))

# * find the positional minimum
# print(tf.argmin(G))

# * find the minimum using the positional minimum index
# print(G[tf.argmin(G)])

# * squeezing a tensor (removing all single dimensions)
# * Create a tensor to get started
tf.random.set_seed(42)
H = tf.constant(tf.random.uniform(shape=[50]), shape=(1, 1, 1, 1, 50))
# print(H.shape)
H_squeezed = tf.squeeze(H)
# print(H_squeezed, H_squeezed.shape)
# * squeeze removes all uneccessary dimensions from a tenosr

# * One Hot Encoding
# * one hot encoding is a form of numerical encoding

# * create a list of indices
# * one hot encode our list of indices
some_encoded_list = [0, 1, 2, 3]
# print(tf.one_hot(some_encoded_list, depth=4))

# specify custom values for one hot encoding
# print(tf.one_hot(some_encoded_list, depth=4, on_value="i be that guy", off_value="I hate sand"))

# * Squaring log Square Root

# * Create a new tensor
tensor_Square = tf.range(1, 10)
# print(tensor_Square)
# print(tf.square(tensor_Square))

# * find the square root (method requires non int type
# print(tf.sqrt(tf.cast(tensor_Square, dtype=tf.float32)))

# * find the log
# print(tf.math.log(tf.cast(tensor_Square, dtype=tf.float32)))

# * Tensors and NumPy
# * tensorflow interacts beautifully with NumPy Arrays
# * create a tensor directly from a numPy array

J = tf.constant(np.array([3., 7., 10.]))
# print(J)

# Convert our tensor back to a NumPy array
# print(np.array(J), type(np.array(J)))

# convert tensor J to a numpy array
# print(J.numpy(), type(J.numpy()))

K = tf.constant([3.])
# print(K.numpy()[0])

# The Default types of each are slightly different
numpy_J = tf.constant(np.array([3.,7.,10.]))
tensor_J = tf.constant([3.,7.,10.])
# check the datatypes of each
# tensor from a numpy array is a float64 and a regular tensor is a float32
# print(numpy_J.dtype)
# print(tensor_J.dtype)


# * Finding access to GPUs
print(tf.config.list_physical_devices())


