import tensorflow as tf
import numpy as np
import keras

scalar = tf.constant(3)
print(scalar)
print(scalar.ndim)
vector = tf.constant([1, 2])
print(vector)
print(vector.ndim)
matrix = tf.constant([[1, 2], [3, 4]])
print(matrix)
print(matrix.ndim)

(train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()
print(train_images.ndim)
print(train_images.shape)
print(train_images.dtype)
v = tf.Variable([1, 2])
c = tf.constant([1, 2])
print(v, c)
print(v[0].assign(7))
print(c)
tf.random.set_seed(42)
tf.random.normal(shape=(3, 2))

tf.random.set_seed(42)
tf.random.normal(shape=(3, 2))

tf.ones(shape=[3, 2])
tf.zeros(shape=[3, 2])
numpy_A = np.arange(1, 25, dtype=np.int32)
print(numpy_A)
print(len(numpy_A))
tensor_A = tf.constant(numpy_A, shape=[2, 4, 3])
print(tensor_A)
rank_4_tensor = tf.zeros([2, 3, 4, 5])
print(rank_4_tensor)
print(rank_4_tensor[:2, :2, :2, :2])
print(rank_4_tensor.ndim)
rank_5_tensor = rank_4_tensor[..., tf.newaxis]
print(rank_5_tensor)
print(rank_4_tensor.shape)
print(rank_5_tensor.shape)
rank_6_tensor = tf.expand_dims(rank_5_tensor, axis=-1)
print(rank_6_tensor.shape)
tensor = tf.constant([[1, 2], [3, 4]])
print(tensor)
print(tensor + 10)
print(tensor - 10)
print(tensor * 10)
print(tf.multiply(tensor, 10))
print(tf.matmul(tensor, tensor))
print(tf.matmul(tensor, tensor))
tensor = tf.constant([[1, 2, 3], [4, 5, 6]])
print(tensor)
print(tf.reshape(tensor, shape=(3, 2)))
print(tf.transpose(tensor))
tensor = tf.constant([1.2, 3.4])
print(tensor)
print(tf.cast(tensor, dtype=tf.float16))
tensor = tf.constant(np.random.randint(0, 100, 50), shape=(1, 1, 1, 1, 50))
print(tensor.shape)
print(tf.squeeze(tensor).shape)
data = [0, 1, 2, 3]
print(tf.one_hot(data, depth=4))


@tf.function
def f(x):
    return x ** 2


data = tf.constant(np.arange(0, 10))
print(f(data))
print(tf.config.list_logical_devices())
print(tf.config.list_logical_devices())
