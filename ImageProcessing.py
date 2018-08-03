import tensorflow as tf
import numpy as np

#tf.keras is a high-level API to build and train models in TensorFlow
#import the matplotlib function-->to plot information/data
import matplotlib.pyplot as plt

#OS-->to load files and save checkpoints
import os

#Load MNIST data from tf examples

image_height = 28
image_width = 28
color_channels = 1
model_name = "mnist"
mnist = tf.contrib.learn.datasets.load_dataset("mnist")
#code below are the training set - data model is fed this data
train_data = mnist.train.images
train_labels = np.asarray(mnist.train.labels, dtype=np.int32)


#lines below are where the model is tested against the test set -test_images & test_label arrays.
eval_data = mnist.test.images
eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

category_names = list(map(str, range(10)))

#TODO: Process mnist data
#print(train_data.shape)
train_data = np.reshape(train_data, (-1, image_height, image_width, color_channels))
#print(train_data.shape)
eval_data = np.reshape(eval_data, (-1, image_height, image_width, color_channels))



# Load cifar data from file
image_height = 32
image_width = 32

color_channels = 3
model_name = "cifar"
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
        return dict
cifar_path = '/home/student/Desktop/AnsonQ/untitled/imageRecognition/cifar-10-batches-py/'
train_data = np.array([])
train_labels = np.array([])

#Load all the data batches.
for i in range(1, 3):
    data_batch = unpickle(cifar_path + 'data_batch_' + str(i))
    train_data = np.append(train_data, data_batch[b'data'])
    train_labels = np.append(train_labels, data_batch[b'labels'])
    print(train_data.shape)

#Load the eval batch
eval_batch = unpickle(cifar_path + 'test_batch')

eval_data = eval_batch[b'data']
eval_labels = eval_batch[b'labels']

#Load the english category names
category_names_bytes = unpickle(cifar_path + 'batches.meta')[b'label_names']
category_names = list(map(lambda x: x.decode("utf-8"), category_names_bytes))


#TODO: Process Cifar data
#Cifar's data not in convenient format --> need's function to process data
#Line below is function that will help process data.
def process_data(data):
    float_data = np.array(data, dtype=float) / 225.0
    reshaped_data = np.reshape(float_data, (-1, color_channels, image_height, image_width))
    print('The dimensions of the data are ' + str(reshaped_data.shape))
    # The incorrect image

    #plt.imshow(reshaped_data[0])
    #plt.show()
    transposed_data = np.transpose(reshaped_data, [0, 2, 3, 1])
    print('THe dimensions of the transposed data are ' + str(transposed_data.shape))
    plt.imshow(transposed_data[150])
    print(str(transposed_data[150].shape))
    plt.show()
process_data(train_data)

#TODO: first image

#TODO: The neural network: this will process the data
class ConvNet:

    def _init_(self, image_height, image_width, channels, num_classes):
        #initizalizer above will take info above and use it to create the shape of the network layers.
        def __init__(self, image_height, image_width, channels, num_classes):
            self.input_layer = tf.placeholder(dtype=tf.float32, shape=[None, image_height, image_width, channels],
                                              name="inputs")
            print(self.input_layer.shape)

            conv_layer_1 = tf.layers.conv2d(self.input_layer, filters=32, kernel_size=[5, 5], padding="same",
                                            activation=tf.nn.relu)
            print(conv_layer_1.shape)

            pooling_layer_1 = tf.layers.max_pooling2d(conv_layer_1, pool_size=[2, 2], strides=2)
            print(pooling_layer_1.shape)

            conv_layer_2 = tf.layers.conv2d(pooling_layer_1, filters=64, kernel_size=[5, 5], padding="same",
                                            activation=tf.nn.relu)
            print(conv_layer_2.shape)

            pooling_layer_2 = tf.layers.max_pooling2d(conv_layer_2, pool_size=[2, 2], strides=2)
            print(pooling_layer_2.shape)

            flattened_pooling = tf.layers.flatten(pooling_layer_2)
            dense_layer = tf.layers.dense(flattened_pooling, 1024, activation=tf.nn.relu)
            print(dense_layer.shape)



