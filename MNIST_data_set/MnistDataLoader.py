import struct  # std lib

"""
allow to encode data to bytes (pack)
    and decode bytes to data (unpack)
string parameter "format" defines the layout

basically, operations to read bytes and write bytes
"""

from array import array  # std lib

"""
memory-efficient alternative to python list 
contains elements of the same basic value (constrained)
"""

"""
files from MNIST data set

# train
train-images-idx3-ubyte
train-labels-idx1-ubyte
# test
t10k-images-idx3-ubyte
t10k-labels-idx1-ubyte


each file starts with a header that needs to be interpreted before unpacking the actual data

labels:
    magic_num  # [0000][data type][number of dimensions]
    size  # amount of labels
images:
    magic_num
    size  # amount of images
    rows  # per image
    cols  # pre image	
"""

"""
    CLASS
"""


class MnistDataloader:
    def __init__(self, training_images_path, training_labels_path, test_images_path, test_labels_path):
        """load files paths"""
        self.training_images_path = training_images_path
        self.training_labels_path = training_labels_path
        self.test_images_path = test_images_path
        self.test_labels_path = test_labels_path

    def read_images_labels(self, images_filepath, labels_filepath):
        """"""

        """labels"""
        with open(labels_filepath, 'rb') as file:  # reading in binary
            # read header
            magic_num, size = struct.unpack(">II", file.read(8))
            # verify magic number
            if magic_num != 2049:
                raise ValueError(f'Magic number mismatch, expected 2049, got {magic_num}')

            # load labels as an array of unsigned char (single byte) from the raw bytes of the file
            #     wich are them transformed to an integer 0-255
            labels = array("B", file.read())

        with open(images_filepath, 'rb') as file:
            # read header
            magic_num, size, rows, cols = struct.unpack(">IIII", file.read(16))
            # verify magic number
            if magic_num != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic_num))
            # load the rest as a flat array of bytes for all images  # will be sliced for each image later
            image_data = array("B", file.read())  # this rest of file has 1 byte per pixel

        """images"""
        images = []
        for i in range(size):  # fill with images shapes
            images.append([0] * rows * cols)
        for i in range(size):
            # slice each image from image_data flat array
            start = i * rows * cols
            end = (i + 1) * rows * cols
            img = image_data[start:end]  # img is a flat array (sooner will be a vector)
            # append to images list
            images[i][:] = list(img)  # [:] replace content of list elements instead of substituting whole list

        return images, labels

    def load_data(self):
        x_train, y_train = self.read_images_labels(self.training_images_path, self.training_labels_path)
        x_test, y_test = self.read_images_labels(self.test_images_path, self.test_labels_path)
        return (x_train, y_train), (x_test, y_test)


#
# Verify Reading Dataset via MnistDataloader class
#

import random
import matplotlib.pyplot as plt

#
# Set file paths based on added MNIST Datasets
#
input_path = 'files/'
training_images_filepath = input_path + 'train-images-idx3-ubyte/train-images-idx3-ubyte'
training_labels_filepath = input_path + 'train-labels-idx1-ubyte/train-labels-idx1-ubyte'
test_images_filepath = input_path + 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte'
test_labels_filepath = input_path + 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte'


#
# Helper function to show a list of images with their relating titles
#
def show_images(images, title_texts):
    cols = 5
    rows = int(len(images) / cols) + 1
    plt.figure(figsize=(30, 20))
    index = 1
    for x in zip(images, title_texts):
        image = x[0]
        image_2d = [image[j * cols:(j + 1) * cols] for j in range(rows)]
        title_text = x[1]
        plt.subplot(rows, cols, index)
        plt.imshow(image_2d, cmap='gray')
        if title_text != '':
            plt.title(title_text, fontsize=15);
        index += 1


#
# Load MINST dataset
#
mnist_dataloader = MnistDataloader(
    training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath
)
(x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

#
# Show some random training and test images
#
images_2_show = []
titles_2_show = []
for i in range(0, 10):
    r = random.randint(1, 60000)
    images_2_show.append(x_train[r])
    titles_2_show.append('training image [' + str(r) + '] = ' + str(y_train[r]))

for i in range(0, 5):
    r = random.randint(1, 10000)
    images_2_show.append(x_test[r])
    titles_2_show.append('test image [' + str(r) + '] = ' + str(y_test[r]))

show_images(images_2_show, titles_2_show)
