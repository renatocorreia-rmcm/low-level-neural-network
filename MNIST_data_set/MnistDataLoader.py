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


"""  ABOUT THE FILES from MNIST data set

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

"""files pathes from main"""
input_path = 'MNIST_data_set/files/'
training_images_path = input_path + 'train-images.idx3-ubyte'
training_labels_path = input_path + 'train-labels.idx1-ubyte'
test_images_path = input_path + 't10k-images.idx3-ubyte'
test_labels_path = input_path + 't10k-labels.idx1-ubyte'


class MnistDataloader:
    def __init__(self) -> None:
        """load files paths"""
        self.training_images_path = training_images_path
        self.training_labels_path = training_labels_path
        self.test_images_path = test_images_path
        self.test_labels_path = test_labels_path

    def read_images_labels(self, images_filepath: str, labels_filepath: str) -> tuple[list[list[float]], list[int]]:
        """
        get images and labels from given filepath
        """

        """labels"""
        with open(labels_filepath, 'rb') as file:  # reading in binary
            # read header
            magic_num, size = struct.unpack(">II", file.read(8))
            # verify magic number
            if magic_num != 2049:
                raise ValueError(f'Magic number mismatch, expected 2049, got {magic_num}')

            # load labels as an array of unsigned char (single byte) from the raw bytes of the file
            #     wich are them transformed to an integer 0-255
            labels = list(array("B", file.read()))

        with open(images_filepath, 'rb') as file:
            # read header
            magic_num, size, rows, cols = struct.unpack(">IIII", file.read(16))
            # verify magic number
            if magic_num != 2051:
                raise ValueError(f'Magic number mismatch, expected 2051, got {magic_num}')
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

    def load_data(self) -> tuple[tuple[list[list[float]], list[float]], tuple[list[list[float]], list[float]]]:
        """
        :return: 2 tuples: training data and testing data
        each tuple contains the features list and the targets list
        """
        x_train, y_train = self.read_images_labels(self.training_images_path, self.training_labels_path)
        x_test, y_test = self.read_images_labels(self.test_images_path, self.test_labels_path)
        return (x_train, y_train), (x_test, y_test)
