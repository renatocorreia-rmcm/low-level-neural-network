"""
brief script to test the access to the data set
"""

import MnistDataLoader
from random import randint
import matplotlib.pyplot as plt

"""files pathes from this file"""
input_path = 'files/'
training_images_filepath = input_path + 'train-images.idx3-ubyte'
training_labels_filepath = input_path + 'train-labels.idx1-ubyte'
test_images_filepath = input_path + 't10k-images.idx3-ubyte'
test_labels_filepath = input_path + 't10k-labels.idx1-ubyte'


def show_images(images, title_texts):
    cols = 5
    rows = int(len(images) / cols) + 1
    plt.figure(figsize=(30, 20))
    index = 1
    for image, title_text in zip(images, title_texts):
        image_2d = [image[(j*28): (j+1)*28] for j in range(28)]
        plt.subplot(rows, cols, index)
        plt.imshow(image_2d, cmap='gray')
        if title_text != '':
            plt.title(title_text, fontsize=15)
        index += 1


"""load data"""
mnist_dataloader = MnistDataLoader.MnistDataloader()  # todo: solve path given in constructor: only work when calling from main
(x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

images = []
titles = []
for i in range(10):
    r = randint(0, 59999)
    images.append(x_train[r])
    titles.append(f'training image [{r}] = {y_train[r]}')

for i in range(0, 5):
    r = randint(0, 9999)
    images.append(x_test[r])
    titles.append(f'test image [{r}] = {y_test[r]}')

show_images(images, titles)
plt.show()
