# Generic Neural Network 

Implementation of a generic MLP that may be trained on miscelaneous goals.


## A low level implementation

**I'm not using any ML libs.** <br>
The goal is to implement all the math used in a neural network by myself. <br>


## Overview

The class initialize a neural network object with random weights and biases, <br>
taking size of layers, backup file and dataset as arguments

Then this object can either load the weights and biases previously stored in a .txt by another instance of this class <br>
or it can be trained by a given data set and store his new parameters


## The Example

In this case im using the model **I trained previously** to recognize handwritten digits using MNIST dataset. <br>
It takes as `28 x 28` image (byte vector `784 x 1`) as input, has 2 hidden layers of  size `16`, <br>
and outputs the result in the correponding neuron of the last layer (size `10`)

### Training the model

### Testing the accuracy

Its parameters are backuped in the "handwritten_digits_recognition.txt", so they are directly loaded into the model. <br>
As seen in the output, this particular instance has an accuracy of **75.60%**
![accuracy testing output](https://github.com/user-attachments/assets/f9788b0c-0c1b-4cf1-bb4e-0ebc134512c7)

