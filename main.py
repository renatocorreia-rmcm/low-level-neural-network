from NeuralNetwork import NeuralNetwork
from data import Vector
from data_set import data_set

jorge: NeuralNetwork = NeuralNetwork([784, 16, 16, 10])


case = data_set[0]


jorge.process(case[0])

jorge.backpropagate(case[1])

jorge.print_output()