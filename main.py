from NeuralNetwork import NeuralNetwork
from data import Vector

jorge: NeuralNetwork = NeuralNetwork([3, 2, 1])

inp: Vector = Vector([40, 50, 60])
jorge.process(inp)
jorge.analyse()
