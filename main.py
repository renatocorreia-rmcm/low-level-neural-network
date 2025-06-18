from NeuralNetwork import NeuralNetwork
from data import Vector

jorge: NeuralNetwork = NeuralNetwork([3, 3, 2])


data_point = [30, 50, 70]


prediction_0: Vector = jorge.process(data_point)


print(prediction_0)
