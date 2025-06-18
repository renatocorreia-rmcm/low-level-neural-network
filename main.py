from NeuralNetwork import NeuralNetwork
from data import Vector

jorge: NeuralNetwork = NeuralNetwork([3, 3, 2])

feature_0: list[float] = [30, 50, 10]

prediction_0: list[float] = jorge.process(feature_0)

print(jorge.analyse())

print(prediction_0)
