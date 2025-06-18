from NeuralNetwork import NeuralNetwork
from data import Vector

"""
    my data types can be entirely "encapsulate"
    do all conversions inside of class, so it can recieve and output lists instead of user defined vectors
"""

jorge: NeuralNetwork = NeuralNetwork([3, 3, 2])


data_point = [30, 50, 70]


prediction_0: Vector = jorge.process(data_point)


print(prediction_0)
