from NeuralNetwork import NeuralNetwork
from data import Vector
from data_set import data_set

jorge: NeuralNetwork = NeuralNetwork([784, 16, 16, 10])

""" generate 28*28 grid on console
print('\"\"\"')
for i in range (28):
	for j in range(28):
		x = str(randint(0,99))
		x = " "*(2-len(x)) + x 
		print(x, end=", ")
	print()
print('\"\"\"')
"""

case = data_set[0]


jorge.process(case[0])
jorge.backpropagate(case[1])

jorge.print_output()