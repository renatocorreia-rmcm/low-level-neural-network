from NeuralNetwork import NeuralNetwork  # my class
from MNIST_data_set import MnistDataLoader  # the MNIST dataset dataloader


# initialize mnist data loader
data_loader = MnistDataLoader.MnistDataloader()
# load data set
data_set = data_loader.load_data()


# initialize neural network
jorge: NeuralNetwork = NeuralNetwork(
	[784, 16, 16, 10], "handwritten_digits_recognition", data_set
)


""" model train example  """
"""
n_epochs = 15
learn_step = 0.8

print(f"initial accuracy: {jorge.get_accuracy()}")

for i_epoch in range(n_epochs):
	print(f"epoch {i_epoch} -----")
	jorge.train_epoch(learn_step)
	print(f"accuracy: {jorge.get_accuracy()}")
	learn_step *= 0.5  # scaffolding technique
	
# jorge.store() if want to keep those parameters in furthers initializations
"""


""" just testing the accuracy of the model i have already trained """
jorge.load()
print(jorge.get_accuracy())
