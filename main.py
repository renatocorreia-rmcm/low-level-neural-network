from NeuralNetwork import NeuralNetwork

jorge: NeuralNetwork = NeuralNetwork([10, 5, 4], "handwritten_digits_recognition", 1.0)  # todo: solve crash when learn_step = 1

feature_0 = [30, 50, 10, 5, 40, 60, 0, 20, 90, 80]
target_0 = [1.0, 0.0, 0.0, 0.0,]

feature_1 = [40, 60, 0, 20, 100, 80, 85, 20, 50, 10]
target_1 = [0.0, 0.0, 0.0, 1.0]

batch = [(feature_0, target_0), (feature_1, target_1)]


jorge.load()

jorge.learn_batch(batch)


jorge.store()
