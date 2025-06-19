from NeuralNetwork import NeuralNetwork

jorge: NeuralNetwork = NeuralNetwork([4, 3, 2, 1])

feature_0 = [30, 50, 10, 5]
target_0 = [0.5]

feature_1 = [40, 60, 0, 20]
target_1 = [0.7]

batch = [(feature_0, target_0), (feature_1, target_1)]

print(f"analysis before:\n")
jorge.analyse()

jorge.learn_batch(batch)

print(f"analysis after:\n")
jorge.analyse()
