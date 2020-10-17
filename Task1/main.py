import numpy as np
import random
import json

with open('config/config.json') as configFile:
    config = json.load(configFile)

epochs = config['epochs']
trainingStep = config['trainingStep']

patterns = np.loadtxt(config['path'], ndmin=2)

inputs = patterns[:, :-1]
expectedOutput = patterns[:, -1:].flatten()

weights = [random.uniform(-1, 1) for _ in range(inputs.shape[1])]

for i in range(epochs):

    for j in range(inputs.shape[0]):
        output = 0
        for k in range(len(inputs[j])):
            output += inputs[j][k]*weights[k]

        for l in range(len(weights)):
            weights[l] += trainingStep * \
                (expectedOutput[j] - output) * inputs[j][l]

print('Epochs: ', epochs)
print('Training step: ', trainingStep)
print('Final weights: ', weights)
