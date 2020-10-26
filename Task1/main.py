import numpy as np
import json

with open('config/config.json') as configFile:
    config = json.load(configFile)

epochs = config['epochs']
trainingStep = config['trainingStep']
wBoundLower = config['wBoundLower']
wBoundUpper = config['wBoundUpper']
boundLower = config['boundLower']
boundUpper = config['boundUpper']
mPatterns = config['mPatterns']
nInputs = config['nInputs']

if config['isRandom']:
    inputs = np.random.uniform(
        boundLower, boundUpper, size=(mPatterns, nInputs))
    expectedOutput = np.random.uniform(boundLower, boundUpper, size=mPatterns)

else:
    patterns = np.loadtxt(config['path'], ndmin=2)
    inputs = patterns[:, :-1]
    expectedOutput = patterns[:, -1:].flatten()

weights = np.random.uniform(wBoundLower, wBoundUpper, size=inputs.shape[1])

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
