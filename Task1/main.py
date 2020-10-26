import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import json

sb.set_theme(style="darkgrid")


def generateInitialWeights(low, high, size):
    return np.random.uniform(low, high, size=size)


def calculateOutput(inputs, weights):
    output = 0
    for i in range(len(inputs)):
        output += inputs[i]*weights[i]
    return output


def calculateWeights(weights, trainingStep, inputs, output, expectedOutput):
    for i in range(len(weights)):
        weights[i] += trainingStep * (expectedOutput - output) * inputs[i]


def main():
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
        expectedOutput = np.random.uniform(
            boundLower, boundUpper, size=mPatterns)

    else:
        patterns = np.loadtxt(config['path'], ndmin=2)
        inputs = patterns[:, :-1]
        expectedOutput = patterns[:, -1:].flatten()

    weights = generateInitialWeights(wBoundLower, wBoundUpper, inputs.shape[1])

    error = []
    for _ in range(epochs):
        errorSum = 0
        for i in range(inputs.shape[0]):
            output = calculateOutput(inputs[i], weights)
            errorSum += (expectedOutput[i] - output) ** 2
            calculateWeights(
                weights, trainingStep, inputs[i],  output, expectedOutput[i])
        error.append(errorSum / inputs.shape[0])

    print('Epochs: ', epochs)
    print('Training step: ', trainingStep)
    print('Final weights: ', weights)

    sb.lineplot(data=error)
    plt.title('Mean Square Error')
    plt.ylabel('Error')
    plt.xlabel('Epoch')
    plt.show()


if __name__ == "__main__":
    main()
