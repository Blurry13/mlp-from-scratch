import numpy as np
import json
from layer import Layer


class Network:
    def __init__(self, *, layersNum=None, dataSize, targetSize, WeightsConfFilePath=None, BiasesConfFilePath=None, ParametersConfFilePath=None):
        self.layers = []
        self.trainingHistory = []
        if WeightsConfFilePath is None and BiasesConfFilePath is None and ParametersConfFilePath is None:
            if layersNum < 1:
                raise ValueError("The network must have at least one layer")
            self.layersNum = layersNum
            for i in range(self.layersNum):
                self.activation_function = input(f"Enter the activation function for layer {i + 1} (tanh, sigmoid, relu, softmax): ").strip().lower()

                if i == 0:
                    self.inputSize = dataSize
                else:
                    self.inputSize = self.layers[i - 1].neuronsNum

                if i == self.layersNum - 1:
                    self.neuronsNum = targetSize
                else:
                    self.neuronsNum = int(input(f"Enter the number of neurons for layer {i + 1}: "))

                self.layers.append(Layer(input_size=self.inputSize, neuronsNum=self.neuronsNum,
                                         activation_function_name=self.activation_function))

            self.lossFunction = input("Enter the loss function(mse, crossentropy): ").strip().lower()

        elif WeightsConfFilePath is not None and BiasesConfFilePath is not None and ParametersConfFilePath is not None:
            fParams = open(ParametersConfFilePath)
            networkParameters = json.load(fParams)
            self.layersNum = len(networkParameters["layers"])
            for index, layerName in enumerate(networkParameters["layers"]):
                self.activation_function = networkParameters["layers"][layerName][1]

                if index == 0:
                    self.inputSize = dataSize
                else:
                    self.inputSize = self.layers[index - 1].neuronsNum

                if index == self.layersNum - 1:
                    self.neuronsNum = targetSize
                else:
                    self.neuronsNum = networkParameters["layers"][layerName][0]

                self.layers.append(Layer(input_size=self.inputSize, neuronsNum=self.neuronsNum,
                                         activation_function_name=self.activation_function))

            self.lossFunction = networkParameters["lossFunction"]
            fParams.close()

            fWeights = np.load(WeightsConfFilePath)
            for index, name in enumerate(fWeights.files):
                if fWeights[name].shape != self.layers[index].weights.shape:
                    raise ValueError(f"Invalid weights shape for {name}")

                self.layers[index].weights = fWeights[name]

            fBiases = np.load(BiasesConfFilePath)
            for index, name in enumerate(fBiases.files):
                if fBiases[name].shape != self.layers[index].biases.shape:
                    raise ValueError(f"Invalid biases shape for {name}")

                self.layers[index].biases = fBiases[name]
        else:
            raise ValueError("Provide all configuration files: weights, biases, and network parameters")

    def __str__(self):
        result = ""
        for index, layer in enumerate(self.layers, start=1):
            result += (f"Input size of layer {index}: {layer.input_size} \n"
                       f"Number of neurons in layer {index}: {layer.neuronsNum} \n"
                       f"Activation function of layer {index}: {layer.activation_name} \n")
        result += f"Loss function: {self.lossFunction} \n"
        return result

    def trainNetwork(self, *, epochs, x, y, lr, stopCondition):
        if stopCondition < 0 or stopCondition > 100:
                raise ValueError("Minimum accuracy percentage must be between 0 and 100")

        for epoch in range(epochs):
            epoch_loss = 0.0
            correct = 0
            perm = np.random.permutation(len(x))
            for sample, target in zip(x[perm], y[perm]):
                for index, layer in enumerate(self.layers):
                    if index == 0:
                        layer.forward(sample)
                    else:
                        layer.forward(self.layers[index - 1].a)

                pred = np.argmax(self.layers[-1].a)
                true = np.argmax(target)
                if pred == true:
                    correct += 1

                if self.lossFunction == "mse":
                    loss = self.layers[-1].MSE(target)
                elif self.lossFunction == "crossentropy":
                    loss = self.layers[-1].CrossEntropyMultiClass(target)
                else:
                    raise ValueError("Unsupported loss function")
                epoch_loss += loss
                for index in range(len(self.layers) - 1, -1, -1):
                    if index == len(self.layers) - 1:
                        if self.lossFunction == "mse":
                            dLoss = self.layers[index].MSE_derivative(target)
                        elif self.lossFunction == "crossentropy":
                            targetVec = np.array(target).reshape(self.layers[index].neuronsNum, 1)
                            if self.layers[index].activation_name == "softmax":
                                dLoss = self.layers[index].a - targetVec
                            else:
                                raise ValueError("Cross-entropy requires softmax in the output layer")
                        else:
                            raise ValueError("Unsupported loss function")

                        self.layers[index].backward(dLoss=dLoss)
                    else:
                        self.layers[index].backward(next_layer=self.layers[index + 1])

                for layer in self.layers:
                    layer.update(lr=lr)
            if epoch > 0 and epoch % 100 == 0:
                lr *= 0.8
            accuracy = correct / len(x) * 100
            print(f"Epoch {epoch + 1}, loss = {epoch_loss} train accuracy = {accuracy:.1f}% LR: {lr}")
            self.trainingHistory.append({"epoch": epoch + 1, "loss": float(epoch_loss), "accuracy": float(accuracy)})
            if accuracy >= stopCondition:
                break

    def evaluateNetwork(self, *, x, y):
        correct = 0
        for sample, target in zip(x, y):
            for index, layer in enumerate(self.layers):
                if index == 0:
                    layer.forward(sample)
                else:
                    layer.forward(self.layers[index - 1].a)

            pred = np.argmax(self.layers[-1].a)
            true = np.argmax(target)
            if pred == true:
                correct += 1

        accuracy = correct / len(x)
        print(f"Test accuracy = {accuracy * 100:.1f}%")

        return accuracy

    def saveWeights(self, *, saveMethod):
        if saveMethod == "json":
            savedWeights = {f"layer_{index}": layer.weights.tolist() for index, layer in
                            enumerate(self.layers, start=1)}
            f = open("./NetworkWeights.json", "w")
            f.write(json.dumps(savedWeights))
            f.close()
        elif saveMethod == "npz":
            savedWeights = {f"layer_{index}": layer.weights for index, layer in enumerate(self.layers, start=1)}
            np.savez("./NetworkWeights.npz", **savedWeights)
        else:
            raise ValueError("Unsupported save method")

    def saveBiases(self, *, saveMethod):
        if saveMethod == "json":
            savedBiases = {f"layer_{index}": layer.biases.tolist() for index, layer in enumerate(self.layers, start=1)}
            f = open("./NetworkBiases.json", "w")
            f.write(json.dumps(savedBiases))
            f.close()
        elif saveMethod == "npz":
            savedBiases = {f"layer_{index}": layer.biases for index, layer in enumerate(self.layers, start=1)}
            np.savez("./NetworkBiases.npz", **savedBiases)
        else:
            raise ValueError("Unsupported save method")

    def saveNetworkConf(self):
        savedParameters = {"layers": {f"layer_{index}": (layer.neuronsNum, layer.activation_name) for index, layer in enumerate(self.layers, start=1)}, "lossFunction": self.lossFunction,"layersNum": self.layersNum}
        f = open("./NetworkParameters.json", "w")
        f.write(json.dumps(savedParameters))
        f.close()

    def saveNetworkParameters(self, *, saveMethod):
        self.saveWeights(saveMethod=saveMethod)
        self.saveBiases(saveMethod=saveMethod)
        self.saveNetworkConf()

    def saveNetworkTrainingHistory(self):
        f = open("./NetworkTrainingHistory.json", "w")
        f.write(json.dumps(self.trainingHistory))
        f.close()