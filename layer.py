import numpy as np

class Layer:
    def __init__(self, *, input_size, neuronsNum, activation_function_name=None):
        self.input_size = input_size
        self.inputs = None
        self.activation_name = activation_function_name
        self.neuronsNum = neuronsNum
        self.weights = np.random.randn(self.neuronsNum, self.input_size) * np.sqrt(2.0 / self.input_size)
        self.biases = np.zeros((self.neuronsNum, 1))
        self.z = np.zeros((self.neuronsNum, 1))
        self.a = np.zeros((self.neuronsNum, 1))
        self.activation_function = {
            "relu": lambda z: np.maximum(0, z),
            "tanh": lambda z: np.tanh(z),
            "sigmoid": lambda z: 1 / (1 + np.exp(-z)),
            "softmax": lambda z: np.exp(z - np.max(z)) / np.sum(np.exp(z - np.max(z)))
        }
        self.activation_function_derivative = {
            "relu": lambda z: np.where(z > 0, 1, 0),
            "tanh": lambda z: 1 - np.tanh(z) ** 2,
            "sigmoid": lambda z: (1 / (1 + np.exp(-z)) * (1 - 1 / (1 + np.exp(-z)))),
        }
        self.z_err = np.zeros((self.neuronsNum, 1))
        self.b_err = np.zeros((self.neuronsNum, 1))
        self.w_err = np.zeros((self.neuronsNum, self.input_size))

    def __str__(self):
        return ("Weights: \n" + str(self.weights) + " \n " + "Biases: \n" + str(self.biases) + "\n" + "z: "
                + str(self.z) + "\n" + "a: " + str(self.a))

    def MSE(self, targets_vec):
        targets_vec = np.array(targets_vec).reshape(self.neuronsNum, 1)
        return np.mean((targets_vec - self.a) ** 2)

    def MSE_derivative(self, targets_vec):
        targets_vec = np.array(targets_vec).reshape(self.neuronsNum, 1)
        return (self.a - targets_vec) * 2 / len(targets_vec)

    def CrossEntropyMultiClass(self, targets_vec):
        targets_vec = np.array(targets_vec).reshape(self.neuronsNum, 1)
        return -1 * np.sum((targets_vec * (np.log(self.a + 1e-15))))

    def forward(self, inputs):
        self.inputs = np.array(inputs).reshape(len(inputs), 1)
        self.z = np.dot(self.weights, self.inputs) + self.biases
        if self.activation_name:
            self.a = self.activation_function[self.activation_name](self.z)
            return self.a
        else:
            self.a = self.z
            return self.a

    def backward(self, *, dLoss=None, next_layer=None):
        if self.activation_name and self.activation_name != "softmax":
            act_derivative = self.activation_function_derivative[self.activation_name](self.z)
        else:
            act_derivative = np.ones_like(self.z)

        if dLoss is not None:
            if self.activation_name == "softmax":
                self.z_err = dLoss
            else:
                self.z_err = dLoss * act_derivative
        elif next_layer is not None:
            self.z_err = np.dot(next_layer.weights.T, next_layer.z_err) * act_derivative
        else:
            raise ValueError("Enter dLoss for out or next_layer for hidden")

        self.b_err = self.z_err
        self.w_err = np.dot(self.z_err, self.inputs.T)

        return self.z_err

    def update(self, *, lr):
        self.weights -= lr * self.w_err
        self.biases -= lr * self.b_err
