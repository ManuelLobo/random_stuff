import numpy as np


def simple_neural():
    def relu(z):
        return max(0, z)

    def feed_forward(x, hidden_weights, output_weights):
        # Hidden Layer
        Zh = x * hidden_weights
        H = relu(Zh)

        # Ouput Layer
        Zo = H * output_weights
        output = relu(Zo)
        print(output)
        return output

    feed_forward(10, 0.5, 0.5)
    feed_forward(20, 0.5, 0.5)

simple_neural()



def complex_neural():

    INPUT_LAYER_SIZE = 1
    HIDDEN_LAYER_SIZE = 2
    OUTPUT_LAYER_SIZE = 2


    def init_weights():
        hidden_weights = np.random.randn(INPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE) * np.sqrt(2.0/INPUT_LAYER_SIZE)
        output_weights = np.random.randn(HIDDEN_LAYER_SIZE, OUTPUT_LAYER_SIZE) * np.sqrt(2.0/HIDDEN_LAYER_SIZE)


    def init_bias():
        hidden_bias = np.full((1, HIDDEN_LAYER_SIZE), 0.1)
        output_bias = np.full((1, HIDDEN_LAYER_SIZE), 0.1)
        return hidden_bias, output_bias


    def relu(Z):
        return np.maximum(0, Z)


    def relu_prime(Z):
        """
        Z - weighted input matrix

        Returns gradient of Z where all negative values are set to 0 and all
        positive values are set to 1.
        """

        Z[Z < 0] = 0
        Z[Z > 0] = 1
        return Z


    def feed_forward(x):
        """
        x - input Matrix
        ZH - hidden layer weighted input
        Zo - output layer weighted input
        H - hidden layer activation
        y - output layer
        yHat - output layer predictions
        """

        # Hidden Layer

        # Update weighted input calculation to handle matrices.
        # Multiply input matrix by weights, connection them to the neurons in the
        # next layer. Also add the bias vector.
        Zh = np.dot(X, Wh) + Bh
        H = relu(Zh)

        # Output Layer
        Zo = np.dot(H, Wo) + Bo
        yHat = relu(Zo)

        return yHat




def back():
    def relu_prime(z):
        if z > 0:
            return 1
        return 0

    def cost(yHat, y):
        return 0.5 * (yHat - y)**2

    def cost_prime(yHat, y):
        return yHat - y

    def backprop(x, y, Wh, Wo, lr):
        yHat = feed_forward(x, Wh, Wo)

        # Layer Error
        Eo = (yHat - y) * relu_prime(Zo)
        Eh = Eo * Wo * relu_prime(Zh)

        # Cost derivative for weights
        dWo = Eo * H
        dWh = Eh * x

        # Update weights
        Wh -= lr * dWh
        Wo -= lr * dWo













    #
