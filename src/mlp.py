import numpy as np
import time

class MLP_NeuralNetwork(object):
    def __init__(self, input, hidden, output):
        """
        :param input: number of input neurons
        :param hidden: number of hidden neurons
        :param output: number of output neurons
        """
        self.input = input + 1 # add 1 for bias node
        self.hidden = hidden
        self.output = output
        # create randomized weights
        # self.wi = np.random.randn(self.input, self.hidden)
        # self.wo = np.random.randn(self.hidden, self.output)
        self.wi = np.zeros((self.input, self.hidden))
        self.wo = np.zeros((self.hidden, self.output))
        self.time_vector = 0

    def tanh(self, x, derivative=False):
        if (derivative == True):
            return (1 - (x ** 2))
        return np.tanh(x)

    def leaky_relu(self, data, epsilon=0.01):
        return np.maximum(data, epsilon * data)

    def feedForward(self, network_inputs):
        # t = time.time()
        input_activations = np.zeros(self.input)
        input_activations[:-1] = network_inputs
        input_activations = self.leaky_relu(input_activations)
        input_activations[-1] = 1
        hidden_activations = self.leaky_relu(np.dot(input_activations, self.wi))
        output_activations = np.dot(hidden_activations, self.wo)
        # self.time_vector = time.time() - t
        # print("vector time: {}".format(self.time_vector))
        return output_activations

    def change_weights(self, input_to_hidden, hidden_to_output):
        self.wi = input_to_hidden
        self.wo = hidden_to_output

    def save_weights_to_file(self, filename=None):
        if filename is None:
            print("save_weights_to_file: Filename unset, can't save!")
            return 0

        filename_wi = filename + "-wi"
        filename_wo = filename + "-wo"

        np.save(filename_wi, self.wi)
        np.save(filename_wo, self.wo)

    def load_weights(self, filename=None):
        if filename is None:
            print("load_weights: Filename unset, can't load!")
            return 0

        filename_wo = filename + "-wo.npy"
        filename_wi = filename + "-wi.npy"

        self.change_weights(np.load(filename_wi), np.load(filename_wo))

def main():

    my_mlp = MLP_NeuralNetwork(input=7, hidden=16, output=2)
    # my_mlp.change_weights(input_to_hidden=np.random.randn(my_mlp.input, my_mlp.hidden), hidden_to_output=np.random.randn(my_mlp.hidden, my_mlp.output))
    # my_mlp.save_weights_to_file("weights")
    my_mlp.load_weights(filename="weights")
    mlp_in = np.ones(7)
    output = my_mlp.feedForward(mlp_in)
    print(output)

if __name__ == '__main__':
    main()
