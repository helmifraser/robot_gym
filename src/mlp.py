import numpy as np
import time

class MLP_NeuralNetwork(object):
    def __init__(self, input, hidden, hidden_2, output):
        """
        :param input: number of input neurons
        :param hidden: number of hidden neurons
        :param output: number of output neurons
        """
        self.input = input + 1 # add 1 for bias node
        self.hidden = hidden
        self.hidden_2 = hidden_2
        self.output = output
        # create randomized weights
        # self.wi = np.random.randn(self.input, self.hidden)
        # self.wo = np.random.randn(self.hidden, self.output)
        self.wi = np.zeros((self.input, self.hidden))
        self.wi_2 = np.zeros((self.hidden, self.hidden_2))
        self.wo = np.zeros((self.hidden_2, self.output))
        self.time_vector = 0

    def tanh(self, x, derivative=False):
        if (derivative == True):
            return (1 - (x ** 2))
        return np.tanh(x)

    def leaky_relu(self, data, epsilon=0.01):
        return np.maximum(data, epsilon * data)

    def feed_forward(self, network_inputs):
        # t = time.time()
        # print(np.shape(network_inputs))
        input_activations = np.zeros(self.input)
        input_activations[:-1] = network_inputs
        input_activations = self.leaky_relu(input_activations)
        input_activations[-1] = 1
        # np.reshape(input_activations, (1, 8))
        # print(np.shape(input_activations))
        # print(np.shape(self.wi))

        # print(np.reshape(input_activations, (8, 1)))
        hidden_activations = self.leaky_relu(np.dot(input_activations, self.wi))
        hidden_activations_2 = self.leaky_relu(np.dot(hidden_activations, self.wi_2))
        output_activations = np.dot(hidden_activations_2, self.wo)
        output_activations[0] = self.leaky_relu(output_activations[0]/5, epsilon=-1)
        output_activations[1] = output_activations[1]/7
        output_activations = np.nan_to_num(output_activations)
        # self.time_vector = time.time() - t
        # print("vector time: {}".format(self.time_vector))
        return np.round(output_activations, decimals=4)

    def change_weights(self, input_to_hidden, hidden_to_hidden, hidden_to_output):
        self.wi = input_to_hidden
        self.wi_2 = hidden_to_hidden
        self.wo = hidden_to_output

    def save_weights_to_file(self, filename=None):
        if filename is None:
            print("save_weights_to_file: Filename unset, can't save!")
            return 0

        filename_wi = filename + "-wi"
        filename_wi_2 = filename + "-wi-2"
        filename_wo = filename + "-wo"

        np.save(filename_wi, self.wi)
        np.save(filename_wi, self.wi_2)
        np.save(filename_wo, self.wo)

    def load_weights(self, filename=None):
        if filename is None:
            print("load_weights: Filename unset, can't load!")
            return 0

        filename_wi = filename + "-wi.npy"
        filename_wi_2 = filename + "-wi-2.npy"
        filename_wo = filename + "-wo.npy"

        self.change_weights(np.load(filename_wi), np.load(filename_wi_2), np.load(filename_wo))

def main():

    my_mlp = MLP_NeuralNetwork(input=7, hidden=16, output=2)
    my_mlp.change_weights(input_to_hidden=np.random.randn(my_mlp.input, my_mlp.hidden), hidden_to_output=np.random.randn(my_mlp.hidden, my_mlp.output))
    # my_mlp.save_weights_to_file("weights")
    # my_mlp.load_weights(filename="weights")
    mlp_in = np.ones(7)
    output = my_mlp.feed_forward(mlp_in)
    print(output)

if __name__ == '__main__':
    main()
