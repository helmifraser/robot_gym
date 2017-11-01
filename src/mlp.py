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
        # set up array of 1s for activations
        self.ai = [1.0] * self.input
        self.ah = [1.0] * self.hidden
        self.ao = [1.0] * self.output
        # create randomized weights
        self.wi = np.random.randn(self.input, self.hidden)
        self.wo = np.random.randn(self.hidden, self.output)
        # create arrays of 0 for changes
        self.ci = np.zeros((self.input, self.hidden))
        self.co = np.zeros((self.hidden, self.output))
        self.time_vector = 0
        self.time_loop = 0

    def tanh(self, x, derivative=False):
        if (derivative == True):
            return (1 - (x ** 2))
        return np.tanh(x)

    def feedForward_old(self, inputs):
        t = time.time()
        if len(inputs) != self.input-1:
            raise ValueError('Wrong number of inputs!')
        # input activations
        for i in range(self.input -1): # -1 is to avoid the bias
            self.ai[i] = inputs[i]
        # hidden activations
        for j in range(self.hidden):
            sum = 0.0
            for i in range(self.input):
                sum += self.ai[i] * self.wi[i][j]
            self.ah[j] = self.tanh(sum)
        # output activations
        for k in range(self.output):
            sum = 0.0
            for j in range(self.hidden):
                sum += self.ah[j] * self.wo[j][k]
            self.ao[k] = self.tanh(sum)
        self.time_loop = time.time() - t
        print("time: {}".format(self.time_loop))
        return self.ao[:]

    def feedForward(self, network_inputs):
        t = time.time()
        input_activations = np.zeros(self.input)
        input_activations[:-1] = network_inputs
        input_activations = self.tanh(input_activations)
        input_activations[-1] = 1
        hidden_activations = self.tanh(np.dot(input_activations, self.wi))
        output_activations = np.tanh(np.dot(hidden_activations, self.wo))
        self.time_vector = time.time() - t
        print("time: {}".format(self.time_vector))

    def change_weights(self, input_to_hidden, hidden_to_output):
        self.wi = input_to_hidden
        self.wo = hidden_to_output

def main():

    my_mlp = MLP_NeuralNetwork(input=8, hidden=16, output=2)
    mlp_in = np.random.randn(8)
    print(mlp_in)
    output = my_mlp.feedForward(mlp_in)
    out_old = my_mlp.feedForward_old(mlp_in)
    # print(output)

if __name__ == '__main__':
    main()
