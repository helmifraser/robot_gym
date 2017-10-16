import numpy as np

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
        print(self.ai)
        self.ah = [1.0] * self.hidden
        self.ao = [1.0] * self.output
        # create randomized weights
        self.wi = np.random.randn(self.input, self.hidden)
        self.wo = np.random.randn(self.hidden, self.output)
        # create arrays of 0 for changes
        self.ci = np.zeros((self.input, self.hidden))
        self.co = np.zeros((self.hidden, self.output))

    def tanh(self, x, derivative=False):
        if (derivative == True):
            return (1 - (x ** 2))
        return np.tanh(x)

    def feedForward(self, inputs):
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
        return self.ao[:]

    def change_weights(self, input_to_hidden, hidden_to_output):
        self.wi = input_to_hidden
        self.wo = hidden_to_output

# def main():
#     input_size = 8
#     hidden_size = 16
#     output_size = 2
#
#     my_mlp = MLP_NeuralNetwork(input=input_size, hidden=hidden_size, output=output_size)
#     # mlp_in = np.random.randn(8)
#     mlp_in = np.ones(8)
#
#     print("Input: {}".format(mlp_in))
#
#     output = my_mlp.feedForward(mlp_in)
#     print("Output 1: {}".format(output))
#
#     print("Changing weights")
#     wi = np.random.randn(input_size + 1, hidden_size)
#     wo = np.random.randn(hidden_size, output_size)
#     my_mlp.change_weights(wi, wo)
#
#     output = my_mlp.feedForward(mlp_in)
#     print("Output 2: {}".format(output))
#
#
# if __name__ == '__main__':
#     main()
