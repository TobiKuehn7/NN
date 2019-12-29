import numpy
import scipy.special
# initialise the neural network class


class neuralNetwork:

    # initialise the neural network (NN)
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # set number of nodes in each layer
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        self.wih = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

        # set learning rate
        self.lr = learningrate

        self.activation_function = lambda x: scipy.special.expit(x)

        pass

    # train NN
    def train(self):
        pass

    # query NN
    def query(self, inputs_list):
        inputs = numpy.array(inputs_list, ndmin=2).T
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        return final_outputs
