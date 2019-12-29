from NN.neuralNetwork import neuralNetwork

# number of nodes in layers
input_nodes = 3
hidden_nodes = 3
output_nodes = 3

# learning rate of NN
learning_rate = 0.3

# create instance
n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
print(n.query([1.0, 0.5, -1.5]))
