import numpy as np

class VNN:
    def __init__(self, network):
        # Organizes all the parameters of the VNN.
        #
        # Separates the name of the parameter from parameter.
        self.parameters = np.array(list(network.named_parameters()))
        self.keys = self.parameters[:, 0]
        self.weights_biases = self.parameters[:, 1]

        self.bias = {}  # Biases
        self.weights = {}  # Weights
        # Separating biases and weights.
        for i, key in enumerate(self.keys):
            if 'bias' in key:
                self.bias[key] = self.weights_biases[i]
            else:
                self.weights[key] = self.weights_biases[i]
        self.bias_key = self.bias.keys()
        self.weights_key = self.weights.keys()

    def weights_to_node(self, layer, node):
        # Gets all the weights coming into the node.
        hidden_layer = self.weights.get(layer)
        node_weights = hidden_layer[node - 1]
        return node_weights

    def node_to_node_weight(self, start_node, end_node, end_layer):
        # Weight going between two nodes.
        e_layer = self.weights.get(end_layer)
        n_to_n_weight = e_layer[end_node - 1][start_node - 1]
        return n_to_n_weight

    def weight_range(self, layer, node, start_weight, end_weight):
        # Get a range of nodes coming into it.
        return self.weights.get(layer)[node - 1][start_weight - 1:end_weight]

    def bias_of_node(self, layer, node):
        # Get a node bias.
        hidden_layer = self.bias.get(layer)
        node_bias = hidden_layer[node - 1]
        return node_bias

    def bias_range(self, layer,  start_node, end_node):
        # Get a range of node biases.
        return self.bias.get(layer)[start_node - 1:end_node]


class CNN(VNN):
    def __init__(self, network):
        # Organizes all the parameters of the CNN.
        #
        # Separates the name of the parameter from parameter.
        #self.parameters = np.array(list(network.named_parameters()))
        #self.keys = self.parameters[:, 0]
        self.kernels_weights_biases = self.parameters[:, 1]

        self.bias = {}  # Biases
        self.kernels = {}  # Kernels
        self.weights = {}  # Weights of fully connected layers.
        # Separating biases, kernels, and weights.
        for i, key in enumerate(self.keys):
            if 'bias' in key:
                self.bias[key] = self.kernels_weights_biases[i]
            elif len(self.kernels_weights_biases[i].shape) >= 3:
                self.kernels[key] = self.kernels_weights_biases[i]
            else:
                self.weights[key] = self.kernels_weights_biases[i]
        self.bias_key = self.bias.keys()
        self.kernels_key = self.kernels.keys()
        self.weights_key = self.weights.keys()

    def kernel_filter(self, conv_layer, filter_group_num, prev_filter_num):
        # View the kernel filter.
        filter = self.kernels.get(conv_layer)[filter_group_num-1]\
        [prev_filter_num-1]
        return filter

    def kernel_filter_range(self, conv_layer, filter_group_num, 
                            prev_filter_num_start,
                            prev_filter_num_end):
        # Gives a range of kernel filters.
        filter_range = self.kernels.get(conv_layer)[filter_group_num-1]\
        [prev_filter_num_start-1:prev_filter_num_end]
        return filter_range

    def fc_weights_to_node(self, layer, node):
        # Gets all the weights coming into the node.
        hidden_layer = self.weights.get(layer)
        node_weights = hidden_layer[node - 1]
        return node_weights

    def fc_node_to_node_weight(self, start_node, end_node, end_layer):
        # Weight going between two nodes.
        e_layer = self.weights.get(end_layer)
        n_to_n_weight = e_layer[end_node - 1][start_node - 1]
        return n_to_n_weight

    def fc_weight_range(self, layer, node, start_weight, end_weight):
        # Get a range of nodes coming into it.
        return self.weights.get(layer)[node - 1][start_weight - 1:end_weight]

    def fc_bias_of_node(self, layer, node):
        # Get a node bias.
        hidden_layer = self.bias.get(layer)
        node_bias = hidden_layer[node - 1]
        return node_bias

    def fc_bias_range(self, layer, start_node, end_node):
        # Get a range of node biases.
        return self.bias.get(layer)[start_node - 1:end_node]

# I don't know much about the GRU, so this is just a skeleton so far.
class GRU(VNN):
    def __init__(self, network):
        # Organizes all the parameters of the GRU.
        #
        # Separates the name of the parameter from parameter.
        #self.parameters = np.array(list(network.named_parameters()))
        #self.keys = self.parameters[:, 0]
        self.gru_paras_embedding = self.parameters[:, 1]

        self.bias = {}  # Biases
        self.gru_ih = {}
        self.gru_hh = {}
        self.embedding = {}
        # Separating GRU units from the embedding.
        for i, key in enumerate(self.keys):
            if 'bias' in key:
                self.bias[key] = self.gru_paras_embedding[i]
            elif 'ih' in key:
                self.gru_ih[key] = self.gru_paras_embedding[i]
            elif 'hh' in key:
                self.gru_hh[key] = self.gru_paras_embedding[i]
            else:
                self.embedding[key] = self.gru_paras_embedding[i]
        self.bias_key = self.bias.keys()
        self.gru_ih_key = self.gru_ih.keys()
        self.gru_hh_key = self.gru_hh.keys()

class output_visualizer:
    def __init__(self, x, w, b):
        self.x = x
        self.w = w
        self.b = b

    def layer_output(self, activation):
        z = activation(np.matmul(np.transpose(self.w), self.x) + self.b)
        return z
    







