import numpy as np
import pandas as pd
import torch
import torchvision as tvn


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
            if key[-4:] == 'bias':
                self.bias[key] = self.weights_biases[i]
            else:
                self.weights[key] = self.weights_biases[i]
        self.bias_key = self.bias.keys()
        self.weights_key = self.weights.keys()

    def weights_to_node(self, layer, node):
        # Gets all the weights coming into the node.
        hidden_layer = self.weights[layer]
        node_weights = hidden_layer[node - 1]
        return node_weights

    def node_to_node_weight(self, start_node, end_layer, end_node):
        # Weight going between two nodes.
        e_layer = self.weights[end_layer]
        n_to_n_weight = e_layer[end_node - 1][start_node - 1]
        return n_to_n_weight

    def weight_range(self, layer, node, start_weight, end_weight):
        # Get a range of nodes coming into it.
        return self.weights[layer][node - 1][start_weight - 1:end_weight]

    def bias_of_node(self, layer, node):
        # Get a node bias.
        hidden_layer = self.bias[layer]
        node_bias = hidden_layer[node - 1]
        return node_bias

    def bias_range(self, layer,start_node, end_node):
        # Get a range of node biases.
        return self.bias[layer][start_node - 1:end_node]





