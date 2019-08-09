import numpy as np
import pandas as pd
import torch
import torchvision as tvn
import matplotlib.pyplot as plt
import sklearn.metrics as skm

# Need a:

# Pre-made network that needs to be named 'network'
# Pre-loaded and normalized datasets (if needed).
# Training and testing sets need to be there.
# Method of SGD.
# Loss function with fill name.
# Need a number of epochs.
# Log interval
# Predetermined classifications.

def train(train_set,loss_func,method,log_interval,epochs):
    network.train() # training the network.
    for index, (data, truth) in enumerate(train_set):
        method.zero_grad()
        output = network(data) 
        loss = loss_func(output[0], truth)
        loss.backward()
        method.step()
        if index % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epochs, index * len(data), len(train_set.dataset),
                                                                    100. * index / len(train_set), loss.item()))
    return [output,truth,'training']


def test(test_set,loss_func):
    test_losses = []
    network.eval()
    total_test_loss = 0
    total_correct = 0
    with torch.no_grad():
        for index, (data, truth) in enumerate(test_set):
            output = network(data)
            test_loss = loss_func(output[0], truth, size_average = False)
            total_test_loss += test_loss.item()
            pred = output[0].data.max(1, keepdim = True)[1]
            total_correct += pred.eq(truth.data.view_as(pred)).sum()
    avg_test_loss = total_test_loss / len(test_set.dataset)
    test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(avg_test_loss, total_correct, len(test_set.dataset),
            100. * total_correct / len(test_set.dataset)))
    return [output, truth,'testing']


def layer_visuals(layer_num,data_results,ith):
    if data_results[2] == 'training':
        chosen_layer = data_results[0][layer_num]
        print(chosen_layer.shape)
        base = input('Base: \n')
        height = input('Height: \n')
        fig = plt.figure(figsize=(10,2))
        for i in range(ith,ith+5):
            image = chosen_layer[i].view(int(height),int(base))
            for j in range(5):
                plt.subplot(1,5,j + 1)
                plt.imshow(image.detach(), cmap = 'gray', interpolation = 'none')
                plt.xticks([])
                plt.yticks([])
        plt.show()
    elif data_results[2] == 'testing':
        chosen_layer = data_results[0][layer_num]
        print(chosen_layer.shape)
        base = input('Base: \n')
        height = input('Height: \n')
        fig = plt.figure(figsize=(10,2))
        for i in range(ith,ith+5):
            image = chosen_layer[i].view(int(height),int(base))
            for j in range(5):
                plt.subplot(1,5,j + 1)
                plt.imshow(image.detach(), cmap = 'gray', interpolation = 'none')
                plt.xticks([])
                plt.yticks([])
        plt.show()

def layer_loss_analysis(layer_num,data_results,loss_func):
    if data_results[2] == 'training':
        chosen_layer = data_results[0][layer_num]
        print(chosen_layer.shape)
        if len(chosen_layer.shape) == 2:
            layer = torch.nn.functional.log_softmax(chosen_layer, dim=1)
            loss = loss_func(layer, data_results[1])
            print('Layer Number: {}, Loss: {:.6f}\n'.format(layer_num, loss)) 
        else:
            print('Invalid')
    if data_results[2] == 'testing':
        chosen_layer = data_results[0][layer_num]
        print(chosen_layer.shape)
        if len(chosen_layer.shape) == 2:
            layer = torch.nn.functional.log_softmax(chosen_layer, dim=1)
            loss = loss_func(layer, data_results[1])
            print('Layer Number: {}, Loss: {:.6f}\n'.format(layer_num, loss))
        else:
            print('Invalid')

def accuracy_individual_classes(classes,test_set):
    class_correct = list(0. for i in range(len(classes)))
    class_total = list(0. for i in range(len(classes)))
    for data in test_set:
        images, labels = data
        outputs = network(images)
        _, predicted = torch.max(outputs[0], 1)
        correct = (predicted == labels).squeeze()
        for i in range(len(labels)):
            label = labels[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1
    for i in range(len(classes)):
        print('Accuracy of %s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))

def saving_textfile(file,pandas_true):
    filename = input('Enter filename: \n')
    directory = input('Enter a directory: \n')
    if pandas_true == False:
        f = open(str(directory)+'\\'+str(filename),'w+')
        for line in file:
            f.writelines(str(list(line.numpy())))
        f.close()
    else:
        file.to_csv(str(directory)+'\\'+str(filename),'w+')

def weights_biases():
    parameters = {}
    for i in network.named_parameters():
        parameters[i[0]] = i[1] 
    specific_parameters = parameters.keys()
    while(True):
        print('The weights and biases of these layers have been identified: \n')
        for j in specific_parameters:
            print(j)
        print('\n')
        wanted_parameter = input('Please enter the wanted parameter or enter 0 to exit. Press E to export a specific parameter. \n')
        print('\n')
        if wanted_parameter == '0':
            break
        elif wanted_parameter == 'E' or wanted_parameter == 'e':
            wanted_parameter = input('Please enter the parameter to export: \n')
            data = parameters[str(wanted_parameter)].detach()
            saving_textfile(data,False)
            break
        else:
            ith_node = input('Enter the node number: \n')
            while(True):
                ith_weight_ith_node, end = input('Enter the input weights: \n').split()
                if end == 'x':
                    break
                else:
                    print('\n')
                    print(parameters[wanted_parameter][int(ith_node)][int(ith_weight_ith_node):int(end)].detach())
                    print('\n')
    print('Closed.')
