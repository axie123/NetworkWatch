import numpy as np
import torch
import torchvision as tvn
import matplotlib.pyplot as plt
import sklearn.metrics as skm
import sys

def train(network,train_set,loss_func,method,log_interval,epochs):
    try:
        network.train()
        for index, (data, truth) in enumerate(train_set):
            method.zero_grad()
            output = network(data) 
            loss = loss_func(output[0], truth)
            loss.backward()
            method.step()
            if index % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epochs, index * len(data), len(train_set.dataset),
                                                                    100. * index / len(train_set), loss.item()))
        return [output,truth]
    except:
        print('Error occurred during training. Please check errors: ', sys.exc_info()[2])


def test(network,test_set,loss_func):
    try:
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
        return [output, truth]
    except:
        print('Error occurred during testing. Please check error: ', sys.exc_info()[2])


def layer_visuals(layer_num,data_results,ith):
    chosen_layer = data_results[0][layer_num]
    while True:
        print('The dimensions of the layer: ' + str(chosen_layer.shape))
        base = input('Base: ')
        if base == 'exit':
            break
        height = input('Height: ')
        try:
            fig = plt.figure(figsize=(int(base),int(height)))
            for i in range(ith,ith+10):
                image = chosen_layer[i].view(int(height),int(base))
                for j in range(10):
                    plt.subplot(4,5,j + 1)
                    plt.tight_layout()
                    plt.title(j + ith, fontsize = 25)
                    plt.imshow(image.detach(), cmap = 'gray', interpolation = 'none')
                    plt.xticks([])
                    plt.yticks([])
            plt.show()
            break
        except (RuntimeError):
            print('The proposed dimensions do not match that of the output.\n')
        except (ValueError):
            print('Please use integers only.\n')
        except:
            print('Unexpected Error: ' + sys.exc_info()[2] + '\n')

def accuracy_individual_classes(network,classes,test_set):
    try:
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
            print('Accuracy of %s : %2d%%  [%i/%i]' % (classes[i], 100 * class_correct[i] / class_total[i], class_correct[i], class_total[i]))
    except:
        print('There is an error: ' + sys.exc_info()[2] + '\n' )

def saving_textfile(file,pandas_true,directory):
    try:
        if pandas_true == False:
            f = open(directory,'w+')
            for line in file:
                f.writelines(str(list(line.numpy())))
            f.close()
        else:
            file.to_csv(directory,'w+')
    except:
        print('There is an error: ' + sys.exc_info()[2] + '\n')

def weights_biases(network):
    parameters = {}
    for i in network.named_parameters():
        parameters[i[0]] = i[1] 
    specific_parameters = parameters.keys()
    try:
        while(True):
            print('The weights and biases of these layers have been identified: \n')
            for j in specific_parameters:
                print(j)
            wanted_parameter = input('Please enter the wanted parameter or enter 0 to exit. Press E to export a specific parameter. \n')
            if wanted_parameter == '0':
                break
            elif wanted_parameter == 'E' or wanted_parameter == 'e':
                wanted_parameter = input('Please enter the parameter to export: \n')
                data = parameters[str(wanted_parameter)].detach()
                saving_textfile(data,False)
                break
            elif wanted_parameter[-4:] == 'bias':
                print('There are %i biases in this layer. \n' % parameters[wanted_parameter].shape)
                while(True):
                    ith_bias_ith_layer, end = input('Enter the bias range. Enter 0 x to exit: \n').split()
                    if end == 'x':
                        break
                    else:
                        print('\n')
                        print(parameters[wanted_parameter][int(ith_bias_ith_layer):int(end)].detach())
                        print('\n')
            else:
                print('\n')
                print('There are %d nodes and %d weights from each node. \n' % (parameters[wanted_parameter].shape[0], parameters[wanted_parameter].shape[1]))
                while(True):
                    ith_node, ith_weight_ith_node = input('Enter the node number and input weights. Enter 0 x to exit: \n').split()
                    if ith_weight_ith_node == 'x':
                        break
                    else:
                        print('\n')
                        print(parameters[wanted_parameter][int(ith_node)][int(ith_weight_ith_node):int(end)].detach())
                        print('\n')
        print('Closed.')
    except:
        print('You entered an invalid input. Please try again.\n')

def save_NN(network,method,directory_network,directory_method):
    torch.save(network.state_dict(), directory_network)
    torch.save(method.state_dict(), directory_method)
    
def load_NN(network,method,directory_network,directory_method):
    network.load_state_dict(torch.load(directory_network))
    method.load_state_dict(torch.load(directory_network))

