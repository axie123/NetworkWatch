from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QMainWindow, QWidget, QLabel, QLineEdit, QPushButton
from PyQt5.QtCore import QSize
import sys
import numpy as np
import torch
import torchvision as tvn
import matplotlib.pyplot as plt

nn_params = {'nodes_layer_0': 1024,        
             'nodes_layer_1': 512,  
             'nodes_layer_2': 256,
             'nodes_layer_3': 128
             }

prop_params = {'ReLU_1': 0.2,               
               'ReLU_2': 0.2,
               'ReLU_3': 0.2,
               'ReLU_4': 0.2,
               'Drop_1': 0.25,
               'Drop_2': 0.25,
               'Drop_3': 0.25,
               'Drop_4': 0.25
               }

class NN(torch.nn.Module):
    def __init__(self,nn_params, prop_params):
        super(NN, self).__init__()
        initial_features = 784
        final_features = 10
        self.hidden_layerin = torch.nn.Sequential(torch.nn.Linear(initial_features,nn_params['nodes_layer_0']),torch.nn.LeakyReLU(prop_params['ReLU_1']),torch.nn.Dropout(prop_params['Drop_1']))
        self.hidden_layer1 = torch.nn.Sequential(torch.nn.Linear(nn_params['nodes_layer_0'],nn_params['nodes_layer_1']),torch.nn.LeakyReLU(prop_params['ReLU_2']),torch.nn.Dropout(prop_params['Drop_2']))
        self.hidden_layer2 = torch.nn.Sequential(torch.nn.Linear(nn_params['nodes_layer_1'],nn_params['nodes_layer_2']),torch.nn.LeakyReLU(prop_params['ReLU_3']),torch.nn.Dropout(prop_params['Drop_3']))
        self.hidden_layer3 = torch.nn.Sequential(torch.nn.Linear(nn_params['nodes_layer_2'],nn_params['nodes_layer_3']),torch.nn.LeakyReLU(prop_params['ReLU_4']),torch.nn.Dropout(prop_params['Drop_4']))
        self.layer_out = torch.nn.Linear(nn_params['nodes_layer_3'],final_features)

    def forward(self, x):
        x = x.view(-1,784)
        x_in = self.hidden_layerin(x)
        x_1 = self.hidden_layer1(x_in)
        x_2 = self.hidden_layer2(x_1)
        x_3 = self.hidden_layer3(x_2)
        x_4 = self.layer_out(x_3)
        x_out = torch.nn.functional.log_softmax(x_4,dim=1)
        return [x_out,x_in,x_1,x_2,x_3]

network = NN(nn_params, prop_params)
loss = torch.nn.functional.nll_loss
optimizer = torch.optim.SGD

#=================================================================================================================================================================================

'train_set,test_set,classes'

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

def layer_loss_analysis(layer_num,data_results,loss_func):
    chosen_layer = data_results[0][layer_num]
    print('The dimensions of the layer: ' + str(chosen_layer.shape) +'\n')
    try:
        if len(chosen_layer.shape) == 2:
            layer = torch.nn.functional.log_softmax(chosen_layer, dim=1)
            loss = loss_func(layer, data_results[1])
            print('Layer Number: {}, Loss: {:.6f}\n'.format(layer_num, loss)) 
        else:
            print('Invalid')
    except:
        print('There is a problem with the information you entered: ' + sys.exc_info()[2] + '\n')

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

#===============================================================================================================================

class MainWindow(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)

        self.setMinimumSize(QSize(900, 500))    
        self.setWindowTitle("NetView v2.0 UI") 

        # The main menu.
        self.nameLabel = QLabel(self)
        self.nameLabel.setText('Main Menu')
        self.nameLabel.move(40, 10)
        b1 = QPushButton('Train Model',self)
        b1.resize(300,32)
        b1.move(40, 50)
        b1.clicked.connect(self.train_test)
        
        # Training Conditions.
        self.e = QLabel(self)
        self.e.setText('Epochs')
        self.e.move(400, 50)
        self.epochs = QLineEdit(self)
        self.epochs.move(400, 80)
        self.epochs.resize(100, 32)
        self.tr = QLabel(self)
        self.tr.setText('Train Size')
        self.tr.move(520, 50)
        self.train_size = QLineEdit(self)
        self.train_size.move(520, 80)
        self.train_size.resize(100, 32)
        self.t = QLabel(self)
        self.t.setText('Test Size')
        self.t.move(640, 50)
        self.test_size = QLineEdit(self)
        self.test_size.move(640, 80)
        self.test_size.resize(100, 32)
        self.r = QLabel(self)
        self.r.setText('Learning Rate')
        self.r.move(760, 50)
        self.rate = QLineEdit(self)
        self.rate.move(760, 80)
        self.rate.resize(100, 32)
        self.p = QLabel(self)
        self.p.setText('Momentum')
        self.p.move(400, 140)
        self.mom = QLineEdit(self)
        self.mom.move(400, 170)
        self.mom.resize(100, 32)
        self.l = QLabel(self)
        self.l.setText('Log Interval')
        self.l.move(520, 140)
        self.log_int = QLineEdit(self)
        self.log_int.move(520, 170)
        self.log_int.resize(100, 32)
        self.s = QLabel(self)
        self.s.setText('Random Seed')
        self.s.move(640, 140)
        self.seed = QLineEdit(self)
        self.seed.move(640, 170)
        self.seed.resize(100, 32)

        # Analysis  Components:
        self.n = QLabel(self)
        self.n.setText('Network')
        self.n.move(400, 270)
        self.network = QLineEdit(self)
        self.network.move(400, 300)
        self.network.resize(200, 32)
        self.ls = QLabel(self)
        self.ls.setText('Loss Function')
        self.ls.move(620, 270)
        self.loss = QLineEdit(self)
        self.loss.move(620, 300)
        self.loss.resize(200, 32)
        self.o = QLabel(self)
        self.o.setText('Optimization')
        self.o.move(400, 350)
        self.method = QLineEdit(self)
        self.method.move(400, 380)
        self.method.resize(300, 32)
        

    def train_test(self):
        train_loader = torch.utils.data.DataLoader(tvn.datasets.MNIST('/files/', train = True, download = True,
                                                            transform = tvn.transforms.Compose([tvn.transforms.ToTensor(),
                                                            tvn.transforms.Normalize((0.1307,), (0.3081,))])),batch_size=int(self.train_size.text()), shuffle=True)
        test_loader = torch.utils.data.DataLoader(tvn.datasets.MNIST('/files/', train = False, download = True,
                                                            transform = tvn.transforms.Compose([tvn.transforms.ToTensor(),
                                                            tvn.transforms.Normalize((0.1307,), (0.3081,))])),batch_size=int(self.test_size.text()), shuffle=True)
        optimizer = globals()[self.method.text()](globals()[self.network.text()].parameters(), lr= float(self.rate.text()),momentum= float(self.mom.text()))
        train_losses = []
        train_counter = []
        test_losses = []
        test_counter = [i*len(train_loader.dataset) for i in range(int(self.epochs.text()) + 1)]
        test(globals()[self.network.text()],test_loader,globals()[self.loss.text()])
        for epoch in range(1, int(self.epochs.text()) + 1):
            training_data_results = train(globals()[self.network.text()],train_loader,globals()[self.loss.text()],optimizer,int(self.log_int.text()),int(self.epochs.text()))
            testing_data_results = test(globals()[self.network.text()],test_loader,globals()[self.loss.text()])
        print('Finished running.')
        self.close()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    mainWin = MainWindow()
    mainWin.show()
    sys.exit( app.exec_() )



