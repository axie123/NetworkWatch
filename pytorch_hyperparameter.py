import torch

class hyperparameter:
    def __init__(self,
                 train_ratio, epochs=10, lr=0.05, batch_size_train=64,
                 batch_size_test=1000, reg=0.1,
                 momentum=0.05, lossf='mse', optimizer='sgd',
                 opt_args={'lr': 0.01, 'momentum': 0.05}, rdm_seed=1,
                 type=None):
        # This helps the user to organize all the essential hyperparameters
        # for machine learning models into different groups. Each class call
        # can be for a different type of training and within them the different
        # hyperparams for that type of training can be stored.
        #
        # The following the essential hyperparameters.
        self.train_ratio = train_ratio
        self.test_ratio = 1 - self.train_ratio
        self.batch_size_train = batch_size_train
        self.batch_size_test = batch_size_test

        self.epochs = epochs
        self.lr = lr
        self.momentum = momentum
        self.reg = reg

        self.loss_function = lossf
        self.optimizer = optimizer
        self.opt_args = opt_args
        self.rdm_seed = rdm_seed
        self.log_interval = 10

        self.models = {}  # Keeps bundles of training hyperparameters.
        self.type = type

    def activation_func(self, network, optimizer, **kwargs):
        if optimizer == 'base':
            return torch.optim.Optimizer(network.parameters())
        elif optimizer == 'adadelta':
            return torch.optim.Adadelta(network.parameters(), **kwargs)
        elif optimizer == 'adagrad':
            return torch.optim.Adagrad(network.parameters(), **kwargs)
        elif optimizer == 'adam':
            return torch.optim.Adam(network.parameters(), **kwargs)
        elif self.optimizer == 'sgd':
            return torch.optim.SGD(network.parameters(), **kwargs)
        elif optimizer == 'asgd':
            return torch.optim.ASGD(network.parameters(), **kwargs)
        elif optimizer == 'rms_prop':
            return torch.optim.RMSprop(network.parameters(), **kwargs)
        elif optimizer == 'rprop':
            return torch.optim.Rprop(network.parameters(), **kwargs)
        elif optimizer == 'lbfgs':
            return torch.optim.LBFGS(network.parameters(), **kwargs)
        else:
            return torch.optim.SGD(network.parameters(), **kwargs)

    def default_package(self, name, network):
        # Keeps all the default parameters into a bundle.
        self.models[name] = {'train-ratio': self.train_ratio,
                             'test-ratio': self.test_ratio,
                             'epochs': self.epochs,
                             'training batch size': self.batch_size_train,
                             'test batch size': self.batch_size_test,
                             'regularization': self.reg,
                             'lr': self.lr,
                             'momentum': self.momentum,
                             'loss_function': self.loss_function,
                             'optimizer': self.optimizer,
                             'opt_args': self.opt_args,
                             'rdm_seed': self.rdm_seed,
                             'network': network
                             }
        torch.manual_seed(self.rdm_seed)
        self.models[name]['optimizer'] = self.activation_func(
            self.models[name]['network'], self.optimizer,
            **self.models[name]['opt_args'])

    def custom_package(self, name, network, epochs, lr, batch_size_train,
                       batch_size_test, reg, momentum,
                       lossf, opt, opt_args, rdm, train_test=None):
        # Custom hyperparameter bundles for additional exploration.
        if train_test == None:
            train_para = {'train-ratio': self.train_ratio,
                          'test-ratio': self.test_ratio,
                          'epochs': epochs,
                          'training batch size': batch_size_train,
                          'test batch size': batch_size_test,
                          'regularization': reg,
                          'lr': lr,
                          'momentum': momentum,
                          'loss_function': lossf,
                          'optimizer': opt,
                          'opt_args': opt_args,
                          'rdm_seed': rdm,
                          'network': network
                          }
        else:
            train_para = {'train-ratio': train_test,
                          'test-ratio': 1 - train_test,
                          'epochs': epochs,
                          'training batch size': batch_size_train,
                          'test batch size': batch_size_test,
                          'regularization': reg,
                          'lr': lr,
                          'momentum': momentum,
                          'loss_function': lossf,
                          'optimizer': opt,
                          'opt_args': opt_args,
                          'rdm_seed': rdm,
                          'network': network
                          }
        train_para['optimizer'] = self.activation_func(train_para['network'],
                                                       opt_args, **train_para[
                'opt_args'])
        self.models[name] = train_para
        torch.manual_seed(rdm)

    def edit_package(self, name, hyper, val):
        # Change hyperparameters in existing bundles.
        self.models[name][hyper] = val

    def delete_package(self, name):
        # Deleting unwanted hyperparameter bundles.
        deleted = self.models.get(name)
        if deleted == None:
            return None
        del self.models[name]
        return deleted





