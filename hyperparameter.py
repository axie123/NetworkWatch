import torch

#model = app.ModelLog.query.get_or_404(app.model_num)
#print(model)

class hyperparameter:
    def __init__(self, model):
        # This helps the user to organize all the essential hyperparameters
        # for machine learning models into different groups. Each class call
        # can be for a different type of training and within them the different
        # hyperparams for that type of training can be stored.
        #
        # The following the essential hyperparameters.
        self.train_ratio = model.train_ratio
        self.test_ratio = model.test_ratio
        self.batch_size_train = model.train_batch_size
        self.batch_size_test = model.test_batch_size

        self.epochs = model.epochs
        self.lr = model.lr
        self.momentum = model.mom
        self.reg = model.reg

        self.loss_function = model.loss_func
        self.optimizer = model.opt
        self.opt_args = model.opt_args
        self.rdm_seed = model.rdm_seed
        self.log_interval = model.log

    def activation_func(self, network, **kwargs):
        if model.optimizer == 'base':
            return torch.optim.Optimizer(network.parameters())
        elif model.optimizer == 'adadelta':
            return torch.optim.Adadelta(network.parameters(), **kwargs)
        elif model.optimizer == 'adagrad':
            return torch.optim.Adagrad(network.parameters(), **kwargs)
        elif model.optimizer == 'adam':
            return torch.optim.Adam(network.parameters(), **kwargs)
        elif model.optimizer == 'sgd':
            return torch.optim.SGD(network.parameters(), **kwargs)
        elif model.optimizer == 'asgd':
            return torch.optim.ASGD(network.parameters(), **kwargs)
        elif model.optimizer == 'rms_prop':
            return torch.optim.RMSprop(network.parameters(), **kwargs)
        elif model.optimizer == 'rprop':
            return torch.optim.Rprop(network.parameters(), **kwargs)
        elif model.optimizer == 'lbfgs':
            return torch.optim.LBFGS(network.parameters(), **kwargs)
        else:
            return torch.optim.SGD(network.parameters(), **kwargs)

network = hyperparameter(model)