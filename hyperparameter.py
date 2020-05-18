"Will Add Comments Later."
class hyperparameter:
    def __init__(self, train_ratio, lr=0.05, batch_size=64, reg=0.1, momentum = 0.05, lossf = 'mse', act = 'relu', type=None):
        self.train_ratio = train_ratio
        self.test_ratio = 1 - self.train_ratio

        self.batch_size = batch_size
        self.reg = reg
        self.lr = lr
        self.momentum = momentum
        self.loss_function = lossf
        self.activation = act

        self.models = {}
        self.type = type

    def default_package(self, name):
        self.models[name] = {'train-ratio': self.train_ratio,
                             'test-ratio': self.test_ratio,
                             'batch size': self.batch_size,
                             'regularization': self.reg,
                             'lr': self.lr,
                             'momentum': self.momentum,
                             'loss_function': self.loss_function,
                             'activation': self.activation
                             }

    def custom_package(self, name, lr, batch_size, reg, momentum, lossf, act, train_test=None):
        if train_test == None:
            train_para = {'train-ratio': self.train_ratio,
                          'test-ratio': self.test_ratio,
                          'batch size': batch_size,
                          'regularization': reg,
                          'lr': lr,
                          'momentum': momentum,
                          'loss_function': lossf,
                          'activation': act
                          }
        else:
            train_para = {'train-ratio': train_test,
                          'test-ratio': 1 - train_test,
                          'batch size': batch_size,
                          'regularization': reg,
                          'lr': lr,
                          'momentum': momentum,
                          'loss_function': lossf,
                          'activation': act
                          }
        self.models[name] = train_para

    def edit_package(self, name, hyper, val):
        self.models[name][hyper] = val

    def delete_package(self, name):
        deleted = self.models[name]
        del self.models[name]
        return deleted

hyps = hyperparameter(0.8)

hyps.default_package('ada')
hyps.custom_package('hello', 0.2, 50, 0, 0.1, 'log', 'tanh')
hyps.custom_package('people', 0.2, 50, 0, 0.1, 'log', 'tanh', train_test=0.7)
print(hyps.models)

hyps.edit_package('people', 'lr', 0.7)
print(hyps.models)

hyps.delete_package('people')
print(hyps.models)




