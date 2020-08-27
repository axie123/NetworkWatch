import time as time
from model import evaluation
from model import name

tr_interval = [1, 50, 100, 150, 200, 250, 300]
tt_interval = [1, 50, 100, 150, 200, 250, 300]
tr_loss = [1, 60, 150, 200, 250, 300]
tt_loss = [15339, 0, 10483, 7503, 5089, 3092, 1234]
tr_acc = [1, 60, 150, 200, 250, 300]
tt_acc = [15339, 0, 10483, 7503, 5089, 3092, 1234]

class running_metrics:
    def __init__(self):
        self.training_ep_speed = []
        self.validation_ep_speed = []
        self.testing_ep_speed = []
        self.training_loss = []
        self.validation_loss = []
        self.testing_loss = []
        self.training_accuracy = []
        self.validation_accuracy = []
        self.testing_accuracy = []
        self.training_time = 0
        self.validation_time = 0
        self.testing_time = 0

        self.model_evals = {}


    def eval_package(self, name):
        self.model_evals[name] = {'training_ep_speed': self.training_ep_speed,
                                  'validation_ep_speed': self.validation_ep_speed,
                                  'testing_ep_speed': self.testing_ep_speed,
                                  'training_loss': self.training_loss,
                                  'validation_loss': self.validation_loss,
                                  'testing_loss': self.testing_loss,
                                  'training_accuracy': self.training_accuracy,
                                  'validation_accuracy': self.validation_accuracy,
                                  'testing_accuracy': self.testing_accuracy,
                                  'training_time': self.training_time,
                                  'validation_time': self.validation_time,
                                  'testing_time': self.testing_time
                                 }

    def running_speed(self, stat):
        if stat == 'start':
            start = time.process_time()
        elif stat == 'finish':
            end = time.process_time()

    def training_update(self, loss, acc, time, name):
        self.model_evals[name]['training_loss'].append(loss)
        self.model_evals[name]['training_accuracy'].append(acc)
        self.model_evals[name]['training_time'] += time

    def validation_update(self, loss, acc, time, name):
        self.model_evals[name]['validation_loss'].append(loss)
        self.model_evals[name]['validation_accuracy'].append(acc)
        self.model_evals[name]['validation_time'] += time

    def testing_update(self, loss, acc, time, name):
        self.model_evals[name]['testing_loss'].append(loss)
        self.model_evals[name]['testing_validation'].append(acc)
        self.model_evals[name]['testing_time'] += time

class running_parameters:
    def __init__(self):
        self.training_grad = []
        self.validation_grad = []
        self.testing_grad = []

    def gradients(self, network, i, parameter):
        grad_list = []
        for para in i:
            para.backward()
            grad_list(parameter.grad)
        return grad_list


tr_loss = evaluation.eval_package(name).model_evals[name]['training_loss']
tt_loss = evaluation.eval_package(name).model_evals[name]['testing_loss']
tr_acc = evaluation.eval_package(name).model_evals[name]['training_accuracy']
tt_acc = evaluation.eval_package(name).model_evals[name]['testing_accuracy']