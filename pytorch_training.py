import time as time

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

import plotly.graph_objects as go

fig = go.Figure()
class visualize:
    def add_line(x, y, label, color, width):
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name=label, line=dict(color=color, width=width)))

    def add_line_marker(x, y, label, color, width):
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines+markers', name=label))

    def add_marker(x, y, label, color, width):
        fig.add_trace(go.Scatter(x=x, y=y, mode='markers', name=label))

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
