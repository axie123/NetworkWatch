import json

class running_metrics:
    def __init__(self, id):
        self.id = id
        self.training_interval = []
        self.validation_interval = []
        self.testing_interval = []
        self.training_loss = []
        self.validation_loss = []
        self.testing_loss = []
        self.training_accuracy = []
        self.validation_accuracy = []
        self.testing_accuracy = []

        self.model_evals = {'id': self.id,
                            'training_interval': self.training_interval,
                            'validation_interval': self.validation_interval,
                            'testing_interval': self.testing_interval,
                            'training_loss': self.training_loss,
                            'validation_loss': self.validation_loss,
                            'testing_loss': self.testing_loss,
                            'training_accuracy': self.training_accuracy,
                            'validation_accuracy': self.validation_accuracy,
                            'testing_accuracy': self.testing_accuracy
                            }

    def training_update(self, loss, acc):
        self.model_evals['training_loss'].append(loss)
        self.model_evals['training_accuracy'].append(acc)

    def validation_update(self, loss, acc):
        self.model_evals[name]['validation_loss'].append(loss)
        self.model_evals[name]['validation_accuracy'].append(acc)

    def testing_update(self, loss, acc):
        self.model_evals[name]['testing_loss'].append(loss)
        self.model_evals[name]['testing_validation'].append(acc)


def dump_into_transferfile(eval_data):
    with open('eval_data.txt', 'w') as file:
        file.write(json.dumps(eval_data))
