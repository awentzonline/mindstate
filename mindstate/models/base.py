import numpy as np

from mindstate.state_mapper import Mapping, StateMapper


class BaseModel(object):
    def __init__(self, state_dims, inputs, outputs, aux):
        self.state_dims = state_dims
        self.inputs = inputs
        self.outputs = outputs
        self.aux = aux
        self.state_mapper = StateMapper(
            state_dims, inputs + outputs + aux)
        self.state = np.random.uniform(0, 1, (state_dims,))

    def step(self, *inputs):
        raise NotImplementedError('Implement `step`')

    def randomize_weights(self):
        self.state_mapper.randomize_weights()

    def reset_state(self):
        self.state = np.random.uniform(0, 1, (state_dims,))

    @property
    def input_names(self):
        return [input.name for input in self.inputs]

    @property
    def num_weights(self):
        return self.state_mapper.num_weights

    @property
    def weights(self):
        return self.state_mapper.weights

    @weights.setter
    def weights(self, weights):
        print('setting weights')
        self.state_mapper.weights = weights

    def __repr__(self):
        return '{}: {}, {}, {}'.format(
            self.__class__.__name__, self.inputs, self.outputs, self.aux
        )
