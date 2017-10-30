import numpy as np

from mindstate.activations import sigmoid, swish
from mindstate.state_mapper import Mapping, StateMapper
from .base import BaseModel


class RNNModel(BaseModel):
    def __init__(self, state_dims, inputs, outputs):
        super(RNNModel, self).__init__(
            state_dims, inputs, outputs, [
                Mapping('state_update', state_dims, f=swish)
            ])

    def step(self, *inputs):
        # contribute input to current state
        for input, input_name in zip(inputs, self.input_names):
            self.state += self.state_mapper.map_input(input_name, input)
        # update state
        self.state = self.state_mapper.map_input('state_update', self.state)
        # calculate outputs
        results = []
        for output in self.outputs:
            results.append(
                self.state_mapper.map_output(output.name, self.state))
        return results


if __name__ == '__main__':
    state_dims = 5
    inputs = [
        Mapping('vision', 5),
    ]
    outputs = [
        Mapping('actions', 4),
    ]
    mind = RNNModel(state_dims, inputs, outputs)
    modelrandomize_weights()
    input_vs = []
    for input in inputs:
        input_vs.append(np.random.uniform(0, 1, (input.dims,)))
    for i in range(5):
        output_vs = model.step(*input_vs)
        # check that the outputs are of the correct shape
        for output_v, output in zip(output_vs, outputs):
            assert output_v.shape == (output.dims,)
    print('tests passed')
