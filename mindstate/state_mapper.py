import numpy as np


class Mapping(object):
    """A named projection to or from state space."""
    def __init__(self, name, dims, f=None):
        self.name = name
        self.dims = dims
        self.f = f
        self.index = -1

    def __repr__(self):
        return "<Mapping: {} d={}, i={}>".format(self.name, self.dims, self.index)


class StateMapper(object):
    """Maps names to submatrices and provides tools for
    projecting vectors in and out of a shared state space.

    `self.weights` contains the following submatrices:
     * One state update matrix (state_dims x state_dims).
     * N (state_dims x mapping_dim_i) matrices representing
     projects to and from state space.
    """
    def __init__(self, state_dims, mappings):
        if not isinstance(mappings, (list, tuple)):
            mappings = [mappings]
        total_dims = 0
        self.mapping_lookup = {}
        for index, mapping in enumerate(mappings):
            total_dims += mapping.dims
            mapping.index = index
            self.mapping_lookup[mapping.name] = mapping
        self.weights = np.zeros((total_dims, state_dims))

    def get_mapping_weights(self, name):
        mapping = self.mapping_lookup[name]
        return self.weights[mapping.index:mapping.index + mapping.dims]

    def map_input(self, name, v):
        w = self.get_mapping_weights(name)
        y = np.dot(v, w)
        mapping = self.mapping_lookup[name]
        if mapping.f:
            y = mapping.f(y)
        return y

    def map_output(self, name, v):
        w = self.get_mapping_weights(name)
        y = np.dot(v, w.T)
        mapping = self.mapping_lookup[name]
        if mapping.f:
            y = mapping.f(y)
        return y

    def randomize_weights(self):
        self.weights = np.random.normal(0., 1., self.weights.shape)

    @property
    def state_dims(self):
        return self.weights.shape[-1]

    @property
    def num_weights(self):
        return np.prod(self.weights.shape)


if __name__ == '__main__':
    state_dims = 3
    inputs = [
        Mapping('state_update', state_dims),
        Mapping('vision', 5),
    ]
    outputs = [
        Mapping('actions', 4),
    ]

    ps = StateMapper(state_dims, inputs + outputs)
    # test input mapping
    for input in inputs:
        v = np.random.uniform(0, 1, (input.dims,))
        w = ps.get_mapping_weights(input.name)
        assert w.shape == (input.dims, state_dims)
        s_in = ps.map_input(input.name, v)
        assert s_in.shape == (state_dims,)
    # test output mapping
    for output in outputs:
        v = np.random.uniform(0, 1, (state_dims,))
        w = ps.get_mapping_weights(output.name)
        assert w.shape == (output.dims, state_dims)
        out = ps.map_output(output.name, v)
        assert out.shape == (output.dims,)
    print('tests passed')
