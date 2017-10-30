import base64
import pickle
import random
import zlib

import numpy as np
from scipy.fftpack import dct, idct


class Gene(object):
    def __init__(self, index, value):
        self.index = index
        self.value = value

    def clone(self):
        return Gene(self.index, self.value)

    def __str__(self):
        return '<Gene i={} v={}>'.format(self.index, self.value)


class Genome(object):
    """Search matrices via DCT coefficients."""
    def __init__(self, num_weights, genes=None, rng=None):
        self.rng = rng or np.random.RandomState()
        self.num_weights = num_weights
        self.genes = genes or []

    def randomize(self, gene_weight_ratio, max_index, value_range, weight_f='log'):
        """Create a random assortment of genes."""
        self.genes = []
        if weight_f == 'linear':
            base_freqs = self.num_weights
        else:  # look up a numpy function
            base_freqs = getattr(np, weight_f)(self.num_weights)
        num_genes = max(2, int(gene_weight_ratio * base_freqs))
        #print(max_index, value_range)
        for i in range(num_genes):
            gene = Gene(
                self.rng.randint(0, max_index),
                self.rng.uniform(*value_range))
            self.genes.append(gene)

    def decode(self, original_shape):
        """Decode the genome into a matrix of the given shape."""
        target = np.zeros(np.prod(original_shape))
        for gene in self.genes:
            target[gene.index] = gene.value
        target = target.reshape(original_shape)
        len_shape = len(original_shape)
        kwargs = dict(norm='ortho')
        if len_shape == 1:
            out = idct(target, **kwargs)
        elif len_shape == 2:
            out = idct(idct(target.T, **kwargs).T, **kwargs)
        elif len_shape >= 3:
            shape = (np.prod(original_shape[:-1]), original_shape[-1])
            target = target.reshape(shape)
            out = idct(idct(target.T, **kwargs).T, **kwargs)
            out = out.reshape(original_shape)
        return out

    def mutate(self, p_index=0.1, p_value=0.8, index_sigma=1., value_sigma=1.):
        for gene in self.genes:
            if self.rng.uniform() < p_index:
                gene.index += int(self.rng.normal(0., index_sigma))
                gene.index = np.clip(gene.index, 0, self.num_weights - 1)
            if self.rng.uniform() < p_value:
                gene.value += self.rng.normal(0., value_sigma)

    def split(self, random=True):
        if random:
            p = self.rng.randint(len(self.genes))
        else:
            p = len(self.genes) // 2
        left = [g.clone() for g in self.genes[:p]]
        right = [g.clone() for g in self.genes[p:]]
        return left, right

    def cut(self, p_cut):
        num_genes = len(self.genes)
        if self.rng.uniform() < p_cut * num_genes:
            pivot = self.rng.randint(num_genes)
            left = [g.clone() for g in self.genes[:pivot]]
            right = [g.clone() for g in self.genes[pivot:]]
            return left, right
        return ([g.clone() for g in self.genes],)

    def splice(self, other, p_cut, p_splice):
        this_cuts = self.cut(p_cut)
        other_cuts = other.cut(p_cut)
        results = []
        current = []
        for chromosome in itertools.chain(this_cuts, other_cuts):
            if not current or self.rng.uniform() < p_splice:
                current += chromosome
            elif current:
                results.append(current)
                current = []
        if current:
            results.append(current)
        return results

    def child(self, a, b, p_splice=0.5):
        left_a, right_a = a.split()
        left_b, right_b = b.split()
        p_splice_next = 1.0
        gs = list(filter(None, (left_a, left_b, right_a, right_b)))
        if None in gs:
            print("FUUUGGGGG");asf
        result = []
        self.rng.shuffle(gs)
        for g in gs:
            if self.rng.uniform() < p_splice_next:
                result += g
            p_splice_next *= p_splice
        self.genes = result

    def __str__(self):
        return '<Genome w={} {}>'.format(
            self.num_weights, ', '.join(str(g) for g in self.genes))

    def summary(self):
        return 'Genome nw={} ng={} mv={}'.format(
            self.num_weights, len(self.genes),
            np.mean([g.value for g in self.genes])
        )

    def serialize(self):
        d = pickle.dumps(self)
        d = zlib.compress(d)
        d = base64.b64encode(d)
        return d

    def deserialize(self, data):
        g = base64.b64decode(data)
        g = zlib.decompress(g)
        g = pickle.loads(g)
        self.num_weights = g.num_weights
        self.genes = g.genes

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['rng']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.rng = np.random.RandomState()


if __name__ == '__main__':
    m = np.zeros((4, 4))
    num_weights = np.prod(m.shape)
    gn = Genome(num_weights=num_weights)
    gn.randomize(0.05, num_weights, (0, 1), weight_f='log')
    print(gn)
    s = gn.serialize()
    print(s)
    gn.deserialize(s)
    print(gn)
    n = gn.decode(m.shape)
    print(n)
