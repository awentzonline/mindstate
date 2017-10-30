import random

from mindstate.genome import Genome


class Agent(object):
    def __init__(self, mind):
        self.model = mind
        self.genome = Genome(self.model.num_weights)
        # create some temp genomes for crossover
        self.genome_a = Genome(self.model.num_weights)
        self.genome_b = Genome(self.model.num_weights)

    def randomize(self, gene_weight_ratio, freq_weight_ratio, init_value_range):
        self.genome.randomize(
             gene_weight_ratio,
             max(1, int(self.genome.num_weights * freq_weight_ratio)),
             init_value_range)

    def crossover(self, best_genomes):
        parents = random.sample(best_genomes, 2)
        # strip scores
        best_genomes = [genome for genome, _ in best_genomes]
        self.genome_a.deserialize(best_genomes[0])
        self.genome_b.deserialize(best_genomes[1])
        self.genome.child(self.genome_a, self.genome_b)

    def mutate(self, index_sigma=1., value_sigma=1.):
        self.genome.mutate(index_sigma=index_sigma, value_sigma=value_sigma)

    def update_model(self):
        weights = self.genome.decode(self.model.weights.shape)
        #print(self.model.weights - weights)
        self.model.weights = weights

    def load_genome(self, genome_data):
        self.genome.deserialize(genome_data)

    def summary(self):
        print(self.model)
        print(self.genome)
