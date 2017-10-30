import random
import time

import gym
import numpy as np
import redis
from scipy.misc import imresize, imsave

from mindstate.activations import sigmoid, softmax, swish
from mindstate.optimizers.agent import Agent
from mindstate.genepool import GenePool
from mindstate.gym.viewer import SimpleAttentiveImageViewer
from mindstate.models.rnn import RNNModel
from mindstate.optimizers.async_genepool import Optimizer
from mindstate.state_mapper import Mapping


class Experiment(object):
    def __init__(self, config):
        self.config = config

    def run(self):
        self.genepool = GenePool(redis_params=self.config.redis_params)
        self.environment = gym.make(config.env)
        self.environment.env.viewer = SimpleAttentiveImageViewer(self.config.attention_size)
        self.attention_size = self.config.attention_size
        state_shape = (1, self.attention_size, self.attention_size)
        num_hidden = self.config.num_hidden
        num_actions = self.environment.action_space.n

        self.model = RNNModel(
            num_hidden,
            [Mapping('vision', np.prod(state_shape), f=swish)],
            [
                Mapping('actions', num_actions),
                Mapping('attention', 3),
            ])
        self.agent = Agent(self.model)
        self.agent.randomize(
            self.config.gene_weight_ratio, self.config.freq_weight_ratio,
            self.config.v_init)
        self.agent.update_model()
        print(self.agent.summary())

        if self.config.exhibition:
            self.exhibition()
        else:
            worker = Optimizer(config, self.agent, self.genepool)
            worker.run(self.run_episode)

    def exhibition(self):
        while True:
            best_genomes = self.genepool.top_n(self.config.num_best)
            best_genome, _ = random.choice(best_genomes)
            self.agent.load_genome(best_genome)
            self.agent.update_model()
            print(self.agent.genome.summary())
            print('Starting episode')
            reward, num_steps = self.run_episode()
            print('Reward {} in {} steps'.format(reward, num_steps))
            self.genepool.report_score(self.agent.genome, reward)

    def run_episode(self):
        total_reward = 0.
        num_steps = 0
        observation = self.environment.reset()
        done = False
        player_ready = False
        start_delay = np.random.randint(self.config.random_start)
        attention_params = np.array([0., 0., 1.])  # full screen
        while not done:
            if self.config.render or self.config.exhibition:
                self.environment.render()
            observation, coords = self.attend_image(observation, attention_params, self.attention_size)
            self.environment.env.viewer.rect = coords
            # to single channel
            observation = np.mean(observation, axis=2, keepdims=True).transpose(2, 0, 1)
            if False and np.random.uniform() < 0.01:
                imsave('observation.png', observation.transpose(1, 2, 0)[:,:,0].astype(np.uint8))
            observation = observation / 255. - 0.5
            # take action
            player_ready = start_delay <= 0
            if player_ready:
                actions, attention_params = self.model.step(observation.ravel())
                attention_params = sigmoid(attention_params)
                #print(attention_params)
                if True or not self.config.exhibition:  # use probabilistic policy
                    actions = softmax(actions)
                    action = np.random.choice(
                        np.arange(actions.shape[0]), p=actions)
                else:
                    action = np.argmax(actions)
                if np.random.uniform(0, 1) < 0.05 and False:
                    print(actions)
                    print(action)
            else:
                action = np.random.randint(self.environment.action_space.n)
                start_delay -= 1
            observation, reward, done, info = self.environment.step(action)
            total_reward += reward
            num_steps += 1
        return total_reward, num_steps

    def attend_image(self, img, params, output_size):
        u, v, size = params
        size = 0.2
        #print(params)
        size = np.clip(size, 0.05, 1.0)
        img_height, img_width, img_channels = img.shape
        att_width = int(size * img_width)
        att_height = int(size * img_height)
        cx = int(u * (img_width - att_width))
        cy = int(v * (img_height - att_height))
        cx = np.clip(cx, 0, img_width - att_width)
        cy = np.clip(cy, 0, img_height - att_height)

        #print(img.shape)
        window = img[cy:cy + att_height, cx:cx + att_width, :]
        #print('window shape', window.shape, params, cy, cx, att_height, att_width)
        window = imresize(
            window.astype(np.float32), (output_size, output_size),
            interp='bicubic')
        return window, (cx, cy, att_width, att_height)


def main(config):
    experiment = Experiment(config)
    experiment.run()


if __name__ == '__main__':
    import argparse
    import multiprocessing

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--env', default='SpaceInvaders-v0')
    argparser.add_argument('--render', action='store_true')
    argparser.add_argument('--num-hidden', type=int, default=256)
    argparser.add_argument('--exhibition', action='store_true')
    argparser.add_argument('--random-start', type=int, default=30)
    argparser.add_argument('--attention-size', type=int, default=50)
    Optimizer.add_config_to_parser(argparser)
    config = argparser.parse_args()

    if config.clear_store:
        genepool = GenePool(redis_params=config.redis_params)
        genepool.clear()

    if config.exhibition:
        main(config)
    else:
        processes = []
        for agent_i in range(config.num_agents):
            p = multiprocessing.Process(target=main, args=(config,))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
