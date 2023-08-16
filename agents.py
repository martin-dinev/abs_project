import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import logging

logging.getLogger("tensorflow").setLevel(logging.ERROR)

import numpy as np
import random
from tensorflow.keras.layers import Input, Dense, Concatenate, Reshape, Conv2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError, MSE
from tensorflow import reduce_mean, convert_to_tensor, squeeze, float32, GradientTape

import tensorflow as tf

os.environ['AUTOGRAPH_VERBOSITY'] = '1'
tf.autograph.set_verbosity(0)

from tqdm import tqdm
from kaggle_environments import make
from deep_q_learning import DQN, DDQN, DuelingDQN
from deep_q_learning_modified import MyDQN
from my_utils import mean_reward, my_evaluate, my_evaluate_parallel


class MyAgent:
    agent: None | DQN | DDQN | DuelingDQN | MyDQN

    def __init__(self, state_space_shape, num_actions, save_name):
        self.num_actions = num_actions
        self.state_space_shape = state_space_shape
        self.save_name = save_name
        self.kaggle_agent = self.get_kaggle_agent()
        self.parallel_kaggle_agent = self.get_parallel_kaggle_agent()
        self.agent = None

    def build_agent(self):
        pass

    def get_agent(self):
        return self.agent

    @staticmethod
    def process_state(_state):
        return np.asarray(_state.board).reshape((6, 7))

    @staticmethod
    def process_states(_states):
        return np.array([np.asarray(_state.board).reshape((6, 7)) for _state in _states])

    @staticmethod
    def process_reward(_reward):
        return (4 if _reward == 1 else 1 if _reward == 0 else -1) if _reward is not None else -4

    @staticmethod
    def process_rewards(_rewards):
        return np.array([MyAgent.process_reward(_reward) for _reward in _rewards])

    @staticmethod
    def process_action(_action):
        return round(_action)

    @staticmethod
    def process_actions(_actions):
        return [int(a) for a in np.rint(_actions)]

    def my_train(self, opponent="random", num_episodes=None, epsilon=None, epsilon_decay=None, epsilon_min=None,
                 save_iter=None, prev_count="max", save_name=None, update_target_steps=None):
        proceed, save_name, epsilon, epsilon_min, epsilon_decay, prev_count = \
            self.pre_train_steps(opponent, num_episodes, epsilon, epsilon_decay, epsilon_min, save_iter, prev_count,
                                 update_target_steps, save_name)
        if not proceed:
            return

        env = make("connectx", debug=False)
        trainer = env.train([None, opponent])
        sum_steps = 0
        for i in tqdm(range(num_episodes)):
            state = self.process_state(trainer.reset())
            done = False
            epsilon = max(epsilon - epsilon_decay, epsilon_min)
            steps = 0
            while not done:
                steps = steps + 1
                action = self.process_action(self.agent.get_action(state, epsilon))
                new_state, reward, done, _ = trainer.step(action)
                new_state = self.process_state(new_state)
                self.agent.update_memory(state, action, self.process_reward(reward), new_state, done)
                state = new_state
            sum_steps += steps
            self.agent.train()
            if sum_steps >= update_target_steps:
                self.agent.update_target_model()
                sum_steps = 0
            if i == num_episodes - 1 or (save_iter is not None and (i + 1) % save_iter == 0):
                self.agent.save(save_name, i + 1 + prev_count)

    def my_train_parallel(self, opponent="random", num_episodes=None, epsilon=None, epsilon_decay=None,
                          epsilon_min=None,
                          save_iter=None, prev_count="max", save_name=None, update_target_steps=None, trainers_count=100):
        proceed, save_name, epsilon, epsilon_min, epsilon_decay, prev_count = \
            self.pre_train_steps(opponent, num_episodes, epsilon, epsilon_decay, epsilon_min, save_iter, prev_count,
                                 update_target_steps, save_name)
        if not proceed:
            return

        trainers = np.array([make("connectx", debug=False).train([None, opponent])
                             for _ in range(min(trainers_count, num_episodes))])
        for trainer in trainers:
            trainer.reset()
        init_state = self.process_state(make("connectx", debug=False).train([None, opponent]).reset())
        states = np.repeat(np.array([init_state]), len(trainers), axis=0)
        sum_steps = 0
        tq_bar = tqdm(range(num_episodes))
        save_count = 0
        while len(trainers) > 0:
            actions = self.process_actions(self.agent.get_actions(states, epsilon))
            outcomes = np.array([trainer.step(int(action)) for trainer, action in zip(trainers, actions)])
            rewards = self.process_rewards(outcomes[:, 1])
            new_observations = outcomes[:, 0]
            new_states = self.process_states(new_observations)
            dones = outcomes[:, 2].astype(bool)
            self.agent.update_memory_multiple(states, actions, rewards, new_states, dones)
            sum_steps += len(trainers)

            finished_count = np.count_nonzero(dones)
            required = num_episodes - len(trainers) - tq_bar.n

            finished = np.where(dones)
            remaining = np.where(~dones)

            if finished_count > 0:
                if required >= finished_count:
                    new_observations[finished] = np.array([trainer.reset() for trainer in trainers[finished]])
                    new_states[finished] = self.process_states(new_observations[finished])
                else:
                    active_trainers = trainers[remaining]
                    reset_trainers = trainers[finished][:required]
                    trainers = np.concatenate((active_trainers, reset_trainers))
                    active_new_states = new_states[remaining]
                    for trainer in reset_trainers:
                        trainer.reset()
                    reset_new_states = np.repeat(np.array([init_state]), len(reset_trainers), axis=0)
                    new_states = np.concatenate((active_new_states, reset_new_states))
                tq_bar.update(finished_count)

            for _ in range(np.count_nonzero(dones)):
                if sum_steps >= update_target_steps:
                    self.agent.update_target_model()
                    sum_steps = 0
                self.agent.train()
                epsilon = max(epsilon - epsilon_decay, epsilon_min)

            if (save_iter is not None and tq_bar.n - save_count >= save_iter) or tq_bar.n == num_episodes:
                self.agent.save(save_name, tq_bar.n + prev_count)
                save_count = tq_bar.n

            states = new_states
        tq_bar.close()

    def get_kaggle_agent(self):
        def kaggle_agent(_observation, _):
            _state = self.process_state(_observation)
            _action = self.agent.get_action(_state, 0)
            _action = self.process_action(_action)
            return _action

        return kaggle_agent

    def get_parallel_kaggle_agent(self):
        def parallel_kaggle_agent(_observations):
            _states = self.process_states(_observations)
            _actions = self.agent.get_actions(_states, 0)
            _actions = self.process_actions(_actions)
            return _actions

        return parallel_kaggle_agent

    def my_test(self, agent_name, at_episode, _num_episodes=100, opponent_agent=None):
        opponent_agent = opponent_agent if opponent_agent is not None else agent_name
        self.agent.load(self.save_name, at_episode)
        _rewards, _time = my_evaluate("connectx", [self.kaggle_agent, opponent_agent], _num_episodes=_num_episodes)
        _mean_reward = mean_reward([r[0] for r in _rewards])
        print(self.save_name + " vs " + agent_name + ":", _mean_reward, " time: ",_time)

    def my_test_parallel(self, agent_name, at_episode="max", _num_episodes=100, opponent_agent=None, batch_size=256):
        opponent_agent = opponent_agent if opponent_agent is not None else agent_name
        if at_episode == "max":
            at_episode = max([int(f.split("_")[2].split(".")[0]) for f in os.listdir("models") if
                              f.split("_")[1].startswith(self.save_name)] + [0])
            if at_episode == 0:
                print("No models found for " + self.save_name)
                return
        self.agent.load(self.save_name, at_episode)
        _rewards, _time = my_evaluate_parallel("connectx", [self.kaggle_agent, opponent_agent], self,
                                        _num_episodes=_num_episodes, batch_size=batch_size)
        _mean_reward = mean_reward(_rewards)
        print(self.save_name + " vs " + agent_name + ":", _mean_reward, " time:", _time)

    def pre_train_steps(self, opponent, num_episodes, epsilon, epsilon_decay, epsilon_min, save_iter, prev_count,
                        update_target_steps, save_name):
        if None in [opponent, num_episodes, epsilon, update_target_steps]:
            print("Can't have None values for some attributes")
            return False, None, None, None, None, None

        if save_name is None:
            save_name = self.save_name

        if epsilon == "default":
            epsilon = 1
            epsilon_min = 0.1
            epsilon_decay = 1.25 * (epsilon - epsilon_min) / num_episodes

        if prev_count == "max":
            prev_count = max([int(f.split("_")[2].split(".")[0]) for f in os.listdir("models") if
                              f.split("_")[1].startswith(save_name)] + [0])

        if prev_count != 0:
            self.agent.load(save_name, prev_count)
        else:
            self.agent = self.build_agent()

        return True, save_name, epsilon, epsilon_decay, epsilon_min, prev_count


def build_method_1(agent):
    i = Input(shape=agent.state_space_shape)
    r = Reshape(target_shape=agent.state_space_shape + (1,))(i)

    cv = Conv2D(16, kernel_size=(2, 1))(r)
    cv = Conv2D(16, kernel_size=(3, 1))(cv)
    cv = Conv2D(1, kernel_size=(3, 1))(cv)
    cv = Reshape(target_shape=(agent.num_actions,))(cv)
    cv = Dense(7)(cv)

    ch = Conv2D(16, kernel_size=(2, 1))(r)
    ch = Conv2D(16, kernel_size=(3, agent.num_actions))(ch)
    ch = Conv2D(16, kernel_size=(3, 1))(ch)
    ch = Reshape(target_shape=(16,))(ch)
    ch = Dense(agent.num_actions)(ch)

    cd = Conv2D(agent.num_actions, kernel_size=(2, 3))(r)
    cd = Conv2D(agent.num_actions, kernel_size=(3, 3))(cd)
    cd = Conv2D(agent.num_actions, kernel_size=(3, 3))(cd)
    cd = Reshape(target_shape=(agent.num_actions,))(cd)
    cd = Dense(agent.num_actions)(cd)

    output_layer = cv + ch + cd
    output_layer = Dense(agent.num_actions)(output_layer)
    output_layer = Dense(agent.num_actions)(output_layer)

    m = Model(inputs=i, outputs=output_layer)
    m.compile(optimizer=Adam(learning_rate=agent.learning_rate), loss=MeanSquaredError())

    return m


class Agent1(MyAgent):

    def __init__(self, learning_rate, discount_factor, memory_size, batch_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate

        self.agent = self.build_agent()

    def build_model(self):
        return build_method_1(self)

    def build_agent(self):
        model = self.build_model()
        target_model = self.build_model()
        agent = DQN(self.state_space_shape, self.num_actions, model, target_model, self.learning_rate,
                    self.discount_factor, self.batch_size, self.memory_size)
        return agent


class Agent2(Agent1):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build_agent(self):
        model = self.build_model()
        target_model = self.build_model()
        agent = MyDQN(self.state_space_shape, self.num_actions, model, target_model, self.learning_rate,
                      self.discount_factor, self.batch_size, self.memory_size)
        return agent
