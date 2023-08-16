from kaggle_environments import make
from keras import Model
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.processors import Processor
from keras.layers import Dense, Dropout, Input, Reshape
import numpy as np

i = Input(shape=(1, 42))
x = Dense(42, activation='relu')(i)
x = Dropout(0.5)(x)
x = Dense(24, activation='relu')(x)
x = Dropout(0.25)(x)
o = Dense(7, activation='linear')(x)
o = Reshape((7,))(o)
model = Model(inputs=i, outputs=o)


class ConnectXProcessor(Processor):
    def process_observation(self, observation):
        return np.array(observation['board'])

    def process_state_batch(self, batch):
        return batch

    def process_reward(self, reward):
        return -10 if reward is None else reward

    def process_action(self, action):
        return int(action)


policy = EpsGreedyQPolicy()
processor = ConnectXProcessor()
memory = SequentialMemory(limit=50000, window_length=1)
agent = DQNAgent(model=model, policy=policy, memory=memory, nb_actions=7, nb_steps_warmup=100, target_model_update=1e-2,
                 processor=processor)
agent.compile(optimizer=Adam(), metrics=['mae'])

env = make("connectx", debug=False)
trainer = env.train([None, "random"])

agent.fit(trainer, nb_steps=50000, visualize=False, verbose=1, nb_max_episode_steps=1000)
