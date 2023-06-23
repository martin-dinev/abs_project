from kaggle_environments import make
from tqdm.notebook import tqdm
import numpy as np


def my_train(agent, opponent="random", num_episodes=None, process_state=None, process_reward=None, process_action=None,
             epsilon=None, epsilon_decay=None, epsilon_min=None, save_iter=None, prev_count=0, save_name=None):
    if None in [opponent, num_episodes, process_state, process_reward, process_action, epsilon, epsilon_min,
                epsilon_decay] or (save_iter is not None and save_name is None):
        print("Can't have None values for some attributes")
        return

    env = make("connectx", debug=False)
    trainer = env.train([None, opponent])

    for i in tqdm(range(num_episodes)):
        state = process_state(trainer.reset())
        done = False
        epsilon = max(epsilon - epsilon_decay, epsilon_min)
        while not done:
            action = process_action(agent.get_action(state, epsilon))
            new_state, reward, done, _ = trainer.step(action)
            new_state = process_state(new_state)
            agent.update_memory(state, action, process_reward(reward), new_state, done)
            state = new_state
        agent.train()
        agent.update_target_model()
        if save_iter is not None and (i + 1) % save_iter == 0:
            agent.save(save_name, i + 1 + prev_count)


def my_evaluate(environment, agents, _num_episodes=100):
    e = make(environment)
    _rewards = [[] for i in range(_num_episodes)]
    for i in tqdm(range(_num_episodes)):
        last_state = e.run(agents)[-1]
        _rewards[i] = [_state.reward for _state in last_state]
    return _rewards


def my_evaluate_parallel(environment, kaggle_agents, agent, _num_episodes=100, batch_size=256):
    _rewards = []
    tq_bar = tqdm(total=_num_episodes)
    trainers = np.array(
        [make(environment).train([None, kaggle_agents[1]]) for i in range(min(batch_size, _num_episodes))])
    observations = np.array([trainer.reset() for trainer in trainers])
    while len(trainers) > 0:
        states = agent.process_states(observations)
        actions = agent.agent.get_actions(states, 0)
        actions = agent.process_actions(actions)

        outcomes = np.array([trainer.step(int(action)) for trainer, action in zip(trainers, actions)])
        dones = outcomes[:, 2].astype(bool)

        rewards = outcomes[:, 1][np.where(dones)]
        _rewards = _rewards + rewards.tolist()
        tq_bar.update(np.count_nonzero(dones))

        trainers = trainers[np.where(~dones)]
        observations = outcomes[:, 0][np.where(~dones)]

        # print(len(trainers), np.where(~outcomes[:, 2]), np.where(outcomes[:, 2]), outcomes[:, 2], ~outcomes[:, 2])

        if len(trainers) < batch_size and tq_bar.n + len(trainers) < _num_episodes:
            count = min(batch_size - len(trainers), _num_episodes - tq_bar.n - len(trainers))
            new_trainers = np.array([make(environment).train([None, kaggle_agents[1]]) for _ in range(count)])
            trainers = np.append(trainers, new_trainers)
            observations = np.append(observations, [trainer.reset() for trainer in new_trainers])

    tq_bar.close()
    return _rewards


def mean_reward(_rewards):
    wins = sum(1 for r in _rewards if r == 1)
    losses = sum(1 for r in _rewards if r == -1)
    mistakes = sum(1 for r in _rewards if r is None)
    opponent_mistakes = sum(1 for r in _rewards if r == 0)
    return "W: " + str(wins) + "; L: " + str(losses) + "; M: " + str(mistakes) + "; O: " + str(opponent_mistakes)
