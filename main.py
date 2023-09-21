import random
from env import maze
import utils
import exp_demo
from replay_buffer import ExperienceReplay, TransitionReplay
import numpy as np
import time
from tqdm import tqdm, trange


#hyper-parameters
total_updates = int(1e4)
train_per_step = 5

policy_episodes = 1 # how many episodes we collect by following a policy
maze_dim = 5
episode_horizon = 25
num_policy = 4 # how many most recent policies we use for update, when num_policy=1, it is on-policy
batch_size = 4
num_exp_traj = 4 # amount of available expert data
seeds = 5 # number of seeds

state_space = maze_dim*maze_dim

#learning rate
sigma = np.sqrt(2*np.log(4)/(episode_horizon*episode_horizon*total_updates)) # policy update step size
eta = 1/np.sqrt(total_updates) # reward update step size

#get expert demonstrations
exp_replay_buffer_list = []
for h in range(episode_horizon):
    exp_replay_buffer = ExperienceReplay(num_exp_traj)
    exp_replay_buffer_list.append(exp_replay_buffer)
for exp_traj in range(num_exp_traj):
    env = maze(maze_dim, episode_horizon)
    state = env.reset()
    for step in range(episode_horizon):
        action = exp_demo.exp_policy(state, maze_dim)
        next_state, reward = env.step(action)
        exp_replay_buffer_list[step].append((state, action))
        state = next_state


total_rewards_list = []

for seed in range(seeds):
    random.seed(seed)

    #initialize policy
    policy_list = []
    for h in range(episode_horizon):
        _policy_h = []
        for state in range(state_space):
            _policy_h.append([0.25]*4)
        policy_list.append(_policy_h)

    policy_list = np.array(policy_list)

    #initialize behavior policy replay buffer
    policy_replay_buffer_list = []

    for h in range(episode_horizon):
        policy_replay_buffer = ExperienceReplay(policy_episodes*num_policy)
        policy_replay_buffer_list.append(policy_replay_buffer)

    #initialize q function and reward function
    q_value = np.zeros((episode_horizon+1,state_space,4), float)
    reward_function = np.zeros((episode_horizon, state_space*4), float)

    #train
    with tqdm(total=total_updates) as pbar:
        k = 0
        total_rewards = np.zeros(total_updates)
        while(k<total_updates):
            reward_list = []
            #rollout trajectory
            for episode in range(policy_episodes):
                episode_reward = 0
                #initialize env
                env = maze(maze_dim, episode_horizon)
                state = env.reset()
                for step in range(episode_horizon):
                    behavior_policy = policy_list[step, state]
                    action = np.random.choice([0,1,2,3], p=behavior_policy)
                    next_state, reward = env.step(action)
                    episode_reward += reward
                    #replay_buffer.add_batch((state, action, next_state)) #to estimate the world model, here we use the ground truth
                    policy_replay_buffer_list[step].append((state, action))
                    state = next_state
    
                reward_list.append(episode_reward)
    
            reward_list = np.array(reward_list)
            average_reward = np.mean(reward_list)
            total_rewards[k] = average_reward
    
    
            if k>=2e3:
    
                #evaluate
                _q_value_h = np.zeros((state_space,4), float)
                for state in range(state_space):
                    for action in range(4):
                        _q_value_h[state, action] = reward_function[episode_horizon-1, state*4+action]
                q_value[episode_horizon-1] = _q_value_h
                for step in range(episode_horizon-2, -1, -1):
                    _q_value_h = np.zeros((state_space,4), float)
                    for state in range(state_space):
                        for action in range(4):
                            next_state = utils.get_next_state(maze_dim, state, action)
                            _q_value_h[state, action] = reward_function[step, state*4+action] + \
                                      0.9*(np.dot(policy_list[step+1, next_state], q_value[step+1,next_state]))+\
                                      0.1*(np.dot(policy_list[step+1, state], q_value[step+1,state]))
    
                    q_value[step] = _q_value_h
    
                #train
                for i in range(train_per_step):
                    for h in range(episode_horizon):
                        for state in range(maze_dim*maze_dim):
                            _new_policy = policy_list[h, state]*np.exp(sigma*q_value[h, state])
                            new_policy = _new_policy/np.sum(_new_policy)
                            policy_list[h, state] = new_policy
    
                    for step in range(episode_horizon):
                        batch = policy_replay_buffer_list[step].sample(batch_size)
                        states, actions = batch
                        state_action_one_hot_rep = utils.state_action_encoder(states, actions, state_space, batch_size)
                        behavior_policy_gradient = np.mean(state_action_one_hot_rep, axis=0)
    
                        exp_batch = exp_replay_buffer_list[step].sample(num_exp_traj)
                        exp_states, exp_actions = exp_batch
                        exp_state_action_rep = utils.state_action_encoder(exp_states, exp_actions, state_space, num_exp_traj)
                        exp_policy_gradient = np.mean(exp_state_action_rep, axis=0)
    
                        _new_reward = reward_function[step] + eta * (exp_policy_gradient - behavior_policy_gradient)
                        new_reward = np.clip(_new_reward, -1, 1)
                        reward_function[step] = new_reward
    
            k += 1
            pbar.update(1)

    total_rewards_list.append(total_rewards)
