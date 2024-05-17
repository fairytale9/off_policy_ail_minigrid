import random
from env import maze
import utils
import exp_demo
from replay_buffer import ExperienceReplay, TransitionReplay
import numpy as np
import time
from tqdm import tqdm, trange


#hyper-parameters
total_steps = int(1e4) #total episodes between agent and env
replay_buffer_size = 128
maze_dim = 3
episode_horizon = 9
batch_size = 32
train_per_step = 4
feature_space = 8
eval_episodes = 5


state_space = maze_dim*maze_dim
feature_encoder = np.random.uniform(low=-1.0, high=1.0, size=(state_space+4, feature_space))


#learning rate
sigma = 10*np.sqrt(2*np.log(4)/(episode_horizon*episode_horizon*total_steps))
eta = 5/np.sqrt(total_steps)

#collect expert trajectories
num_exp_traj = 4
exp_replay_buffer_list = []
for h in range(episode_horizon):
    exp_replay_buffer = ExperienceReplay(num_exp_traj)
    exp_replay_buffer_list.append(exp_replay_buffer)
for exp_traj in range(num_exp_traj):
    env = maze(maze_dim, episode_horizon)
    state = env.reset()
    for step in range(episode_horizon):
        action = exp_policy(state, maze_dim)
        next_state, reward = env.step(action)
        exp_replay_buffer_list[step].append((state, action))
        state = next_state

for num_recent_policies in [1, 4, 32, 128]:
  episodes_per_policy = replay_buffer_size / num_recent_policies
  #batch_size = num_recent_policies
  for seed in range(5):
      random.seed(seed)

      #initialize policy
      policy_list = []
      for h in range(episode_horizon):
          _policy_h = []
          for state in range(state_space):
              _policy_h.append([0.25]*4)
          policy_list.append(_policy_h)

      policy_list = np.array(policy_list)

      #initialize replay buffer
      policy_replay_buffer_list = []

      for h in range(episode_horizon):
          policy_replay_buffer = ExperienceReplay(replay_buffer_size)
          policy_replay_buffer_list.append(policy_replay_buffer)

      #initialize q value and reward function
      q_value = np.zeros((episode_horizon+1,state_space,4), float)
      reward_function = np.zeros((episode_horizon, feature_space), float)

      with tqdm(total=total_steps) as pbar:
          eval_rewards = []
          k = 0
          while(k<total_steps):
              #evaluate
              eval_rewards_list = []
              for episode in range(eval_episodes):
                  episode_reward = 0
                  #initialize env
                  eval_env = maze(maze_dim, episode_horizon)
                  state = eval_env.reset()
                  for step in range(episode_horizon):
                      behavior_policy = policy_list[step, state]
                      action = np.random.choice([0,1,2,3], p=behavior_policy)
                      next_state, reward = eval_env.step(action)
                      episode_reward += reward
                      state = next_state

                  eval_rewards_list.append(episode_reward)

              eval_rewards_list = np.array(eval_rewards_list)
              average_reward = np.mean(eval_rewards_list)
              eval_rewards.append(average_reward)

              #rollout trajectory
              env = maze(maze_dim, episode_horizon)
              state = env.reset()
              for step in range(episode_horizon):
                  behavior_policy = policy_list[step, state]
                  action = np.random.choice([0,1,2,3], p=behavior_policy)
                  next_state, reward = env.step(action)
                  #replay_buffer.add_batch((state, action, next_state))
                  policy_replay_buffer_list[step].append((state, action))
                  state = next_state

              k += 1
              pbar.update(1)


              #train
              if k>=batch_size and k % episodes_per_policy==0:

                  #evaluate
                  _q_value_h = np.zeros((state_space,4), float)
                  for state in range(state_space):
                      for action in range(4):
                          state_action_reps = state_action_encoder([state], [action], state_space, 1, feature_encoder, feature_space)
                          state_action_rep = state_action_reps[0]
                          _q_value_h[state, action] = np.dot(state_action_rep, reward_function[episode_horizon-1, :])
                  q_value[episode_horizon-1] = _q_value_h
                  for step in range(episode_horizon-2, -1, -1):
                      _q_value_h = np.zeros((state_space,4), float)
                      for state in range(state_space):
                          for action in range(4):
                              next_state = get_next_state(maze_dim, state, action)
                              state_action_reps = state_action_encoder([state], [action], state_space, 1, feature_encoder, feature_space)
                              state_action_rep = state_action_reps[0]
                              _q_value_h[state, action] = np.dot(state_action_rep, reward_function[step, :]) + \
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
                          state_action_one_hot_rep = state_action_encoder(states, actions, state_space, batch_size, feature_encoder, feature_space)
                          behavior_policy_gradient = np.mean(state_action_one_hot_rep, axis=0)

                          exp_batch = exp_replay_buffer_list[step].sample(num_exp_traj)
                          exp_states, exp_actions = exp_batch
                          exp_state_action_rep = state_action_encoder(exp_states, exp_actions, state_space, num_exp_traj, feature_encoder, feature_space)
                          exp_policy_gradient = np.mean(exp_state_action_rep, axis=0)

                          _new_reward = reward_function[step] + eta * (exp_policy_gradient - behavior_policy_gradient)
                          new_reward = np.clip(_new_reward, -1, 1)
                          reward_function[step] = new_reward

      log_filename = os.path.join('/content/minigrid_results_S_3/', 'Maze_'+str(maze_dim)+'H_'+str(episode_horizon)+'N_'+str(num_recent_policies)+'seed_'+str(seed))
      np.save(log_filename, eval_rewards)
