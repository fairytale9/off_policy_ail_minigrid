import random

def exp_policy(state, maze_dim):
    row = state // maze_dim + 1
    column = state % maze_dim + 1
    rn = random.random()
    if rn<0.5:
        action = 1
    else:
        action = 2
    if row==maze_dim:
        action = 1
    if column==maze_dim:
        action = 2
    return action
