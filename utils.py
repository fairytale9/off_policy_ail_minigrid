import numpy as np

def state_action_encoder(states, actions, state_space, batch_size, feature_encoder, feature_space):
    state_action_rep = np.zeros((batch_size, feature_space), dtype=float)
    for idx in range(batch_size):
        state = states[idx]
        action = actions[idx]
        _state_action_rep = np.zeros(state_space+4, dtype=float)
        _state_action_rep[state_space+action] = 1
        _state_action_rep[state] = 1
        _state_action_rep = np.dot(_state_action_rep, feature_encoder)
        state_action_rep[idx, :] = _state_action_rep
    return state_action_rep

def get_next_state(maze_dim, state, action):
    row = state // maze_dim + 1
    column = state % maze_dim + 1
    if action==0:
        row -= 1
    elif action==2:
        row += 1
    elif action==1:
        column += 1
    elif action==3:
        column -= 1
    else:
        raise ValueError("invalid action")

    if row<1:
        row = 1
    if row>maze_dim:
        row = maze_dim
    if column<1:
        column = 1
    if column>maze_dim:
        column = maze_dim

    next_state = (row-1)*maze_dim+column-1

    return next_state
