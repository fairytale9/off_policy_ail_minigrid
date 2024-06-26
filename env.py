import random

class maze():
    def __init__(self, dim, horizon, stochasticity=0.1, start_pos=(1,1)):
        self.dim = dim
        self.horizon = horizon
        self.state = start_pos
        self.stochasticity = stochasticity

    def reset(self):
        self.state = (1, 1)
        state = (self.state[0]-1)*self.dim+self.state[1]-1
        return state

    def step(self, action):
        row, column = self.state
        rn = random.random()
        if rn>self.stochasticity: # with some probability, the agent will stay at the same place no matter what action it takes
            if action==0: # action: 0 up 1 right 2 down 3 left
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
            if row>self.dim:
                row = self.dim
            if column<1:
                column = 1
            if column>self.dim:
                column = self.dim

        next_state = (row-1)*self.dim+column-1
        self.state = (row, column)
        reward = 0.0
        if row==self.dim and column==self.dim:
            reward = 1.0

        return next_state, reward
