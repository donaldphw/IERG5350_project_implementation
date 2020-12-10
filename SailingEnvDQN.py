import sys
from contextlib import closing
import random
import numpy as np
from io import StringIO
from utils import *
from gym import utils, Env, spaces
from gym.utils import seeding
from gym.envs.toy_text import discrete
from gym.utils import seeding
from collections import deque



LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

MAPS = {
#     "4x4": [
#         "SFFF",
#         "FHFH",
#         "FFFH",
#         "HFFG"
#     ],
    "8x8": [
        "SWWWWWWD",
        "WWWOOWWW",
        "WWWOOWWW",
        "WWWWWWWW",
        "WOOWWWWW",
        "WWWWWOOW",
        "WWWWWOOW",
        "DWWWWWWD"
    ],
    "16x16": [
        "SWWWWWWWWWWWWWWD",
        "WWWWWWWWWWWWWWWW",
        "WWWWWWWWWWWOOWWW",
        "WWWWOOOWWWWOOWWW",
        "WWWWOOOWWWWWWWWW",
        "WWWWWWWWWWWWWWWW",
        "WOOWWWWWWWWWWOOO",
        "WOOWWWWWWWWWWOOO",
        "WWWWWWWWWWWWWWWW",
        "WWWWWWOOOWWWWWWW",
        "WWWWWWOOOWWWWWWW",
        "WWWWWWWWWWWWWWWW",
        "WOOWWWWWWWWWWWWW",
        "WOOWWWWWWWOOOWWW",
        "WWWWWWWWWWOOOWWW",
        "DWWWWWWWWWWWWWWD"
    ],
}

rewards_dict = {
    "8x8":
    {
        7 : 200,
        56 : 400,
        63 : 1000
    },
    "16x16":
    {
        15: 4000.0,
        240: 8000.0,
        255 : 20000.0
    }
    
}



def categorical_sample(prob_n, np_random):
    """
    Sample from categorical distribution
    Each row specifies class probabilities
    """
    prob_n = np.asarray(prob_n)
    csprob_n = np.cumsum(prob_n)
    return (csprob_n > np_random.rand()).argmax()

def get_destination(MAP):
            destination = []
            row = len(MAP)
            col = len(MAP[row-1])

            for i in range(row):
                for j in range(col):

                    newletter = MAP[i][j]
                    if newletter == "D":

                        destination.append(i*col + j)
            return destination




class SailingEnvDQN():

    metadata = {'render.modes': ['human', 'ansi']}
    def __init__(self, config):
#         if desc is None and map_name is None:
#             desc = generate_random_map()
#         elif desc is None:
#             desc = MAPS[map_name]
        
        self.map_name = config["map_name"]
        desc = MAPS[self.map_name]
        is_slippery=config["is_slippery"]
        self.current_step = 0
        
        self.total_steps = config["total_steps"] 
        self.destinations = get_destination(desc)
        self.total_destinations = len(self.destinations)
        self.destinations_dict = {D: False for D in self.destinations}
        self.num_reached_destinations = 0
        
        if config["is_random_env"] == False:
            self.random_seed = config["random_seed"]
            random.seed(self.random_seed)
            
        self.desc = desc = np.asarray(desc, dtype='c')
        self.nrow, self.ncol = nrow, ncol = desc.shape
        self.reward_range = (0, self.total_destinations)
        self.initial_state = np.array([0,0])
        self.current_state = self.initial_state
        self.nA = 4
        self.nS = 2
        self.action_space = spaces.Discrete(self.nA)
        self.observation_space = spaces.Discrete(self.nS)
        self.seed()
        self.isd = np.array(desc == b'S').astype('float64').ravel()
        self.isd /= self.isd.sum()
        
    def transition_dynamics(self, action, state):
        # given the action (direction), calculate the next state (UAV current position)
        assert action in self.action_space
        row, col = state[0], state[1]
        next_state = list((row, col))
        if action == 0:
            # move up
            next_state[1] = max(col - 1, 0)
        if action == 1:
            # move right
            next_state[0] = min(row + 1, self.nrow - 1)
        if action == 2:
            # move down
            next_state[1] = min(col + 1, self.ncol - 1)
        if action == 3:
            # move left
            next_state[0]  = max(row - 1, 0)
        return np.array(next_state)
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        
    def to_s(self, row, col):
            return row*self.ncol + col
        
    def update_reached_destinations(self, newstate):
        if newstate in self.destinations_dict:
            if self.destinations_dict[newstate] == False:
                self.destinations_dict[newstate] = True
                self.num_reached_destinations +=1
                return True
            else:
                return False
            
    def step(self, action):
        # assume we use the max speed as the default speed, when come near to the opt-position, we can slow down the speed

        
        prev_state = self.current_state
        #update pos
        self.current_state = self.transition_dynamics(action, self.current_state)
        
        newstate = self.to_s(self.current_state[0], self.current_state[1])
#         print(self.desc)
        # print(newstate)
        newletter = self.desc[self.current_state[0]][self.current_state[1]]
        
        # s_updated_destinations = self.update_reached_destinations(newstate)
        
        self.s = newstate
        self.lastaction = action
        
        done = bytes(newletter) in b'OD'
        if self.current_step == self.total_steps:
            done =  True
            
        reward = -1
        # reward = 0.05

        is_get_reward = newletter == b'D'

        # if is_get_reward == True and newstate in rewards_dict[self.map_name]:
        if is_get_reward == True:

            reward =  rewards_dict[self.map_name][newstate]
            # print(reward, newstate)

#                 if is_updated_destinations == True:


        if done != True:
            self.current_step += 1
        # else:
        #     reward -= self.current_step

        
        return self.current_state, reward, done
    
    def reset(self):
        self.current_step = 0
        self.current_state = self.initial_state
        self.lastaction = None
        self.num_reached_destinations = 0

        self.destinations_dict = {D: False for D in self.destinations}
        
        return self.current_state
    
    def render(self, mode='human'):
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        row, col = self.s // self.ncol, self.s % self.ncol
        desc = self.desc.tolist()
        desc = [[c.decode('utf-8') for c in line] for line in desc]
        desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)
        if self.lastaction is not None:
            outfile.write("  ({})\n".format(
                ["Left", "Down", "Right", "Up"][self.lastaction]))
        else:
            outfile.write("\n")
        outfile.write("\n".join(''.join(line) for line in desc)+"\n")

        if mode != 'human':
            with closing(outfile):
                return outfile.getvalue()
                    