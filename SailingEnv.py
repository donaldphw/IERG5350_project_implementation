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
        "SWWOOWWD",
        "WWWOOWWW",
        "WWWWWWWW",
        "WWWWWWWW",
        "OOWWWWWW",
        "WWWWWWWW",
        "WWWWWWOO",
        "DWWWWWWD"
    ],
    "16x16": [
        "SWWWWWWWWWWOOWWD",
        "WWWWWWWWWWWOOWWW",
        "WWWWWWWWWWWWWWWW",
        "WWWWOOOWWWWWWWWW",
        "WWWWOOOWWWWWWWWW",
        "WWWWWWWWWWWWWWWW",
        "OOOWWWWWWWWWWOOO",
        "OOOWWWWWWWWWWOOO",
        "WWWWWWWWWWWWWWWW",
        "WWWWWWOOOWWWWWWW",
        "WWWWWWOOOWWWWWWW",
        "WWWWWWWWWWWWWWWW",
        "OOOWWWWWWWWWWWWW",
        "OOOWWWWWWWWWWWWW",
        "WWWWWWWWWWOOOOOO",
        "DWWWWWWWWWWWWWWD"
    ],
}


rewards_dict = {
    "8x8":
    {
        7 : 2.0,
        56 : 4.0,
        63 : 10.0
    },
    "16x16":
    {
        15: 400.0,
        240: 800.0,
        255 : 2000.0
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


class SailingEnv():

    metadata = {'render.modes': ['human', 'ansi']}

#     def __init__(self, desc=None, map_name="8x8", is_slippery=True):
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
        
        self.nA = 4
        self.nS = nrow * ncol
        self.action_space = spaces.Discrete(self.nA)
        self.observation_space = spaces.Discrete(self.nS)
        self.seed()
        self.isd = np.array(desc == b'S').astype('float64').ravel()
        self.isd /= self.isd.sum()

        self.P = {s: {a: [] for a in range(self.nA)} for s in range(self.nS)}
        
        
        def to_s(row, col):
            return row*ncol + col

        def inc(row, col, a):
            if a == LEFT:
                col = max(col - 1, 0)
            elif a == DOWN:
                row = min(row + 1, nrow - 1)
            elif a == RIGHT:
                col = min(col + 1, ncol - 1)
            elif a == UP:
                row = max(row - 1, 0)
            return (row, col)
        
        
        def seed(self, seed=None):
            self.np_random, seed = seeding.np_random(seed)
            return [seed]
        

        

        
        def update_reached_destinations(newstate):
            if newstate in self.destinations_dict:
                if self.destinations_dict[newstate] == False:
                    self.destinations_dict[newstate] = True
                    self.num_reached_destinations +=1
                    return True
                else:
                    return False
            
        def get_reward():
            for key, value in self.destinations_dict.items():
                if value == False:
                    return float(0.0)
            return float(1.0)
            

        def update_probability_matrix(row, col, action):
            newrow, newcol = inc(row, col, action)
            
            newstate = to_s(newrow, newcol)
            newletter = desc[newrow, newcol]
            is_updated_destinations = update_reached_destinations(newstate)
            
                
            done = bytes(newletter) in b'OD'
#             done = self.current_step == self.total_steps

        

            # if is_updated_destinations == True:
            #     done =  self.num_reached_destinations == self.total_destinations
            
            reward = - 1
            is_get_reward = float(newletter == b'D')
            if is_get_reward == True and newstate in rewards_dict[self.map_name]:
#                 if is_updated_destinations == True:
                reward+= rewards_dict[self.map_name][newstate]
#             reward = float(newletter == b'D')
#             reward = get_reward()
#             reward = float(self.num_reached_destinations == self.total_destinations)
            return newstate, reward, done
        
        for row in range(nrow):
            for col in range(ncol):
                s = to_s(row, col)
                for a in range(4):
                    li = self.P[s][a]
                    letter = desc[row, col]
                    if letter in b'OD':
                        li.append((1.0, s, 0, True))
                    else:
                        if is_slippery:
                            for b in [(a - 1) % 4, a, (a + 1) % 4]:
                                li.append((
                                    1. / 3.,
                                    *update_probability_matrix(row, col, b)
                                ))
                        else:
                            li.append((
                                1., *update_probability_matrix(row, col, a)
                            ))

    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    def update_reached_destination(self, newstate):
        if newstate in self.destinations_dict:
            if self.destinations_dict[newstate] == False:
                self.destinations_dict[newstate] = True
                self.num_reached_destinations +=1
                return True
            else:
                return False
            
    def step(self, a):
        transitions = self.P[self.s][a]
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        p, s, r, d = transitions[i]
        
#         is_updated_destinations = self.update_reached_destination(s)
#         r = float(self.num_reached_destinations == self.total_destinations)
        self.s = s
        self.lastaction = a
        
        
#         if is_updated_destinations == True:
#             d =  self.num_reached_destinations == self.total_destinations

        if self.current_step == self.total_steps:
            d =  True
        if d != True:
            self.current_step = self.current_step + 1
        # else:
        #     r -= self.current_step

            
        return (int(s), r, d, {"prob": p})
    
    def reset(self):
        self.current_step = 0
        self.s = categorical_sample(self.isd, self.np_random)
        self.lastaction = None
        self.num_reached_destinations = 0

        self.destinations_dict = {D: False for D in self.destinations}
        return int(self.s)
    
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