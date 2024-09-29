from gymnasium import Env
from gymnasium.spaces import Discrete, Box, Dict
import numpy as np
import random
from math import sqrt, cos, sin
from config import DEFAULT_CONFIG

class rlEnvs(Env):
    def __init__(self, conf: dict = None):
        if conf is None:
            self.conf = DEFAULT_CONFIG
            print("Using default config")
        else:
            self.conf = conf
            print("Using custom config")
            print(self.conf["delt"])

        self.d = self.conf['d']
        self.visibility = self.conf['visiblitity']
        self.truncated = False 
        self.done = False
        self.action_space = Discrete(8)
        vector_space = Box(low=-float('inf'), high=float('inf'), shape=(5,), dtype=np.float64)
        self.observation_space = Dict({
            "vector": vector_space
        })

        self.velocity = 1
        self.noise = self.conf['noise']
        self.reward = 0 
        self.delt = self.conf['delt']
        self.total_time = 0
        self.del_r = self.conf['delta_r']
        self.function = self.conf['function']
        self.start_x, self.start_y = self.conf['start'][0], self.conf['start'][1]
        self.target_x, self.target_y = self.conf['end'][0], self.conf['end'][1]
        print("Starting position: ", self.start_x, self.start_y)
        print("Target position: ", self.target_x, self.target_y)
        self.trajectory = []
        self.reward_trajectory = []
        self.random_start = self.conf['random_start']
        self.state = [self.start_y, self.start_x]
        self.theta = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.random_start:
            self.state = [random.uniform(self.conf["start"][0], self.conf["end"][0]),
                          random.uniform(self.conf["start"][1], self.conf["end"][1])]
        else:
            self.state = [self.start_y, self.start_x]

        self.done = False
        self.reward = 0
        self.total_time = 0
        self.velocity = 1
        self.truncated = False
        self.trajectory = []
        self.reward_trajectory = []
        self.trajectory.append(self.state)
        # action = np.random.randint(0, 8)
        self.theta = 0 #(action - 0) * np.pi / 4

        return np.array(self.state, dtype=np.float32)  # Ensure dtype matches observation space

    def step(self, action: int):
        y_old, x_old = self.state
        self.reward = 0

        # Check if the agent is within the target range
        if sqrt((self.target_x - x_old) ** 2 + (self.target_y - y_old) ** 2) < self.del_r:
            self.done = True
            self.reward = 100
        elif self.velocity < 0 or self.total_time >= 10 or (x_old - self.target_x) > 0.001:
            self.reward = -50
            self.truncated = True
        
        # Compute the new angle based on the action
        theta = 1*(action - 0) * np.pi / 4
        if self.noise:
            theta += sqrt(2 * self.d * self.delt) * np.random.normal(0, 1)
            # print(np.degrees(theta))
        self.theta += theta
        
        # Update new positions based on the angle and velocity
        x_new = x_old + self.velocity * np.cos(self.theta) * self.delt
        y_new = y_old + self.velocity * np.sin(self.theta) * self.delt

        # Update state and total time
        self.state = [y_new, x_new]
        self.total_time += self.delt
        
        # Reward based on the step
        self.reward += -self.delt
        
        # Record trajectory and rewards
        self.reward_trajectory.append(self.reward)
        self.trajectory.append(self.state)
        
        # Construct observation
        vec = [x_new, y_new, self.target_x, self.target_y, int(self.velocity)]
        obs = {
            "vector": np.array(vec, dtype=np.float32)
        }

        return self.state, self.reward, self.done, self.truncated, {}

    def render(self, mode='human'):
        return self.trajectory
