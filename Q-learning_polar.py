import numpy as np 
import matplotlib.pyplot as plt
from config import DEFAULT_CONFIG
from Envs_polar import rlEnvs
from utils import *
import pickle 
import time
import sys
import os
from tqdm import tqdm
from colorama import Fore , Back

def save_q_table(Q, filename):
        with open(filename, 'wb') as f:
            pickle.dump(Q, f)


def run(EPISODES, verbose, epsilon_value, print_val,q, env, filename):
    
    learning_rate=0.9
    discount_factor=0.9
    epsilon= epsilon_value ## 100% random actions
    epsilon_decay_rate=0.0000015
    rng = np.random.default_rng()
    reward_per_episodes=np.zeros(EPISODES)
    print(q)
    for i in tqdm(range(EPISODES)):
        state,_ = env.reset()
        done= False
        truncated= False
        while (not done and  not truncated):
            # print("*********************",state[0])
            if rng.random() <epsilon:
                action= env.action_space.sample()
            else:
                max_actions = np.where(q[int(state[0]), :] == np.max(q[int(state[0]), :]))[0]
                action = rng.choice(max_actions)
            new_state,reward,done,truncated,info= env.step(action)
            q[int(state[0]),action] = q[int(state[0]),action] + learning_rate * (reward + discount_factor * np.max(q[int(new_state[0]),:])-q[int(state[0]),action])
            if verbose == True:
                print(f"new state {divmod(int(state[0]),env.W)} DONE {done}  TRUNCATEDDD {truncated}  Rewards {reward}")
            state = new_state
            reward_per_episodes[i]+=reward
            # print(q)
            # break
        epsilon= max(epsilon-epsilon_decay_rate,0.15)
        if (done):
            print_grid_and_path(env.grid,env.state_trajectory ,conf=None,save_path='Q-learning /', plotting=False, graph_title= None)
        
        if (i %print_val==0):
            print(i, epsilon)
        ##------idhar bhi save wali cheez daalni hai------##
            save_q_table(q, 'q_table.pkl')  
            # print(q)   
        #----yahan tak----#
            print(f"Truncated {truncated} DONE {done}")
            if (done):
                print_grid_and_path(env.grid,env.state_trajectory ,conf=None,save_path='Q-learning/donea/', plotting=False, graph_title= None)

    save_q_table(q, 'q_table.pkl')            
    


      

def main():
    print("Q-learning with polar coordinates")
    env = rlEnvs()
    q_table_filename= "IDK"
    filename= "IDK"
    if len(sys.argv) >1 :
        q_table_filename = sys.argv[1]
        # q = np.load(q_table_filename)
        if os.path.exists(q_table_filename):
            print("present",q_table_filename)
            with open(q_table_filename, 'rb') as f:
                q = pickle.load(f)
                print("opened")
                print(q)
        else:
            q = np.zeros((env.W*env.H,env.action_space.n))
            with open(q_table_filename, 'wb') as f:
                pickle.dump(q, f)
    else:
        q = np.zeros((env.W*env.H,env.action_space.n))

    run(800000,False,0.05,1000,q,env,filename)
    



if __name__ == "__main__":
    main()


