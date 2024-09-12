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

def mapping(radius, theta, r, t):
    if (radius ==0 ):
        return 0
    radius_index = np.abs(radius - r).argmin()
    theta_index = np.abs(theta - t).argmin()
    return (radius_index-1)*33 + theta_index  

def save_q_table(Q, filename):
    with open(filename, 'wb') as f:
        pickle.dump(Q, f)

def run(EPISODES,verbose,epsilon,print_val,q,env,filename,r,t):
    learning_rate=0.9
    discount_factor=1
    epsilon_decay_rate=0.00015
    rng = np.random.default_rng()
    reward_per_episodes=np.zeros(EPISODES)
    env= rlEnvs()
    for i in tqdm(range(EPISODES)):
        state, _ = env.reset()
        done= False
        truncated= False
        while (not done and  not truncated):
            state = state["vector"]
            state_radius = state[0]
            state_theta = state[1]
            state_index = mapping(state_radius, state_theta, r, t)
            if rng.random() <epsilon:
                action= env.action_space.sample()
            else:
                max_actions = np.where(q[int(state_index), :] == np.max(q[int(state_index), :]))[0]
                action = rng.choice(max_actions)
            new_state,reward,done,truncated,info= env.step(action)
            # q[int(state[0]),action] = q[int(state[0]),action] + learning_rate * (reward + discount_factor * np.max(q[int(new_state[0]),:])-q[int(state[0]),action])
            if verbose == True:
                print(f"New State {state_radius} {state_theta} {state_index} DONE {done}  TRUNCATEDDD {truncated}  Rewards {reward}")
            state = new_state
            reward_per_episodes[i]+=reward
        epsilon= max(epsilon-epsilon_decay_rate,0.05)
        save_q_table(q, "IDK.pkl")

def main():
    print("Q-learning with polar coordinates")
    env = rlEnvs()
    q_table_filename= "IDK"
    filename= "IDK"
    
    conf = DEFAULT_CONFIG
    r_len =int((abs(conf['start'][0])+abs(conf['end'][0]))/(2*conf["delr"]))
    
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
            q = np.zeros(((r_len+1)*34,env.action_space.n))
            with open(q_table_filename, 'wb') as f:
                pickle.dump(q, f)
    else:
        q = np.zeros(((r_len+1)*34,env.action_space.n))
    radius = np.arange(0, (abs(conf['start'][0])+abs(conf['end'][0]))/2+conf["delr"], conf["delr"])
    theta = np.arange(0, 2*np.pi,conf["deltheta"])
        
    print(f"Q_table Shape ",q.shape)

    run(2,False,0.95,1000,q,env,filename,radius,theta)
    



if __name__ == "__main__":
    main()


