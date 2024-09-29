import numpy as np
import gymnasium as gym
import random
from check_bc import rlEnvs
import matplotlib.pyplot as plt
import matplotlib
import time as TIME
_CONFIG = {
    'd': 0.50,
    'visiblitity': 18,
    'space': 0.01,
    'noise': True,
    'delt': 0.001,
    'start': [0, 0],
    'end': [2e5, 2e5],
    'random_start': False,
    'delta_r': 0.05,
    'function': 1
}
TIME_CONSTANT = 1
delta_t = [1.0, 0.1, 0.01 , 0.001 , 0.0001] 
colors = ['blue', 'green', 'orange', 'purple', 'pink', 'brown']

_CONFIG['delt'] = delta_t[TIME_CONSTANT]
env = rlEnvs(_CONFIG)

paths = []
rewards = []
time = []
all_x = []
all_y = []
EPISODES = 100
for episode in range(1, EPISODES + 1):
    state = env.reset()
    done = False
    score = 0
    truncated = False
    while not done and not truncated:
        action = np.random.randint(0, 800)
        next_state, reward, done, truncated, _ = env.step(action)
        score += reward

    paths.append(env.render())
    rewards.append(score)
    time.append(env.total_time)
    print(f'Episode: {episode}, Score: {score:.3f}, Time: {env.total_time:.3f}')

num_steps = len(paths[0])
x_positions = np.array([[state[0] for state in path] for path in paths])
y_positions = np.array([[state[1] for state in path] for path in paths])

msd = np.mean((x_positions - x_positions[:, [0]])**2 + (y_positions - y_positions[:, [0]])**2, axis=0)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))  # Two subplots side by side

for path in paths:
    x = [state[1] for state in path]
    y = [state[0] for state in path]
    ax1.scatter(x[0], y[0], color='green', s=20) 
    ax1.scatter(x[-1], y[-1], color='red', s=20)  
    ax1.plot(x, y,alpha=0.2)
ax1.set_aspect('equal', adjustable='box')
ax1.grid()
ax1.set_title('Particle Paths')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')


ax1.set_xlim([-10, 10]) 
ax1.set_ylim([-10, 10])  

time_steps = np.arange(num_steps, dtype=float) * _CONFIG['delt']
slope, intercept = np.polyfit(time_steps, msd, 1)

# ax2.plot(time_steps, slope * time_steps+intercept , color='red', linestyle='--')
ax2.text(0.1, 0.9, f'Slope: {slope:.2f}', transform=ax2.transAxes, color='red')
ax2.text(0.1, 0.86, f'Theoritical Slope: {4*_CONFIG["d"]:.2f}', transform=ax2.transAxes, color='red')


ax2.plot(time_steps, msd, color='blue')
ax2.set_title('Mean Square Displacement (MSD)')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('MSD')
ax2.grid()

plt.tight_layout()

# plt.savefig(f'MSD{delta_t[TIME_CONSTANT]}.png')
plt.show()