import numpy as np 
import gymnasium as gym 
import random
from Envs_random import rlEnvs
import matplotlib.pyplot as plt
import matplotlib

env = rlEnvs()

paths =[]
rewards = []
time = []
all_x = []
all_y = []
EPISODES =100
for episode in range(1,EPISODES+1):
    state = env.reset()
    done = False
    score=0
    truncated = False
    while not done and not truncated:
    # actions = [5,3,3,3,3,2,3,3,3,3,3,3,3,3,3,3,3,2,3,3,3,3,3,3,3,3,3,3,2,3,3,3,3,3,3,3,3,3,3,2,3,3,3,3]
    # for i in range(1):
        action =0# np.random.randint(0,8)
        next_state,reward,done,truncated,_ = env.step(action)
        # print(done,truncated)
        score+=reward
    # for _ in range(10):
    #     action = 4#np.random.randint(0,7)
    #     next_state,reward,done,truncated,_ = env.step(action)
    #     # print(done,truncated)
    #     score+=reward
    # for _ in range(6):
    #     action = 3#np.random.randint(0,7)
    #     next_state,reward,done,truncated,_ = env.step(action)
    #     # print(done,truncated)
    #     score+=reward
    # for _ in range(11):
    #     action = 2#np.random.randint(0,7)
    #     next_state,reward,done,truncated,_ = env.step(action)
    #     # print(done,truncated)
    #     score+=reward
    # for _ in range(3):
    #     action = 1#np.random.randint(0,7)
    #     next_state,reward,done,truncated,_ = env.step(action)
    #     # print(done,truncated)
    #     score+=reward
    #     print(reward)
    paths.append(env.render())
    rewards.append(score)
    time.append(env.total_time)
    print(f'Episode: {episode}, Score: {score:.3f} , Time {env.total_time:.3f}') 


##plot each path one one graph \
plt.xlim(-0.75, 0.75)
plt.ylim(-0.75, 0.75)
plt.gca().set_aspect('equal', adjustable='box')
plt.grid()
for path in paths:
    # Unpack the y and x values from the path
    x = [state[1] for state in path]
    y = [state[0] for state in path]
    all_x.extend(x)
    all_y.extend(y)
    # print("PATHS \n\n\n")
    # print(x)
    # print(y)
    plt.plot(x, y, marker='o', markersize=2)
    # circle = plt.Circle((0, 0), 0.5, color='red', fill=False , linestyle='-.')
    # plt.gca().add_patch(circle)
    # for i in range(len(x)):
    #     plt.text(x[i], y[i], f'Point {x[i]:.4f}, {y[i]:.4f}')   


# # Show the plot
plt.show()
# plt.savefig("Figure_k7.png")
# Plot histograms/ of rewards
plt.figure()
plt.hist(rewards, bins=50,label='Rewards',color='blue',edgecolor='black')
plt.xlabel('Rewards')
plt.ylabel('Frequency')
plt.title('Histogram of Rewards')
# plt.show()

plt.figure()
plt.hist(time, bins=50,label='Rewards',color='green',edgecolor='black')
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.title('Histogram of time')
# plt.show()



plt.figure(figsize=(10, 8))
plt.hist2d(all_x, all_y, bins=75, range=[[-0.75, 0.75], [-0.75, 0.75]], cmap='hot_r', norm=matplotlib.colors.LogNorm(vmin=1, vmax=10) )
plt.colorbar(label='Density')
plt.xlabel('X-coordinate')
plt.ylabel('Y-coordinate')
plt.title('2D Histogram Heatmap of Agent\'s Path Density')
plt.grid(True)
plt.show()


heatmap, xedges, yedges = np.histogram2d(all_x, all_y, bins=50, range=[[-0.75, 0.75], [-0.75, 0.75]])

# Convert frequency counts to probability density
prob_density = heatmap / np.sum(heatmap)


# plt.figure(figsize=(10, 8))
# extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
# plt.imshow(prob_density.T, extent=extent, origin='lower', cmap='hot_r', interpolation='nearest')
# plt.colorbar(label='Log-Scaled Probability Density')
# plt.xlabel('X-coordinate')
# plt.ylabel('Y-coordinate')
# plt.title('Log-Scaled Probability Density Heatmap of Agent\'s Path')
# plt.grid(True)
# plt.show()

# plt.figure()
# plt.hist2d(all_x, all_y, bins=50, range=[[-0.75, 0.75], [-0.75, 0.75]],density=True, cmap='hot_r', norm=matplotlib.colors.LogNorm(vmax=1e+2))
# circle = plt.Circle((0, 0), 0.5, color='black', fill=False , linestyle='-.')
# plt.gca().add_patch(circle)
# plt.xlim(-0.75, 0.75)
# plt.ylim(-0.75, 0.75)
# plt.gca().set_aspect('equal', adjustable='box')
# plt.colorbar(label='Probability Density')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.grid()
# plt.show()

