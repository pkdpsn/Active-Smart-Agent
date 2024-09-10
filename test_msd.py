import numpy as np
import matplotlib.pyplot as plt

# Parameters
num_steps = 100 # Number of time steps
num_particles = 500  # Number of particles
step_size = 1.0  # Step size for the random walk

# Initialize positions
x_positions = np.zeros((num_particles, num_steps))
y_positions = np.zeros((num_particles, num_steps))

# Simulate Brownian motion for each particle
for i in range(1, num_steps):
    # Random displacements (normal distribution)
    dx = np.random.normal(0, step_size, num_particles)
    dy = np.random.normal(0, step_size, num_particles)

    # Update positions
    x_positions[:, i] = x_positions[:, i-1] + dx
    y_positions[:, i] = y_positions[:, i-1] + dy

# Calculate MSD
msd = np.mean((x_positions - x_positions[:, [0]])**2 + (y_positions - y_positions[:, [0]])**2, axis=0)
print(msd)
# Plot MSD vs time step
time_steps = np.arange(num_steps)
plt.plot(time_steps, msd)
plt.xlabel('Time Step')
plt.ylabel('Mean Square Displacement (MSD)')
plt.title('MSD of 2D Brownian Motion')
plt.show()
