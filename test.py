import numpy as np
import matplotlib.pyplot as plt

# Parameters
n_particles = 1000  # number of particles
dt = 0.01  # time step
n_steps = int(10 / dt)
v = 1.0  # constant velocity
rotational_diffusion = 0.1  # strength of rotational noise (D_r)

# Initialize positions and orientations
positions = np.zeros((n_particles, n_steps, 2))
orientations = np.zeros((n_particles, n_steps))

# Random initial orientations (in radians)
orientations[:, 0] = np.random.uniform(0, 2 * np.pi, n_particles)

# Run simulation
for t in range(1, n_steps):
    # Update orientations due to rotational noise (Brownian motion in angle)
    dtheta = np.sqrt(2 * rotational_diffusion * dt) * np.random.randn(n_particles)
    orientations[:, t] = orientations[:, t - 1] + dtheta
    
    # Compute velocity components from current orientations
    dx = v * np.cos(orientations[:, t]) * dt
    dy = v * np.sin(orientations[:, t]) * dt
    
    # Update positions based on velocities
    positions[:, t, 0] = positions[:, t - 1, 0] + dx  # x position
    positions[:, t, 1] = positions[:, t - 1, 1] + dy  # y position

# Calculate Mean Squared Displacement (MSD)
msd = np.zeros(n_steps)
for t in range(1, n_steps):
    displacement = positions[:, t, :] - positions[:, 0, :]
    msd[t] = np.mean(np.sum(displacement**2, axis=1))

# Plot individual particle paths
plt.figure(figsize=(14, 6))

# Plotting the paths of a subset of particles
for i in range(min(n_particles, 100)):  # Limiting to the first 10 particles for clarity
    plt.plot(positions[i, :, 0], positions[i, :, 1], alpha=0.5)

plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('Paths of Individual Particles')
plt.grid(True)
plt.gca().set_aspect('equal')
plt.show()

# Plot the MSD
plt.figure(figsize=(8, 6))
plt.plot(np.arange(n_steps) * dt, msd, label="MSD (rotational noise only)")
plt.xlabel('Time')
plt.ylabel('Mean Squared Displacement')
plt.title('Mean Squared Displacement for ABM with Rotational Noise Only')
plt.legend()
plt.grid(True)
plt.show()
