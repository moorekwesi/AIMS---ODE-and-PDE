import numpy as np
import matplotlib.pyplot as plt

# Parameters
alpha = 0.01  # Thermal diffusivity
dx = dy = 0.1  # Spatial step size
dt = 0.01      # Time step size
nx, ny = 50, 50  # Number of grid points in x and y
nt = 100  # Number of time steps

# Initialize temperature grid
u = np.zeros((nx, ny))

# Initial condition (e.g., a spot in the middle)
u[nx//2, ny//2] = 200

# Finite difference method
for t in range(nt):
    u_new = u.copy()
    for i in range(1, nx-1):
        for j in range(1, ny-1):
            u_new[i, j] = u[i, j] + alpha * dt * (
                (u[i+1, j] - 2*u[i, j] + u[i-1, j]) / dx**2 +
                (u[i, j+1] - 2*u[i, j] + u[i, j-1]) / dy**2
            )
    u = u_new

    # Visualization at certain steps
    if t % 10 == 0:
        plt.figure(figsize=(4,4))
        plt.imshow(u, cmap='jet', interpolation='nearest')
        plt.title(f'Time step {t}')
        plt.colorbar()
        plt.show()