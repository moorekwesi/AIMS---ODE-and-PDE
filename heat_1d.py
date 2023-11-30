import numpy as np
import matplotlib.pyplot as plt

# Parameters
alpha = 0.01  # Thermal diffusivity
dx = 0.1      # Spatial step size
dt = 0.01     # Time step size
nx = 100      # Number of grid points
nt = 200      # Number of time steps

# Initialize temperature grid
u = np.zeros(nx)

# Initial condition (e.g., a spike in the middle)
u[nx//2] = 50

# Finite difference method
for t in range(nt):
    u_new = u.copy()
    for i in range(1, nx-1):
        u_new[i] = u[i] + alpha * dt * (u[i+1] - 2*u[i] + u[i-1]) / dx**2
    u = u_new

    # Visualization at certain steps
    if t % 40 == 0:
        plt.figure(figsize = (3, 3))
        plt.plot(u)
        plt.title(f'Time step {t}')
        plt.ylim(0, 100)
        plt.show()