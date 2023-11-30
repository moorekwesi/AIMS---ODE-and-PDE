import numpy as np
import matplotlib.pyplot as plt

# Parameters
L = 10          # Length of the domain
Nx = 100         # Number of spatial points
dx = L / (N-1)  # Spatial step size
u = 10.0         # Speed of the wave
dt = 0.01       # Time step size
T = 2           # Total time

# Grids
x = np.linspace(0, L, N)
Nt = int(T / dt)

# Initial condition (e.g., a Gaussian pulse)
initial_condition = np.exp(-((x - L/4)**2) / 0.1)

# Solution array
solution = np.zeros((Nt, Nx))
solution[0, :] = initial_condition

# Time-stepping loop (using upwind scheme for stability)
for i in range(0, Nt-1):
    for j in range(1, Nx):
        solution[i + 1, j] = solution[i, j] - u * dt / dx * (solution[i, j] - solution[i, j-1])

# Plotting the solution
plt.figure(figsize=(10, 6))
for n in range(0, timesteps, 10):
    plt.plot(x, solution[n, :], label=f't={n*dt:.2f}')
plt.title('Numerical solution of the 1D transport equation')
plt.xlabel('x')
plt.ylabel('u')
plt.legend()
plt.show()