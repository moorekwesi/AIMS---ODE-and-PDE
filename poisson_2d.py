import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def poisson_2d(f, ax, bx, ay, by, Nx, Ny):
    """
    Solves the 2D Poisson problem:
    u_xx(x,y) + u_yy(x,y) = f(x, y) for (x,y) in [ax,bx]x[ay,by]
    with Dirichlet boundary conditions u = 0 on the boundary.

    f: The source term function f(x, y)
    ax, bx: The x-boundaries of the domain
    ay, by: The y-boundaries of the domain
    Nx, Ny: Number of interior grid points in x and y directions
    """

    # Define the grid
    x = np.linspace(ax, bx, Nx+2)
    y = np.linspace(ay, by, Ny+2)
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    # Initialize the solution array
    u = np.zeros((Nx+2, Ny+2))

    # Construct the coefficient matrix
    A = np.zeros((Nx*Ny, Nx*Ny))
    for j in range(Ny):
        for i in range(Nx):
            row = j * Nx + i
            A[row, row] = -2/dx**2 - 2/dy**2
            if i > 0:
                A[row, row - 1] = 1/dx**2
            if i < Nx - 1:
                A[row, row + 1] = 1/dx**2
            if j > 0:
                A[row, row - Nx] = 1/dy**2
            if j < Ny - 1:
                A[row, row + Nx] = 1/dy**2

    # Construct the right-hand side
    b = np.zeros(Nx*Ny)
    for j in range(Ny):
        for i in range(Nx):
            b[j * Nx + i] = f(x[i+1], y[j+1])

    # Solve the linear system
    u_flat = np.linalg.solve(A, b)
    u[1:-1, 1:-1] = u_flat.reshape((Ny, Nx))

    return x, y, u

# Example usage
def f(x, y):
    return np.sin(np.pi * x) * np.sin(np.pi * y)

ax, bx, ay, by = 0, 1, 0, 1  # Domain boundaries
Nx, Ny = 50, 50  # Number of grid points

x, y, u = poisson_2d(f, ax, bx, ay, by, Nx, Ny)

# Plotting the solution
X, Y = np.meshgrid(x, y)
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, u, cmap='viridis')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('U')
plt.title('Solution of the 2D Poisson Problem')
plt.show()
