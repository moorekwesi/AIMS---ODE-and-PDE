import numpy as np
import matplotlib.pyplot as plt

def solve_1d_heat_equation(num_x_points, num_t_points, alpha, x_max, t_max, u_initial, u_boundary):
    """
    Solves the 1D heat equation with the given parameters and initial/boundary conditions.

    Parameters:
    num_x_points (int): Number of spatial points.
    num_t_points (int): Number of time points.
    alpha (float): Thermal diffusivity.
    x_max (float): Maximum spatial extent.
    t_max (float): Maximum time extent.
    u_initial (function): Initial temperature distribution as a function of x.
    u_boundary (float): Boundary temperature (constant).

    Returns:
    numpy.ndarray: Temperature distribution over time and space.
    """

    # Define the spatial and temporal grid
    dx = x_max / (num_x_points - 1)
    dt = t_max / (num_t_points - 1)
    x = np.linspace(0, x_max, num_x_points)
    t = np.linspace(0, t_max, num_t_points)

    # Stability criterion for the explicit scheme
    if alpha * dt / dx**2 >= 0.5:
        raise ValueError("The scheme is unstable. Choose different dt, dx, or alpha.")

    # Initialize the temperature distribution
    u = np.zeros((num_t_points, num_x_points))
    u[0, :] = u_initial(x)
    u[:, 0] = u_boundary
    u[:, -1] = u_boundary

    # Apply the explicit scheme
    for j in range(0, num_t_points - 1):
        for i in range(1, num_x_points - 1):
            u[i , j+1] = u[i, j] + alpha * dt / dx**2 * (u[i+1, j] - 2 * u[i, j] + u[i-1, i])

    return u, x, t

# Example parameters
num_x_points = 50
num_t_points = 2000
alpha = 0.01  # Thermal diffusivity
x_max = 10
t_max = 2
u_initial = lambda x: np.sin(np.pi * x / x_max)  # Initial condition: sin wave
u_boundary = 0  # Boundary condition: temperature is 0 at both ends

# Solve the 1D heat equation
u, x, t = solve_1d_heat_equation(num_x_points, num_t_points, alpha, x_max, t_max, u_initial, u_boundary)

# Plotting
plt.imshow(u, extent=[0, t_max, 0, x_max], origin='lower',
           aspect='auto', cmap='hot')
plt.colorbar(label='Temperature')
plt.xlabel('Time')
plt.ylabel('Position')
plt.title('1D Heat Equation')
plt.show()
