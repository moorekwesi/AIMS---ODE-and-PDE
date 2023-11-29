import numpy as np
import matplotlib.pyplot as plt

def poisson_1d(f, a, b, alpha, beta, N):
    """
    Solves the 1D Poisson problem u''(x) = f(x) for x in (a, b)
    with boundary conditions u(a) = alpha, u(b) = beta.

    f: The source term function f(x)
    a, b: The boundaries of the domain
    alpha, beta: The boundary conditions
    N: Number of interior grid points
    """

    # Define the grid
    x = np.linspace(a, b, N+2)
    dx = x[1] - x[0]

    # Initialize the solution array
    u = np.zeros(N+2)
    u[0], u[-1] = alpha, beta

    # Construct the coefficient matrix
    A = np.diag([-2] * N) + np.diag([1] * (N-1), k=1) + np.diag([1] * (N-1), k=-1)

    # Scale by dx^2
    A /= dx**2

    # Construct the right-hand side
    b = f(x[1:-1])
    b[0] -= alpha / dx**2
    b[-1] -= beta / dx**2

    # Solve the linear system
    u[1:-1] = np.linalg.solve(A, b)

    return x, u

# Example usage
a = 0
b = 1
alpha = 0
beta = 0
N = 100

f = lambda x: np.sin(np.pi * x)  # Example source term

x, u = poisson_1d(f, a, b, alpha, beta, N)

# Plotting the solution
plt.plot(x, u)
plt.xlabel("x")
plt.ylabel("u(x)")
plt.title("Solution of the 1D Poisson Problem")
plt.show()
