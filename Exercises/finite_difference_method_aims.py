# -*- coding: utf-8 -*-
"""Finite_Difference_Method_AIMS

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1SGdH9NZPaUGWuUWiwsdBV53MLxWbJm3T

This is prepared for studying the numerics of ODEs and PDEs.
The hands-on session is prepared by Pr. Stephen Moore | moorestephen.info
"""

# Commented out IPython magic to ensure Python compatibility.
# Exercise to

# Import two essential libraries, NumPy and Matplotlib,
import numpy as np
import matplotlib.pyplot as plt
#plt.style.use('seaborn-poster')
# %matplotlib inline

# step size
h = 0.1
# define grid
x = np.arange(0, 2*np.pi, h)
# compute function
y = np.cos(x)

# compute vector of forward differences
forward_diff = np.diff(y)/h
# compute corresponding grid
x_diff = x[:-1:]
# compute exact solution
exact_solution = -np.sin(x_diff)

# Plot solution
plt.figure(figsize = (8, 8))
plt.plot(x_diff, forward_diff, '--', \
         label = 'Finite difference approximation')
plt.plot(x_diff, exact_solution, \
         label = 'Exact solution')
plt.legend()
plt.show()

# Compute max error between
# numerical derivative and exact solution
max_error = max(abs(exact_solution - forward_diff))
print(max_error)

"""**Note:** The following code computes the numerical derivative of $f(x)=\cos(x)$
 using the forward difference formula for decreasing step sizes, $h.$ It then plots the maximum error between the approximated derivative and the true derivative versus $h.$
 The slope of the line in log-log space is 1; therefore, the error is proportional to $h^1$
, which means that, as expected, the forward difference formula is $\mathcal{O}(h).$

"""

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-poster')
# %matplotlib inline

# define step size
h = 1
# define number of iterations to perform
iterations = 20
# list to store our step sizes
step_size = []
# list to store max error for each step size
max_error = []

for i in range(iterations):
    # halve the step size
    h /= 2
    # store this step size
    step_size.append(h)
    # compute new grid
    x = np.arange(0, 2 * np.pi, h)
    # compute function value at grid
    y = np.cos(x)
    # compute vector of forward differences
    forward_diff = np.diff(y)/h
    # compute corresponding grid
    x_diff = x[:-1]
    # compute exact solution
    exact_solution = -np.sin(x_diff)

    # Compute max error between
    # numerical derivative and exact solution
    max_error.append(\
            max(abs(exact_solution - forward_diff)))

# produce log-log plot of max error versus step size
plt.figure(figsize = (8, 6))
plt.loglog(step_size, max_error, 'v')
plt.show()

"""**Poisson problem 1D**"""

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
    #A = np.diag([-2] * N) + np.diag([1] * (N-1), k=1) + np.diag([1] * (N-1), k=-1)
    A = np.diag([-2.0] * N) + np.diag([1.0] * (N-1), k=1) + np.diag([1.0] * (N-1), k=-1)


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

"""**Poisson Problem 2D**"""

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

"""**Burger's Equation**"""

# Commented out IPython magic to ensure Python compatibility.
# vector manipulation
import numpy as np
# math functions
import math

# THIS IS FOR PLOTTING
# %matplotlib inline
import matplotlib.pyplot as plt # side-stepping mpl backend
import warnings
warnings.filterwarnings("ignore")

# Discete Grid
N=10 # you can modify here for spatial grid
Nt=10
h=2*np.pi/N
k=1/Nt
r=k/(h*h)
time_steps=10
time=np.arange(0,(time_steps+.5)*k,k)
x=np.arange(0,2*np.pi+h/2,h)

# Create mesh grid in 2D
X, Y = np.meshgrid(x, time)

fig = plt.figure(figsize=(6,3))
plt.subplot(121)
plt.plot(X,Y,'ro');
plt.plot(x,0*x,'bo',label='Initial Condition');
plt.xlim((-h,2*np.pi+h))
plt.ylim((-k,max(time)+k))
plt.xlabel('x')
plt.ylabel('time (ms)')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.title(r'Discrete Mesh $\Omega_h$ ',fontsize=24,y=1.08)
plt.show();

# Initial Conditions
w=np.zeros((time_steps+1,N+1))
b=np.zeros(N-1)
# Initial Condition
for j in range (0,N+1):
    w[0,j]=1-np.cos(x[j])


fig = plt.figure(figsize=(6,3))
plt.subplot(122)
plt.plot(x,w[0,:],'o:',label='Initial Condition')
plt.xlim([-0.1,max(x)+h])
plt.title('Initial Condition',fontsize=24)
plt.xlabel('x')
plt.ylabel('w')
plt.legend(loc='best')
plt.show()
ipos = np.zeros(N+1)
ineg = np.zeros(N+1)
for i in range(0,N+1):
   ipos[i] = i+1
   ineg[i] = i-1

ipos[N] = 0
ineg[0] = N

"""**Lax-Friedrichs Method**"""

lamba=k/h
for j in range(0,time_steps):
    for i in range (0,N+1):
        w[j+1,i]=(w[j,int(ipos[i])]+w[j,int(ineg[i])])/2+lamba*w[j,i]/2*(-(w[j,int(ipos[i])]-w[j,int(ineg[i])]))

fig = plt.figure(figsize=(12,6))

plt.subplot(121)
for j in range (1,time_steps+1):
    plt.plot(x,w[j,:],'o:')
plt.xlabel('x')
plt.ylabel('w')

plt.subplot(122)
X, T = np.meshgrid(x, time)
z_min, z_max = np.abs(w).min(), np.abs(w).max()


plt.pcolormesh( X,T, w, vmin=z_min, vmax=z_max)


#plt.imshow(w, aspect='auto')
plt.xlabel('x')
plt.ylabel('time')
clb=plt.colorbar()
clb.set_label('Temperature (w)')
plt.suptitle('Numerical Solution of the  Burger Equation'%(np.round(r,3)),fontsize=24,y=1.08)
fig.tight_layout()
plt.show()

"""***Lax-Wendroff***"""

lamba = k / h  # lambda = dt / dx

for j in range(0, time_steps - 1):
    for i in range(1, N):  # Assuming N+1 points, iterate from 1 to N-1
        w[j+1, i] = w[j, i] - 0.5 * lamba * (w[j, i+1] - w[j, i-1]) + 0.5 * lamba**2 * (w[j, i+1] - 2*w[j, i] + w[j, i-1])

fig = plt.figure(figsize=(12,6))

plt.subplot(121)
for j in range (1,time_steps+1):
    plt.plot(x,w[j,:],'o:')
plt.xlabel('x')
plt.ylabel('w')

plt.subplot(122)
X, T = np.meshgrid(x, time)
z_min, z_max = np.abs(w).min(), np.abs(w).max()


plt.pcolormesh( X,T, w, vmin=z_min, vmax=z_max)


#plt.imshow(w, aspect='auto')
plt.xlabel('x')
plt.ylabel('time')
clb=plt.colorbar()
clb.set_label('Temperature (w)')
plt.suptitle('Numerical Solution of the  Burger Equation using Lax-Wendroff'%(np.round(r,3)),fontsize=24,y=1.08)
fig.tight_layout()
plt.show()

"""**Transport Equation**"""

import numpy as np
import matplotlib.pyplot as plt

# Parameters
L = 10          # Length of the domain
Nx = 100         # Number of spatial points
dx = L / (Nx-1)  # Spatial step size
u = 1.0         # Speed of the wave
dt = 0.01       # Time step size
T = 2           # Total time

# Grids
x = np.linspace(0, L, Nx)
timesteps = int(T / dt)

# Initial condition (e.g., a Gaussian pulse)
initial_condition = np.exp(-((x - L/4)**2) / 0.1)

# Solution array
u_new = np.zeros((timesteps, Nx))
u_new[0, :] = initial_condition

# Time-stepping loop (using upwind scheme for stability)
for i in range(0, timesteps-1):
    for j in range(1, Nx):
        u_new[i + 1, j] = u_new[i, j] - u * dt / dx * (u_new[i, j] - u_new[i, j-1])

# Plotting the solution
plt.figure(figsize=(10, 6))
for i in range(0, timesteps, 10):
    plt.plot(x, u_new[i, :], label=f't={i*dt:.2f}')
plt.title('Numerical solution of the 1D transport equation')
plt.xlabel('x')
plt.ylabel('u')
plt.legend()
plt.show()

"""***1D Wave Equation***"""

def solve_1d_wave_equation(num_x_points, num_t_points, c, x_max, t_max, u_initial, v_initial):
    """
    Solves the 1D wave equation with the given parameters and initial conditions.

    Parameters:
    num_x_points (int): Number of spatial points.
    num_t_points (int): Number of time points.
    c (float): Speed of the wave.
    x_max (float): Maximum spatial extent.
    t_max (float): Maximum time extent.
    u_initial (function): Initial displacement as a function of x.
    v_initial (function): Initial velocity as a function of x.

    Returns:
    numpy.ndarray: Displacement over time and space.
    """

    # Define the spatial and temporal grid
    dx = x_max / (num_x_points - 1)
    dt = t_max / (num_t_points - 1)
    x = np.linspace(0, x_max, num_x_points)

    # Stability criterion
    if c * dt / dx > 1:
        raise ValueError("The scheme is unstable. Choose different dt, dx, or c.")

    # Initialize the displacement array
    u = np.zeros((num_t_points, num_x_points))
    u[0, :] = u_initial(x)

    # Initial velocity handling for the first time step
    for j in range(1, num_x_points - 1):
        u[1, j] = u[0, j] + v_initial(x[j]) * dt + c**2 * dt**2 / (2 * dx**2) * (u[0, j + 1] - 2 * u[0, j] + u[0, j - 1])

    # Apply the Central Difference Scheme
    for n in range(1, num_t_points - 1):
        for j in range(1, num_x_points - 1):
            u[n + 1, j] = 2 * u[n, j] - u[n - 1, j] + c**2 * dt**2 / dx**2 * (u[n, j + 1] - 2 * u[n, j] + u[n, j - 1])

    return u, x

# Example parameters
num_x_points = 100
num_t_points = 1000
c = 1  # Speed of the wave
x_max = 10
t_max = 10
u_initial = lambda x: np.sin(np.pi * x / x_max)  # Initial displacement: sin wave
v_initial = lambda x: np.zeros_like(x)  # Initial velocity: zero

# Solve the 1D wave equation
u, x = solve_1d_wave_equation(num_x_points, num_t_points, c, x_max, t_max, u_initial, v_initial)

# Plotting
plt.imshow(u, extent=[0, t_max, 0, x_max], origin='lower',
           aspect='auto', cmap='viridis')
plt.colorbar(label='Displacement')
plt.xlabel('Time')
plt.ylabel('Position')
plt.title('1D Wave Equation')
plt.show()