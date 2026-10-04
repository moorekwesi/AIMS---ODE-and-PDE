# SciPy integrate and optimize: quad, solve_ivp, solve_bvp, brentq, fsolve
# Applied ODE & PDE with Python, App. A | (c) 2026 Stephen E. Moore | MIT Licence
import numpy as np
from scipy.integrate import quad, solve_ivp, solve_bvp
from scipy.optimize import brentq, fsolve

# quad: int_0^inf exp(-x^2) dx = sqrt(pi)/2
val, est = quad(lambda x: np.exp(-x**2), 0, np.inf)
print(f"quad      : {val:.12f}  (exact {np.sqrt(np.pi)/2:.12f}, error estimate {est:.1e})")

# solve_ivp: pendulum theta'' = -sin(theta) as a first-order system, with dense output
def pendulum(t, z):
    theta, omega = z
    return [omega, -np.sin(theta)]

def crossing(t, z):                 # event: theta passes through 0 going downwards
    return z[0]
crossing.direction = -1

sol = solve_ivp(pendulum, (0, 20), [1.0, 0.0], method="DOP853", rtol=1e-10,
                atol=1e-12, dense_output=True, events=crossing)
print(f"solve_ivp : theta(5) = {sol.sol(5.0)[0]:.8f},  status = {sol.status}")
tev = sol.t_events[0]
print(f"            period from events = {tev[1] - tev[0]:.8f}")

# solve_bvp: y'' = -y, y(0) = 0, y(pi/2) = 1  (exact y = sin x)
def rhs(x, Y):
    return np.vstack([Y[1], -Y[0]])

def bc(Ya, Yb):
    return np.array([Ya[0], Yb[0] - 1.0])

xm = np.linspace(0, np.pi/2, 11)
bvp = solve_bvp(rhs, bc, xm, np.zeros((2, xm.size)))
xx = np.linspace(0, np.pi/2, 101)
print(f"solve_bvp : success = {bvp.success}, max error = "
      f"{np.max(np.abs(bvp.sol(xx)[0] - np.sin(xx))):.2e}")

# Root finding: a bracketing method (brentq) and a system (fsolve)
r = brentq(lambda x: x - np.cos(x), 0, 1)
print(f"brentq    : root of x = cos x is {r:.12f}")
sys = lambda z: [z[0]**2 + z[1]**2 - 4, z[0] - z[1]**3]
print("fsolve    :", np.round(fsolve(sys, [1.0, 1.0]), 10))
