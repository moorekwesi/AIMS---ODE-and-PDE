# Energy of the undamped and damped vibrating string
# Applied ODE & PDE with Python, Ch. 8 | (c) 2026 Stephen E. Moore | MIT Licence
import numpy as np

# u_tt + 2 gamma u_t = c^2 u_xx on (0,1), u = 0 at the ends,
# u(x,0) = sin(pi x), u_t(x,0) = 0.  One mode: u = T(t) sin(pi x).
c, w = 1.0, np.pi                       # omega_1 = c*pi


def mode(t, gamma):
    """Return T(t), T'(t) for T'' + 2 gamma T' + w^2 T = 0, T(0)=1, T'(0)=0."""
    mu = np.sqrt(w**2 - gamma**2)       # under-damped: gamma < w
    T = np.exp(-gamma * t) * (np.cos(mu * t) + gamma / mu * np.sin(mu * t))
    dT = -np.exp(-gamma * t) * (w**2 / mu) * np.sin(mu * t)
    return T, dT


def energy(t, gamma):
    """E = 1/2 int_0^1 (u_t^2 + c^2 u_x^2) dx = (T'^2 + w^2 T^2)/4."""
    T, dT = mode(t, gamma)
    return 0.25 * (dT**2 + w**2 * T**2)


print("   t     E (gamma=0)    E (gamma=0.5)    E(0) exp(-2 gamma t)")
for t in [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]:
    print(f"{t:4.1f}   {energy(t, 0.0):.8f}    {energy(t, 0.5):.8f}"
          f"       {energy(0, 0.5) * np.exp(-2 * 0.5 * t):.8f}")
# check dE/dt = -2 gamma int u_t^2 = -gamma T'^2 at t = 1
t, g, d = 1.0, 0.5, 1e-6
dE = (energy(t + d, g) - energy(t - d, g)) / (2 * d)
print(f"\nt = 1: dE/dt = {dE:.8f},  -gamma*T'^2 = {-g * mode(t, g)[1]**2:.8f}")
