# Loops versus vectorised NumPy: timing a finite-difference update
# Applied ODE & PDE with Python, App. A | (c) 2026 Stephen E. Moore | MIT Licence
import time
import numpy as np

def step_loop(u, r):
    """One explicit heat-equation step written with a Python loop."""
    v = u.copy()
    for j in range(1, len(u) - 1):
        v[j] = u[j] + r * (u[j+1] - 2*u[j] + u[j-1])
    return v

def step_vec(u, r):
    """The same step written with array slices (no Python loop)."""
    v = u.copy()
    v[1:-1] = u[1:-1] + r * (u[2:] - 2*u[1:-1] + u[:-2])
    return v

N, r, nsteps = 100000, 0.4, 20
x = np.linspace(0, 1, N + 1)
u0 = np.sin(np.pi * x)

t0 = time.perf_counter(); u = u0
for _ in range(nsteps):
    u = step_loop(u, r)
t_loop = time.perf_counter() - t0

t0 = time.perf_counter(); w = u0
for _ in range(nsteps):
    w = step_vec(w, r)
t_vec = time.perf_counter() - t0

print(f"same result?      max difference = {np.max(np.abs(u - w)):.1e}")
print(f"loop version      : {t_loop:8.3f} s")
print(f"vectorised version: {t_vec:8.3f} s")
print(f"speed-up          : about {round(t_loop / t_vec, -1):.0f} times (machine dependent)")
