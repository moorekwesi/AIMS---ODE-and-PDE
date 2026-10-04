# The Robertson chemical kinetics problem: explicit versus stiff solvers
# Applied ODE & PDE with Python, Ch. 4 | (c) 2026 Stephen E. Moore | MIT Licence
import time
from scipy.integrate import solve_ivp


def robertson(t, y):
    y1, y2, y3 = y
    return [-0.04 * y1 + 1e4 * y2 * y3,
            0.04 * y1 - 1e4 * y2 * y3 - 3e7 * y2**2,
            3e7 * y2**2]


def jac(t, y):
    y1, y2, y3 = y
    return [[-0.04, 1e4 * y3, 1e4 * y2],
            [0.04, -1e4 * y3 - 6e7 * y2, -1e4 * y2],
            [0.0, 6e7 * y2, 0.0]]


y0 = [1.0, 0.0, 0.0]
print("t_end = 40")
print("method   steps     nfev   njev   cpu time (s)   y1(40)")
for method in ["RK45", "Radau", "BDF", "LSODA"]:
    kw = {} if method == "RK45" else {"jac": jac}
    t0 = time.perf_counter()
    s = solve_ivp(robertson, (0, 40), y0, method=method, rtol=1e-6, atol=1e-10, **kw)
    cpu = time.perf_counter() - t0
    print(f"{method:6s} {s.t.size - 1:7d} {s.nfev:8d} {s.njev:6d}     {cpu:7.2f}      {s.y[0, -1]:.6f}")

# long-time behaviour: only feasible with a stiff solver
s = solve_ivp(robertson, (0, 1e11), y0, method="BDF", jac=jac, rtol=1e-6, atol=1e-12)
print(f"BDF on [0, 1e11]: {s.t.size - 1} steps, y1 = {s.y[0, -1]:.3e}, y3 = {s.y[2, -1]:.6f}")
