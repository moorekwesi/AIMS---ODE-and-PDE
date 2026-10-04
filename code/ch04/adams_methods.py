# Adams-Bashforth, Adams-Moulton predictor-corrector and empirical orders
# Applied ODE & PDE with Python, Ch. 4 | (c) 2026 Stephen E. Moore | MIT Licence
import numpy as np

f = lambda t, y: y * np.cos(t)
exact = lambda t: np.exp(np.sin(t))


def rk4_step(t, y, h):
    k1 = f(t, y)
    k2 = f(t + h / 2, y + h / 2 * k1)
    k3 = f(t + h / 2, y + h / 2 * k2)
    k4 = f(t + h, y + h * k3)
    return y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def multistep(N, T=5.0, scheme="AB4"):
    h = T / N
    t = h * np.arange(N + 1)
    y = np.zeros(N + 1)
    y[0] = 1.0
    for n in range(3):                       # starting values from RK4
        y[n + 1] = rk4_step(t[n], y[n], h)
    F = f(t, y)                              # stored slopes f_n (only first 4 used)
    for n in range(3, N):
        if scheme == "AB2":
            y[n + 1] = y[n] + h / 2 * (3 * F[n] - F[n - 1])
        else:   # AB4 predictor
            y[n + 1] = y[n] + h / 24 * (55 * F[n] - 59 * F[n - 1] + 37 * F[n - 2] - 9 * F[n - 3])
        if scheme == "ABM4":                 # PECE with the 4th-order Adams-Moulton corrector
            fp = f(t[n + 1], y[n + 1])
            y[n + 1] = y[n] + h / 24 * (9 * fp + 19 * F[n] - 5 * F[n - 1] + F[n - 2])
        F[n + 1] = f(t[n + 1], y[n + 1])
    return y[-1]


Ns = [50, 100, 200, 400, 800]
print("   N   " + "".join(f"{s:>12s}" for s in ["AB2", "AB4", "ABM4 (PECE)"]))
E = {s: [abs(multistep(N, scheme=s) - exact(5.0)) for N in Ns] for s in ["AB2", "AB4", "ABM4"]}
for i, N in enumerate(Ns):
    print(f"{N:5d} " + "".join(f"{E[s][i]:12.3e}" for s in E))
print("order  " + "".join(f"{np.log2(E[s][-2] / E[s][-1]):12.3f}" for s in E))
print(f"error ratio AB4/ABM4 at N = 800: {E['AB4'][-1] / E['ABM4'][-1]:.1f}")
