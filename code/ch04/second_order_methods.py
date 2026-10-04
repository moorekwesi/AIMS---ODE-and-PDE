# Taylor, Heun and midpoint methods of order two
# Applied ODE & PDE with Python, Ch. 4 | (c) 2026 Stephen E. Moore | MIT Licence
import numpy as np

f = lambda t, y: y - t**2 + 1
ft = lambda t, y: -2 * t            # partial derivative of f with respect to t
fy = lambda t, y: 1.0               # partial derivative of f with respect to y
exact = lambda t: (t + 1)**2 - 0.5 * np.exp(t)


def step_euler(t, y, h):
    return y + h * f(t, y)


def step_taylor2(t, y, h):
    ypp = ft(t, y) + fy(t, y) * f(t, y)          # y'' = f_t + f_y f
    return y + h * f(t, y) + 0.5 * h**2 * ypp


def step_heun(t, y, h):
    k1 = f(t, y)
    k2 = f(t + h, y + h * k1)
    return y + 0.5 * h * (k1 + k2)


def step_midpoint(t, y, h):
    k1 = f(t, y)
    k2 = f(t + 0.5 * h, y + 0.5 * h * k1)
    return y + h * k2


def solve(step, N, T=2.0, y0=0.5):
    h, t, y = T / N, 0.0, y0
    for n in range(N):
        y = step(t, y, h)
        t += h
    return y


methods = {"Euler": step_euler, "Taylor2": step_taylor2,
           "Heun": step_heun, "Midpoint": step_midpoint}
Ns = [10, 20, 40, 80, 160]
print("errors |y_N - y(2)|")
print("   N   " + "".join(f"{name:>12s}" for name in methods))
E = {name: [abs(solve(s, N) - exact(2.0)) for N in Ns] for name, s in methods.items()}
for i, N in enumerate(Ns):
    print(f"{N:4d}   " + "".join(f"{E[name][i]:12.3e}" for name in methods))
print("observed order log2(E(h)/E(h/2)) on the two finest grids:")
print("       " + "".join(f"{np.log2(E[n][-2] / E[n][-1]):12.3f}" for n in methods))
