# A consistent but zero-unstable two-step method
# Applied ODE & PDE with Python, Ch. 4 | (c) 2026 Stephen E. Moore | MIT Licence
import numpy as np

# y_{n+2} + 4 y_{n+1} - 5 y_n = h (4 f_{n+1} + 2 f_n): order 3, roots of rho: 1 and -5
f = lambda t, y: -y                     # test problem y' = -y, y(0) = 1


def unstable_two_step(N, T=1.0):
    h = T / N
    y = np.zeros(N + 1)
    y[0], y[1] = 1.0, np.exp(-h)        # exact starting values
    for n in range(N - 1):
        y[n + 2] = -4 * y[n + 1] + 5 * y[n] + h * (4 * f(0, y[n + 1]) + 2 * f(0, y[n]))
    return y[-1]


print("    N      h        error at t = 1")
for N in [10, 20, 40, 80]:
    print(f"{N:5d}  {1 / N:.4f}     {abs(unstable_two_step(N) - np.exp(-1)):.3e}")
print("roots of rho(z) = z^2 + 4z - 5:", np.roots([1, 4, -5]))
