# Grid refinement study, three-grid order estimate and Richardson extrapolation
# Applied ODE & PDE with Python, Ch. 12 | (c) 2026 Stephen E. Moore | MIT Licence
import numpy as np
from scipy.linalg import solve_banded

u = lambda x: np.exp(x) * np.sin(np.pi * x) + x
f = lambda x: np.exp(x) * ((np.pi**2 - 1) * np.sin(np.pi * x)
                           - 2 * np.pi * np.cos(np.pi * x))


def solve(N):
    """Second-order FD solution of -u'' = f, Dirichlet data from u."""
    h = 1.0 / N
    x = np.linspace(0, 1, N + 1)
    ab = np.zeros((3, N - 1))
    ab[0, 1:], ab[1], ab[2, :-1] = -1.0, 2.0, -1.0
    b = h**2 * f(x[1:-1]); b[0] += u(0); b[-1] += u(1)
    return x, np.concatenate(([u(0)], solve_banded((1, 1), ab, b), [u(1)]))


Q_exact = u(0.5)                    # quantity of interest: u(1/2)
print(f"{'n':>2} {'N':>4} {'max error':>10} {'p':>5} {'Q_h = U(1/2)':>14} "
      f"{'p (3 grids)':>11} {'|Q_h - u|':>10} {'|R_h - u|':>10}")
E, Q = [], []
for n in range(1, 8):
    N = 2**n
    x, U = solve(N)
    E.append(np.max(np.abs(U - u(x))))
    Q.append(U[N // 2])
    p = f"{np.log2(E[-2] / E[-1]):5.2f}" if n > 1 else " " * 5
    p3 = (f"{np.log2((Q[-3] - Q[-2]) / (Q[-2] - Q[-1])):11.3f}" if n > 2
          else " " * 11)
    # Richardson: R_h = (4 Q_h - Q_{2h}) / 3 removes the h^2 term
    R = (f"{abs((4 * Q[-1] - Q[-2]) / 3 - Q_exact):10.2e}" if n > 1 else "")
    print(f"{n:2d} {N:4d} {E[-1]:10.3e} {p} {Q[-1]:14.10f} {p3} "
          f"{abs(Q[-1] - Q_exact):10.2e} {R}")
print(f"exact u(1/2) = {Q_exact:.10f}")
