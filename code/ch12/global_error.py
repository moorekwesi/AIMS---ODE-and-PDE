# The global error equation A E = -tau for the 1D Poisson problem
# Applied ODE & PDE with Python, Ch. 12 | (c) 2026 Stephen E. Moore | MIT Licence
import numpy as np
import scipy.sparse as sps
from scipy.sparse.linalg import spsolve

u = lambda x: np.exp(x) * np.sin(np.pi * x) + x
f = lambda x: np.exp(x) * ((np.pi**2 - 1) * np.sin(np.pi * x)
                           - 2 * np.pi * np.cos(np.pi * x))    # f = -u''

print(f"{'N':>5} {'||tau||':>10} {'||tau||/h^2':>11} {'||E||':>10} "
      f"{'bound':>10} {'||E + A^-1 tau||':>17}")
for N in [8, 16, 32, 64, 128, 256]:
    h = 1.0 / N
    x = np.linspace(0, 1, N + 1)
    A = sps.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(N - 1, N - 1),
                  format="csc") / h**2
    b = f(x[1:-1]); b[0] += u(0) / h**2; b[-1] += u(1) / h**2
    U = spsolve(A, b)                                   # numerical solution
    ue = u(x)
    tau = (-ue[:-2] + 2 * ue[1:-1] - ue[2:]) / h**2 - f(x[1:-1])  # exact u in scheme
    E = U - ue[1:-1]                                    # global error
    print(f"{N:5d} {np.abs(tau).max():10.3e} {np.abs(tau).max() / h**2:11.4f} "
          f"{np.abs(E).max():10.3e} {np.abs(tau).max() / 8:10.3e} "
          f"{np.abs(E + spsolve(A, tau)).max():17.2e}")
