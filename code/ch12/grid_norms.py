# Grid norms: the same error measured in different norms
# Applied ODE & PDE with Python, Ch. 12 | (c) 2026 Stephen E. Moore | MIT Licence
import numpy as np


def norm_inf(e):
    return np.max(np.abs(e))


def norm_2h(e, h):
    """Discrete L2 norm  (h sum e_j^2)^(1/2),  approximates (int e^2 dx)^(1/2)."""
    return np.sqrt(h * np.sum(e**2))


def norm_1h(e, h):
    return h * np.sum(np.abs(e))


def order(a, b):
    return np.log2(a / b)


print("smooth error e_j = h^2 sin(pi x_j)    |  layer error e_j = h exp(-(1-x_j)/h)")
print(f"{'N':>5} {'max':>9} {'L2,h':>9} {'l2 (no h)':>10} |"
      f" {'max':>9} {'L2,h':>9} {'L1,h':>9}")
prev = None
for N in [16, 32, 64, 128, 256, 512]:
    h = 1.0 / N
    x = np.linspace(0, 1, N + 1)
    es = h**2 * np.sin(np.pi * x)                # smooth, spread out
    el = h * np.exp(-(1 - x) / h)                # concentrated near x = 1
    row = [norm_inf(es), norm_2h(es, h), np.linalg.norm(es),
           norm_inf(el), norm_2h(el, h), norm_1h(el, h)]
    print(f"{N:5d} {row[0]:9.2e} {row[1]:9.2e} {row[2]:10.2e} |"
          f" {row[3]:9.2e} {row[4]:9.2e} {row[5]:9.2e}")
    if prev is not None:
        rates = [order(p, r) for p, r in zip(prev, row)]
    prev = row
print("observed orders (last refinement):",
      "  ".join(f"{r:5.2f}" for r in rates))

# relative errors: divide by the size of the solution in the same norm
N = 64; h = 1 / N; x = np.linspace(0, 1, N + 1)
u = 1e3 * np.sin(np.pi * x)                      # a solution of size 1000
U = u + h**2 * np.sin(np.pi * x)
print(f"N = 64: absolute max error {norm_inf(U - u):.2e}, "
      f"relative max error {norm_inf(U - u) / norm_inf(u):.2e}")
