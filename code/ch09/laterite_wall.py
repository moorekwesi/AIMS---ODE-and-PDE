# Steady heat flow through a plastered laterite wall, -(k u')' = 0
# Applied ODE & PDE with Python, Ch. 9 | (c) 2026 Stephen E. Moore | MIT Licence
import numpy as np
import scipy.sparse as sps
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

# layers (thickness in m, conductivity in W/(m K)), illustrative values
layers = [(0.02, 0.72), (0.20, 0.80), (0.02, 0.72)]   # plaster|laterite|plaster
L = sum(t for t, _ in layers)
T_out, T_room, H = 45.0, 25.0, 8.0    # sunlit face, room air, film coefficient
edges = np.cumsum([0] + [t for t, _ in layers])               # layer interfaces
cumR = np.cumsum([0] + [t / kk for t, kk in layers])        # int_0^edge dx/k


def k(x):
    """Piecewise constant conductivity."""
    return np.select([(x >= a) & (x < b) for a, b in zip(edges[:-1], edges[1:])],
                     [kk for _, kk in layers], default=layers[-1][1])


def Rint(x):
    """Thermal resistance R(x) = int_0^x ds/k(s) (piecewise linear in x)."""
    return np.interp(x, edges, cumR)


def solve_wall(N, mean):
    """Conservative scheme: (k_{j+1/2}(U_j-U_{j+1}) + k_{j-1/2}(U_j-U_{j-1}))/h^2 = 0,
    U_0 = T_out, Robin at x = L:  -k (U_N - U_{N-1})/h = H (U_N - T_room)."""
    h = L / N
    x = np.linspace(0, L, N + 1)
    kj = k(x)
    if mean == "cell-average":                       # h / int_{x_j}^{x_j+1} dx/k
        kh = h / np.diff(Rint(x))
    elif mean == "arithmetic":
        kh = 0.5 * (kj[:-1] + kj[1:])
    else:                                            # harmonic mean
        kh = 2 * kj[:-1] * kj[1:] / (kj[:-1] + kj[1:])
    main = np.zeros(N + 1); lo = np.zeros(N); up = np.zeros(N)
    main[1:N] = kh[:-1] + kh[1:]
    lo[:N - 1] = -kh[:-1]; up[1:] = -kh[1:]
    b = np.zeros(N + 1)
    main[0], up[0], b[0] = 1.0, 0.0, T_out           # Dirichlet row
    main[N], lo[N - 1] = kh[-1] + H * h, -kh[-1]     # Robin row (times h)
    b[N] = H * h * T_room
    A = sps.diags([lo, main, up], [-1, 0, 1], format="csr")
    return x, spsolve(A, b), kh


R = sum(t / kk for t, kk in layers) + 1 / H          # thermal resistances in series
q = (T_out - T_room) / R
print(f"exact: flux q = {q:.4f} W/m^2, inner surface T = {T_room + q / H:.4f} C")
print(f"{'N':>4} {'k_{j+1/2}':>12} {'flux':>9} {'T(L)':>9} {'max error':>10}")
for N in [25, 50, 100, 200]:
    for mean in ["arithmetic", "harmonic", "cell-average"]:
        x, U, kh = solve_wall(N, mean)
        exact = T_out - q * Rint(x)      # exact piecewise linear profile
        flux = H * (U[-1] - T_room)
        print(f"{N:4d} {mean:>12} {flux:9.4f} {U[-1]:9.4f} "
              f"{np.max(np.abs(U - exact)):10.2e}")

x, U, _ = solve_wall(120, "cell-average")
plt.figure(figsize=(7, 3.5))
plt.plot(100 * x, U, "-", lw=2)
for e in edges[1:-1]:
    plt.axvline(100 * e, color="gray", ls=":")
plt.xlabel("depth into wall (cm)"); plt.ylabel("temperature (C)")
plt.title("plaster | laterite block | plaster")
plt.tight_layout()
plt.savefig("ch09_laterite_wall.pdf", bbox_inches="tight")
plt.close()
