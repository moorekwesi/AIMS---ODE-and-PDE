# Series solution of the heat equation with Dirichlet conditions
# Applied ODE & PDE with Python, Ch. 8 | (c) 2026 Stephen E. Moore | MIT Licence
import numpy as np
import matplotlib.pyplot as plt


def heat_series(x, t, N, kappa=1.0, L=1.0):
    """u_t = kappa u_xx, u(0)=u(L)=0, u(x,0)=1: sum over odd n <= N."""
    n = np.arange(1, N + 1, 2)
    bn = 4 / (n * np.pi)
    lam = kappa * (n * np.pi / L) ** 2                  # decay rates
    return np.sin(np.outer(x, n) * np.pi / L) @ (bn * np.exp(-lam * t))


print("    t      u(0.5,t)       one-term     terms for 1e-10")
for t in [0.001, 0.01, 0.05, 0.1, 0.2, 0.5]:
    ref = heat_series(np.array([0.5]), t, 4001)[0]
    one = 4 / np.pi * np.exp(-np.pi**2 * t)
    N = 1
    while abs(heat_series(np.array([0.5]), t, N)[0] - ref) > 1e-10:
        N += 2
    print(f"{t:7.3f}  {ref:.10f}  {one:.10f}  {(N + 1) // 2:6d}")

# Physical time scale for a laterite wall
L, kappa = 0.30, 5.0e-7                       # m, m^2/s
tau1 = L**2 / (kappa * np.pi**2)              # e-folding time of mode 1
print(f"\nlaterite wall: L^2/kappa = {L**2 / kappa / 3600:.1f} h, "
      f"1/lambda_1 = {tau1 / 3600:.2f} h")

x = np.linspace(0, 1, 401)
plt.figure(figsize=(7, 3.8))
for t in [0.0005, 0.005, 0.02, 0.05, 0.1, 0.2]:
    plt.plot(x, heat_series(x, t, 401), label=f"$t={t}$")
plt.xlabel("$x$"); plt.ylabel("$u(x,t)$"); plt.legend(fontsize=8, ncol=2)
plt.tight_layout()
plt.savefig("ch08_heat_series.pdf", bbox_inches="tight")
