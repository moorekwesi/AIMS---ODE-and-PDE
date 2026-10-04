# Stability of the 1D Poisson scheme: norms of the inverse and Green's function
# Applied ODE & PDE with Python, Ch. 12 | (c) 2026 Stephen E. Moore | MIT Licence
import numpy as np
import matplotlib.pyplot as plt


def poisson_matrix(N):
    """A = (1/h^2) tridiag(-1, 2, -1) of size N-1 (dense, small N only)."""
    h = 1.0 / N
    m = N - 1
    return (2 * np.eye(m) - np.eye(m, k=1) - np.eye(m, k=-1)) / h**2


print(f"{'N':>5} {'||A^-1||_inf':>13} {'||A^-1||_2':>11} {'min entry':>10}")
for N in [4, 8, 16, 32, 64, 128, 256]:
    Ainv = np.linalg.inv(poisson_matrix(N))
    print(f"{N:5d} {np.linalg.norm(Ainv, np.inf):13.6f} "
          f"{np.linalg.norm(Ainv, 2):11.6f} {Ainv.min():10.2e}")
print(f"limits: 1/8 = {1/8:.6f},  1/pi^2 = {1/np.pi**2:.6f}")

# Columns of A^{-1} are h * G(x_i, x_j): the discrete Green's function
N = 16
h = 1 / N
x = np.linspace(0, 1, N + 1)[1:-1]
Ainv = np.linalg.inv(poisson_matrix(N))
G = lambda x, s: np.where(x <= s, x * (1 - s), s * (1 - x))   # continuous G
print(f"N = {N}: max |A^-1 - h G(x_i, x_j)| = "
      f"{np.max(np.abs(Ainv - h * G(x[:, None], x[None, :]))):.2e}")

plt.figure(figsize=(7, 3.6))
xf = np.linspace(0, 1, 200)
for j in [3, 7, 11]:
    plt.plot(x, Ainv[:, j] / h, "o", ms=4, label=f"column j = {j + 1}")
    plt.plot(xf, G(xf, x[j]), "k-", lw=0.7)
plt.xlabel("x"); plt.ylabel("(A^-1)_{ij} / h  and  G(x, x_j)")
plt.legend()
plt.tight_layout()
plt.savefig("ch12_green.pdf", bbox_inches="tight")
plt.close()
