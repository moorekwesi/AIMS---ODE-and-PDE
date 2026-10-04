# Eigenvalues of the discrete Laplacian and the condition number
# Applied ODE & PDE with Python, Ch. 9 | (c) 2026 Stephen E. Moore | MIT Licence
import numpy as np
import scipy.sparse as sps
import matplotlib.pyplot as plt


def A1d(N):
    """-d^2/dx^2 with Dirichlet conditions: (1/h^2) tridiag(-1, 2, -1)."""
    h = 1.0 / N
    return sps.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(N - 1, N - 1)) / h**2


def lam_exact(N):
    """lambda_k = (4/h^2) sin^2(k pi h / 2),  k = 1..N-1."""
    h, k = 1.0 / N, np.arange(1, N)
    return 4 / h**2 * np.sin(k * np.pi * h / 2) ** 2


N = 10
lam_num = np.linalg.eigvalsh(A1d(N).toarray())
lam = lam_exact(N)
print(f"N = {N}: max |eigvalsh - formula| = {np.max(np.abs(lam_num - lam)):.2e}")
print(f"{'k':>2} {'lambda_k (h)':>13} {'(k pi)^2':>10} {'ratio':>7}")
for k in range(1, N):
    print(f"{k:2d} {lam[k-1]:13.4f} {(k*np.pi)**2:10.4f} {lam[k-1]/(k*np.pi)**2:7.4f}")

print(f"{'N':>5} {'lambda_min':>11} {'lambda_max':>11} {'cond_2(A)':>11} "
      f"{'4/(pi h)^2':>11}")
for N in [10, 20, 40, 80, 160, 320]:
    lam = lam_exact(N)
    print(f"{N:5d} {lam[0]:11.5f} {lam[-1]:11.4e} {lam[-1]/lam[0]:11.4e} "
          f"{4 * N**2 / np.pi**2:11.4e}")

plt.figure(figsize=(7, 3.6))
for N in [10, 40, 160]:
    k = np.arange(1, N)
    plt.plot(k / N, lam_exact(N) / (k * np.pi) ** 2, ".-", label=f"N = {N}")
th = np.linspace(1e-3, 1, 200)
plt.plot(th, (np.sin(np.pi * th / 2) / (np.pi * th / 2)) ** 2, "k:",
         label="sinc^2 curve")
plt.xlabel("k h  (fraction of resolvable frequencies)")
plt.ylabel("lambda_k(h) / (k pi)^2")
plt.legend()
plt.tight_layout()
plt.savefig("ch09_laplacian_eigs.pdf", bbox_inches="tight")
plt.close()
