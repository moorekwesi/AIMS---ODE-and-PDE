# The matrix method: spectral radius versus norms of powers
# Applied ODE & PDE with Python, Ch. 12 | (c) 2026 Stephen E. Moore | MIT Licence
import numpy as np
import matplotlib.pyplot as plt

N = 20
# FTCS for u_t = u_xx with Dirichlet conditions: U^{n+1} = B U^n, B symmetric
T = 2 * np.eye(N - 1) - np.eye(N - 1, k=1) - np.eye(N - 1, k=-1)
print("FTCS, N = 20 (B symmetric, so ||B^n||_2 = rho(B)^n):")
for r in [0.4, 0.5, 0.51, 0.6]:
    B = np.eye(N - 1) - r * T
    rho = np.max(np.abs(np.linalg.eigvals(B)))
    print(f"   r = {r:4.2f}: rho(B) = {rho:.4f}, ||B||_2 = {np.linalg.norm(B, 2):.4f},"
          f" ||B||_inf = {np.linalg.norm(B, np.inf):.4f}")

# Upwind for u_t + u_x = 0 with inflow value U_0 = 0: B = (1-nu) I + nu S
N = 50
print("upwind, N = 50 (B non-normal, all eigenvalues equal 1 - nu):")
fig, ax = plt.subplots(figsize=(7, 3.6))
for nu in [0.8, 1.5]:
    B = (1 - nu) * np.eye(N) + nu * np.eye(N, k=-1)
    P, norms = np.eye(N), []
    for n in range(1, 151):
        P = B @ P
        norms.append(np.linalg.norm(P, 2))
    print(f"   nu = {nu}: rho(B) = {abs(1 - nu):.2f}, ||B^n||_2 at n = 10, 50, 150:"
          f" {norms[9]:.2e}, {norms[49]:.2e}, {norms[149]:.2e}")
    ax.semilogy(range(1, 151), norms, label=f"nu = {nu}")
    ax.semilogy(range(1, 151), abs(1 - nu) ** np.arange(1, 151), "--",
                label=f"rho(B)^n, nu = {nu}")
ax.set_xlabel("time step n"); ax.set_ylabel("||B^n||_2")
ax.legend(); plt.tight_layout()
plt.savefig("ch12_nonnormal.pdf", bbox_inches="tight")
plt.close()
