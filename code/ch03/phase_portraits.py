# Phase portraits of x' = A x and the trace-determinant classification
# Applied ODE & PDE with Python, Ch. 3 | (c) 2026 Stephen E. Moore | MIT Licence
import numpy as np
import matplotlib.pyplot as plt


def classify(A, tol=1e-12):
    """Type of the equilibrium 0 of x' = A x from trace and determinant."""
    T, D = np.trace(A), np.linalg.det(A)
    disc = T**2 - 4 * D
    if D < -tol:
        return "saddle"
    if abs(D) <= tol:
        return "degenerate (det = 0)"
    if abs(T) <= tol:
        return "centre" if disc < 0 else "?"
    stab = "stable" if T < 0 else "unstable"
    if disc < -tol:
        return stab + " spiral"
    if abs(disc) <= tol:
        return stab + " degenerate node"
    return stab + " node"


examples = [np.array(M, dtype=float) for M in (
    [[1, 1], [4, 1]],          # eigenvalues 3, -1
    [[-3, 1], [1, -3]],        # -2, -4
    [[1, 0], [0, 2]],          # 1, 2
    [[-0.5, 2], [-2, -0.5]],   # -0.5 +- 2i
    [[0, 1], [-4, 0]],         # +- 2i
    [[-1, 1], [0, -1]],        # -1 (double, one eigenvector)
)]

fig, axs = plt.subplots(2, 3, figsize=(10.5, 6.6))
g = np.linspace(-2, 2, 25)
X, Y = np.meshgrid(g, g)
for A, ax in zip(examples, axs.flat):
    kind = classify(A)
    ev = np.linalg.eigvals(A)
    print(f"tr = {np.trace(A):5.2f}, det = {np.linalg.det(A):5.2f}, "
          f"eig = {np.array2string(ev, precision=3)}: {kind}")
    U, V = A[0, 0] * X + A[0, 1] * Y, A[1, 0] * X + A[1, 1] * Y
    ax.streamplot(X, Y, U, V, density=0.9, color="C0", linewidth=0.8, arrowsize=0.8)
    lam, vec = np.linalg.eig(A)
    for l, v in zip(lam, vec.T):                 # real eigenvector directions
        if abs(l.imag) < 1e-12:
            ax.plot([-2 * v[0].real, 2 * v[0].real], [-2 * v[1].real, 2 * v[1].real],
                    "C3", lw=1.5)
    ax.plot(0, 0, "ko", ms=4)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_title(kind, fontsize=10)
    ax.set_aspect("equal")
plt.tight_layout()
plt.savefig("ch03_phase_portraits.pdf", bbox_inches="tight")
plt.close()
