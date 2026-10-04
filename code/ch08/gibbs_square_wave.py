# Partial sums of the square wave and the Gibbs phenomenon
# Applied ODE & PDE with Python, Ch. 8 | (c) 2026 Stephen E. Moore | MIT Licence
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad


def S(x, N):
    """Partial sum up to harmonic N of the square wave sign(x) on (-pi, pi)."""
    k = np.arange(1, N + 1, 2)                      # odd harmonics only
    return (4 / np.pi) * np.sum(np.sin(np.outer(x, k)) / k, axis=1)


print("   N    max S_N    x of max    overshoot/jump")
for N in [9, 19, 49, 99, 199, 999]:
    x_star = np.pi / (N + 1)                        # first maximum of S_N
    m = S(np.array([x_star]), N)[0]
    print(f"{N:4d}  {m:.6f}   {x_star:.6f}    {(m - 1) / 2:.4%}")

# The limit: (2/pi) * Si(pi), Si = sine integral
G = (2 / np.pi) * quad(lambda t: np.sin(t) / t, 0, np.pi)[0]
print(f"limit (2/pi) Si(pi) = {G:.6f},  overshoot = {(G - 1) / 2:.4%} of jump")

x = np.linspace(-np.pi, np.pi, 2001)
fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
ax[0].plot(x, np.sign(x), "k", lw=1, label="sign$(x)$")
for N, ls in [(5, ":"), (15, "--"), (49, "-")]:
    ax[0].plot(x, S(x, N), ls, lw=1.2, label=f"$S_{{{N}}}$")
ax[0].set_xlabel("$x$"); ax[0].legend(fontsize=8, loc="upper left")
xz = np.linspace(0, 0.5, 1001)
for N in [49, 99, 199]:
    ax[1].plot(xz, S(xz, N), lw=1.2, label=f"$S_{{{N}}}$")
ax[1].axhline(G, color="r", ls="--", lw=0.8, label="1.17898")
ax[1].set_xlabel("$x$"); ax[1].set_ylim(0.8, 1.25); ax[1].legend(fontsize=8)
plt.tight_layout()
plt.savefig("ch08_gibbs.pdf", bbox_inches="tight")
