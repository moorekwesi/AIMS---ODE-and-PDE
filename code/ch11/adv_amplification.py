# Amplification factors of four advection schemes: magnitude and phase
# Applied ODE & PDE with Python, Ch. 11 | (c) 2026 Stephen E. Moore | MIT Licence
import numpy as np
import matplotlib.pyplot as plt

E = lambda th, k=1: np.exp(-1j * k * th)        # shift operator symbol e^{-ik theta}
schemes = {
    "upwind": lambda th, nu: 1 - nu * (1 - E(th)),
    "Lax-Friedrichs": lambda th, nu: np.cos(th) - 1j * nu * np.sin(th),
    "Lax-Wendroff": lambda th, nu: 1 - 1j * nu * np.sin(th) - nu**2 * (1 - np.cos(th)),
    "Beam-Warming": lambda th, nu: (1 - nu / 2 * (3 - 4 * E(th) + E(th, 2))
                                    + nu**2 / 2 * (1 - 2 * E(th) + E(th, 2))),
}

nu = 0.8
th = np.linspace(1e-3, np.pi, 400)
print(f"nu = {nu}")
print("scheme            |g(pi/4)|  |g(pi/2)|   phase ratio at pi/4, pi/2")
for name, g in schemes.items():
    gq, gh = g(np.pi / 4, nu), g(np.pi / 2, nu)
    rq = np.angle(gq) / (-nu * np.pi / 4); rh = np.angle(gh) / (-nu * np.pi / 2)
    print(f"{name:16s}   {abs(gq):.4f}     {abs(gh):.4f}      {rq:.4f}  {rh:.4f}")
print("max |g| over theta for nu = 1.2:",
      ", ".join(f"{n} {np.max(abs(g(th, 1.2))):.2f}" for n, g in schemes.items()))

fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
for name, g in schemes.items():
    G = g(th, nu)
    ax[0].plot(th, abs(G), label=name)
    ax[1].plot(th, np.unwrap(np.angle(G)) / (-nu * th), label=name)
ax[0].set_ylabel("|g(theta)|"); ax[1].set_ylabel("relative phase arg(g)/(-nu theta)")
ax[1].axhline(1, color="k", lw=0.6); ax[1].set_ylim(-0.2, 2.2)
for a in ax:
    a.set_xlabel("theta"); a.set_xticks([0, np.pi / 2, np.pi])
    a.set_xticklabels(["0", "pi/2", "pi"]); a.legend(fontsize=8)
fig.suptitle(f"Courant number nu = {nu}")
plt.tight_layout(); plt.savefig("ch11_amplification.pdf", bbox_inches="tight")
plt.close()
