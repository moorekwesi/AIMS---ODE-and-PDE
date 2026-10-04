# Amplification factor of the FTCS scheme for the heat equation
# Applied ODE & PDE with Python, Ch. 10 | (c) 2026 Stephen E. Moore | MIT Licence
import numpy as np
import matplotlib.pyplot as plt

theta = np.linspace(0.0, np.pi, 401)


def g_ftcs(theta, r):
    """g(theta) = 1 - 4 r sin^2(theta/2)."""
    return 1.0 - 4.0 * r * np.sin(theta / 2)**2


print("   r     g(pi)    max|g|   steps to amplify by 1e6")
for r in (0.25, 0.4, 0.5, 0.6, 1.0):
    gmax = np.max(np.abs(g_ftcs(theta, r)))
    steps = "never" if gmax <= 1 else f"{np.log(1e6) / np.log(gmax):.0f}"
    print(f"{r:5.2f}  {g_ftcs(np.pi, r):7.3f}  {gmax:7.3f}   {steps}")

plt.figure(figsize=(7, 4))
for r in (0.25, 0.4, 0.5, 0.6):
    plt.plot(theta, g_ftcs(theta, r), label=f"FTCS, r = {r}")
plt.plot(theta, np.exp(-0.4 * theta**2), "k--", label="exact exp(-r theta^2), r = 0.4")
plt.axhline(1, color="gray", lw=0.6); plt.axhline(-1, color="gray", lw=0.6)
plt.fill_between(theta, -1, 1, color="green", alpha=0.06)
plt.xlabel("theta = xi h"); plt.ylabel("g(theta)"); plt.ylim(-1.5, 1.1)
plt.xticks([0, np.pi / 2, np.pi], ["0", "pi/2", "pi"]); plt.legend(fontsize=9)
plt.tight_layout(); plt.savefig("ch10_amplification.pdf", bbox_inches="tight")
plt.close()
