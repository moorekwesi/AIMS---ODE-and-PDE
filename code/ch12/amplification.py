# Amplification factors: dissipation and dispersion of the classical schemes
# Applied ODE & PDE with Python, Ch. 12 | (c) 2026 Stephen E. Moore | MIT Licence
import numpy as np
import matplotlib.pyplot as plt

th = np.linspace(1e-6, np.pi, 400)
S = np.sin(th / 2) ** 2
heat = {"FTCS": lambda r: 1 - 4 * r * S,
        "BTCS": lambda r: 1 / (1 + 4 * r * S),
        "Crank-Nicolson": lambda r: (1 - 2 * r * S) / (1 + 2 * r * S)}
adv = {"upwind": lambda v: 1 - v + v * np.exp(-1j * th),
       "Lax-Friedrichs": lambda v: np.cos(th) - 1j * v * np.sin(th),
       "Lax-Wendroff": lambda v: 1 - 2 * v**2 * S - 1j * v * np.sin(th)}

print("max over theta of |g|  (stable iff <= 1)")
print(f"{'scheme':>15} " + " ".join(f"r={r:<5}" for r in [0.25, 0.5, 0.6, 2.0]))
for name, gf in heat.items():
    print(f"{name:>15} " + " ".join(f"{np.max(np.abs(gf(r))):7.3f}"
                                    for r in [0.25, 0.5, 0.6, 2.0]))
print(f"{'scheme':>15} " + " ".join(f"nu={v:<4}" for v in [0.5, 0.8, 1.0, 1.2]))
for name, gf in adv.items():
    print(f"{name:>15} " + " ".join(f"{np.max(np.abs(gf(v))):7.3f}"
                                    for v in [0.5, 0.8, 1.0, 1.2]))

fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
r = 2.0
ax[0].plot(th, np.exp(-r * th**2), "k-", lw=2, label="exact exp(-r theta^2)")
for name, gf in heat.items():
    ax[0].plot(th, gf(r if name != "FTCS" else 0.4), label=name +
               (" (r=0.4)" if name == "FTCS" else ""))
ax[0].axhline(0, color="gray", lw=0.5)
ax[0].set_title("heat, g(theta), r = 2"); ax[0].legend(fontsize=7)
v = 0.8
for name, gf in adv.items():
    G = gf(v)
    ax[1].plot(th, np.abs(G), label=name)
    # relative phase speed: arg g / (-nu theta); exact value 1
    ax[2].plot(th, -np.angle(G) / (v * th), label=name)
ax[1].set_title("advection, |g(theta)|, nu = 0.8"); ax[1].legend(fontsize=7)
ax[2].set_title("relative phase speed, nu = 0.8"); ax[2].legend(fontsize=7)
for a in ax:
    a.set_xlabel("theta = xi h")
plt.tight_layout()
plt.savefig("ch12_amplification.pdf", bbox_inches="tight")
plt.close()
