# Dam break for the shallow-water equations with the Rusanov (local Lax-Friedrichs) flux
# Applied ODE & PDE with Python, Ch. 11 | (c) 2026 Stephen E. Moore | MIT Licence
import numpy as np
from scipy.optimize import brentq
import matplotlib.pyplot as plt

g, hL, hR, T, xa, xb = 9.81, 2.0, 1.0, 0.5, -5.0, 5.0


def physical_flux(q):
    h, m = q                                     # q = (h, hu)
    return np.array([m, m**2 / h + 0.5 * g * h**2])


def rusanov(N):
    dx = (xb - xa) / N; x = xa + (np.arange(N) + 0.5) * dx
    q = np.array([np.where(x < 0, hL, hR), np.zeros(N)])
    t = 0.0
    while t < T - 1e-12:
        h, u = q[0], q[1] / q[0]
        dt = min(0.9 * dx / np.max(np.abs(u) + np.sqrt(g * h)), T - t)
        qe = np.concatenate([q[:, :1], q, q[:, -1:]], axis=1)   # transmissive ends
        qL, qR = qe[:, :-1], qe[:, 1:]
        a = np.maximum(np.abs(qL[1] / qL[0]) + np.sqrt(g * qL[0]),
                       np.abs(qR[1] / qR[0]) + np.sqrt(g * qR[0]))
        F = 0.5 * (physical_flux(qL) + physical_flux(qR)) - 0.5 * a * (qR - qL)
        q = q - dt / dx * (F[:, 1:] - F[:, :-1]); t += dt
    return x, q[0], q[1] / q[0]


def exact(x, t):
    """Exact dam-break solution on a wet bed: rarefaction + middle state + shock."""
    cL, cR = np.sqrt(g * hL), np.sqrt(g * hR)
    phi = lambda hm: (2 * (cL - np.sqrt(g * hm))
                      - (hm - hR) * np.sqrt(0.5 * g * (hm + hR) / (hm * hR)))
    hm = brentq(phi, hR, hL); um = 2 * (cL - np.sqrt(g * hm)); s = hm * um / (hm - hR)
    xi = x / t
    h = np.where(xi < -cL, hL, np.where(xi < um - np.sqrt(g * hm),
                 (2 * cL - xi)**2 / (9 * g), np.where(xi < s, hm, hR)))
    u = np.where(xi < -cL, 0.0, np.where(xi < um - np.sqrt(g * hm),
                 2 / 3 * (cL + xi), np.where(xi < s, um, 0.0)))
    return h, u, hm, um, s


_, _, hm, um, s = exact(np.zeros(1), T)
print(f"exact middle state: h_m = {hm:.4f}, u_m = {um:.4f}, shock speed s = {s:.4f}")
print("    N    L1 error in h   rate    shock position (exact {:.4f})".format(s * T))
prev = None
for N in (100, 200, 400, 800, 1600):
    x, h, u = rusanov(N)
    he = exact(x, T)[0]; e = np.sum(np.abs(h - he)) * (xb - xa) / N
    j = np.where(h > 0.5 * (hm + hR))[0][-1]    # last cell above the mid level
    print(f"{N:5d}    {e:.3e}      " + ("  -- " if prev is None else f"{np.log2(prev / e):5.2f}")
          + f"       {x[j]:.4f}")
    prev = e

x, h, u = rusanov(400); he, ue, *_ = exact(x, T)
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
ax[0].plot(x, he, "k-", label="exact"); ax[0].plot(x, h, "r.", ms=2, label="Rusanov, N = 400")
ax[1].plot(x, ue, "k-", label="exact"); ax[1].plot(x, u, "r.", ms=2, label="Rusanov, N = 400")
ax[0].set_ylabel("depth h"); ax[1].set_ylabel("velocity u")
for a in ax:
    a.set_xlabel("x"); a.legend(fontsize=8)
plt.tight_layout(); plt.savefig("ch11_dambreak.pdf", bbox_inches="tight")
plt.close()
