# Linear advection: unstable FTCS versus the upwind scheme
# Applied ODE & PDE with Python, Ch. 11 | (c) 2026 Stephen E. Moore | MIT Licence
import numpy as np
import matplotlib.pyplot as plt

c, N = 1.0, 100
h = 1.0 / N
x = np.arange(N) * h                            # periodic grid on [0, 1)
u0 = lambda x: np.exp(-200 * (x - 0.3)**2)


def exact(x, t):
    return u0((x - c * t) % 1.0)                # periodic translation


def ftcs(U, nu):                                # forward time, centred space
    return U - 0.5 * nu * (np.roll(U, -1) - np.roll(U, 1))


def upwind(U, nu):                              # backward difference, c > 0
    return U - nu * (U - np.roll(U, 1))


def run(step, nu, T):
    dt = nu * h / c; nsteps = int(round(T / dt))
    U = u0(x)
    for n in range(nsteps):
        U = step(U, nu)
    return U, nsteps * dt


print("scheme    nu     T    max|U|      max error")
for step, nu in ((ftcs, 0.5), (upwind, 0.5), (upwind, 1.0), (upwind, 1.1)):
    for T in (0.5, 1.0, 2.0):
        U, t = run(step, nu, T)
        print(f"{step.__name__:7s}  {nu:4.1f}  {t:4.1f}  {np.max(abs(U)):.3e}"
              f"   {np.max(abs(U - exact(x, t))):.3e}")

fig, ax = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
U, t = run(ftcs, 0.5, 1.0)
ax[0].plot(x, exact(x, t), "k-", label="exact"); ax[0].plot(x, U, ".-", label="FTCS")
ax[0].set_title("FTCS, nu = 0.5, t = 1"); ax[0].set_ylim(-1.2, 1.6)
for nu in (0.5, 1.0):
    U, t = run(upwind, nu, 1.0)
    ax[1].plot(x, U, ".-", label=f"upwind, nu = {nu}")
ax[1].plot(x, exact(x, 1.0), "k-", label="exact"); ax[1].set_title("upwind, t = 1")
for a in ax:
    a.set_xlabel("x"); a.legend(fontsize=8)
plt.tight_layout(); plt.savefig("ch11_ftcs_upwind.pdf", bbox_inches="tight")
plt.close()
