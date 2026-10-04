# Traffic at a red light: the LWR model solved with the Godunov scheme
# Applied ODE & PDE with Python, Ch. 11 | (c) 2026 Stephen E. Moore | MIT Licence
import numpy as np
import matplotlib.pyplot as plt

f = lambda rho: rho * (1 - rho)                 # Greenshields flux, v_max = rho_max = 1
demand = lambda r: np.where(r <= 0.5, f(r), 0.25)
supply = lambda r: np.where(r <= 0.5, 0.25, f(r))
xa, xb, N, T_red, T = -3.0, 3.0, 600, 1.0, 3.0
h = (xb - xa) / N
x = xa + (np.arange(N) + 0.5) * h
light = int(round(-xa / h))                     # index of the face at x = 0


def tail_exact(t):
    """Exact position of the back of the queue (the shock)."""
    tau = t - T_red
    return -0.4 * t if tau <= 2 / 3 else 0.2 * tau - 0.8 * np.sqrt(1.5 * tau)


rho = np.where(x < 0, 0.4, 0.0)
dt = 0.9 * h; nsteps = int(round(T / dt)); dt = T / nsteps
history, report = [rho.copy()], (0.5, 1.0, 5 / 3, 2.0, 3.0)
print("   t     back of queue (Godunov)   exact")
for n in range(1, nsteps + 1):
    re = np.r_[0.4, rho, rho[-1]]               # inflow 0.4 on the left, free outflow
    F = np.minimum(demand(re[:-1]), supply(re[1:]))   # Godunov flux = min(D, S)
    if n * dt <= T_red + 1e-12:
        F[light] = 0.0                          # red light: no flux through x = 0
    rho = rho - dt / h * (F[1:] - F[:-1])
    history.append(rho.copy())
    t = n * dt
    for tr in report:
        if abs(t - tr) < dt / 2:
            j = np.argmax(rho[: light] > 0.4 + 0.02)     # first cell above the inflow
            print(f"{t:6.3f}      {x[j] - h / 2:8.3f}             {tail_exact(t):8.3f}")
print(f"cars on the road at t = 0 and t = {T}: {h * history[0].sum():.3f}, "
      f"{h * rho.sum():.3f}  (inflow adds 0.24 per unit time: {1.2 + 0.24 * T:.3f})")

fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
H = np.array(history)
im = ax[0].imshow(H, origin="lower", aspect="auto", extent=[xa, xb, 0, T], cmap="viridis")
ts = np.linspace(0, T, 200)
ax[0].plot([tail_exact(t) for t in ts], ts, "w--", lw=1)
ax[0].set_xlabel("x"); ax[0].set_ylabel("t"); plt.colorbar(im, ax=ax[0], label="density")
for t in (0.5, 1.0, 2.0, 3.0):
    ax[1].plot(x, H[int(round(t / dt))], label=f"t = {t}")
ax[1].set_xlabel("x"); ax[1].set_ylabel("density"); ax[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig("ch11_traffic_red_light.pdf", bbox_inches="tight")
plt.close()
