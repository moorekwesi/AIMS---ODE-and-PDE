# Starter code: LWR traffic flow at a traffic light with the Godunov scheme
# Applied ODE & PDE with Python, Ch. 13 | (c) 2026 Stephen E. Moore | MIT Licence
import numpy as np
import matplotlib.pyplot as plt

f = lambda rho: rho * (1 - rho)                 # flux (v_max = rho_max = 1)
demand = lambda rho: f(np.minimum(rho, 0.5))    # what can leave a cell
supply = lambda rho: f(np.maximum(rho, 0.5))    # what can enter a cell

L, N = 4.0, 400                                  # road [-2, 2], light at x = 0
h = L / N
x = -2 + h * (np.arange(N) + 0.5)               # cell centres
rho = np.where(x < 0, 0.3, 0.0)                 # initial density
rho_in, T_red, T = 0.3, 1.0, 3.0                # inflow density, red phase, final time
dt = T / np.ceil(T / (0.9 * h))                # CFL: max |f'(rho)| = 1, dt <= 0.9 h
light = int(round(2.0 / h))                     # face k sits at x = -2 + k h

snaps, t, passed = {}, 0.0, 0.0
for n in range(int(round(T / dt))):
    ext = np.concatenate([[rho_in], rho, [rho[-1]]])        # ghost cells
    F = np.minimum(demand(ext[:-1]), supply(ext[1:]))      # Godunov flux at N+1 faces
    if t < T_red:
        F[light] = 0.0                                      # red light: nothing passes
    passed += dt * F[light]
    rho = rho - dt / h * (F[1:] - F[:-1])
    t += dt
    for ts in [0.5, 1.0, 2.0, 3.0]:
        if abs(t - ts) < dt / 2:
            snaps[ts] = rho.copy()

q = snaps[1.0]
print(f"cells: {N}, dt = {dt:.4f}, steps = {int(round(T / dt))}")
print(f"queue at t = 1 (rho > 0.9) occupies x in [{x[q > 0.9].min():.3f}, 0]")
print(f"cars that crossed the light by t = {T}: {passed:.4f} (density units x length)")
print(f"min/max density at t = {T}: {rho.min():.4f} / {rho.max():.4f}")
plt.figure(figsize=(7, 3.6))
for ts, r in snaps.items():
    plt.plot(x, r, label=f"t = {ts}")
plt.axvline(0, color="r", ls=":")
plt.xlabel("x")
plt.ylabel("density rho")
plt.legend()
plt.tight_layout()
plt.savefig("ch13_traffic_light.pdf", bbox_inches="tight")
plt.close()
