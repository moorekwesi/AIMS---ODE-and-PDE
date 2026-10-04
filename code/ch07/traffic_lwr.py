# Traffic flow with the Lighthill-Whitham-Richards model (Greenshields flux)
# Applied ODE & PDE with Python, Ch. 7 | (c) 2026 Stephen E. Moore | MIT Licence
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

vmax, rmax = 20.0, 0.15            # 20 m/s = 72 km/h, 0.15 cars per metre (one lane)
v = lambda r: vmax * (1 - r / rmax)          # Greenshields velocity-density law
f = lambda r: r * v(r)                       # flux (cars per second)


def green_light(x, t):
    """Rarefaction: queue rho = rmax for x < 0 released at t = 0."""
    return np.clip(0.5 * rmax * (1 - x / (vmax * t)), 0.0, rmax)


def car_rhs(t, x):
    """A car moves with the local traffic speed v(rho(x, t))."""
    return [v(green_light(x[0], t))]


def at_light(t, x):
    """Event: the car passes the traffic light at x = 0."""
    return x[0]


at_light.terminal = True


print("--- red light turns green ---")
print("capacity f(rmax/2) =", f"{f(rmax / 2):.3f} cars/s =", f"{3600 * f(rmax / 2):.0f} cars/h")
print("  x0 (m)   starts (s)   crosses light: formula (s)  numerical (s)")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
for x0 in [-20.0, -50.0, -100.0, -200.0]:
    t0 = -x0 / vmax                               # the fan edge x = -vmax t arrives
    cross = solve_ivp(car_rhs, (t0, 10 * t0), [x0], max_step=0.05, rtol=1e-9,
                      events=at_light)
    print(f"{x0:8.0f}  {t0:10.2f}  {4 * t0:22.2f}  {cross.t_events[0][0]:14.4f}")
    tt = np.linspace(0, 60, 400)
    xx = np.where(tt < t0, x0, vmax * tt - 2 * np.sqrt(np.maximum(-x0 * vmax * tt, 0)))
    ax2.plot(xx, tt, "k-", lw=1.2)
for c in np.linspace(-vmax, vmax, 11):            # fan characteristics x = c t
    ax2.plot(c * np.array([0, 60]), [0, 60], "g-", lw=0.6)
x = np.linspace(-400, 400, 801)
for t in [1, 5, 10, 20]:
    ax1.plot(x, green_light(x, t), label=f"t = {t} s")
ax1.set_xlabel("x (m)"); ax1.set_ylabel("density (cars/m)"); ax1.legend()
ax2.set_xlim(-250, 400); ax2.set_ylim(0, 60)
ax2.set_xlabel("x (m)"); ax2.set_ylabel("t (s)")
ax2.set_title("car paths (black) and fan characteristics (green)")
plt.tight_layout()
plt.savefig("ch07_traffic_green.pdf", bbox_inches="tight")

print("--- arriving at a stationary queue ---")
rl, rr = 0.25 * rmax, rmax                        # light traffic meets a jam
s = (f(rl) - f(rr)) / (rl - rr)                   # Rankine-Hugoniot speed
print(f"f'(rho_l) = {vmax * (1 - 2 * rl / rmax):.1f} m/s, f'(rho_r) = "
      f"{vmax * (1 - 2 * rr / rmax):.1f} m/s, shock speed s = {s:.1f} m/s")
print(f"car speed before the shock v(rho_l) = {v(rl):.1f} m/s")
fig, ax = plt.subplots(figsize=(7, 4))
t = np.linspace(0, 60, 400)
ax.plot(s * t, t, "r-", lw=2.5, label="shock (back of the queue)")
for x0 in np.arange(-1200.0, 0.0, 1 / rl * 4):    # every 4th car of the inflow
    tstop = x0 / (s - v(rl))                      # time at which the car hits the shock
    ax.plot(np.where(t < tstop, x0 + v(rl) * t, x0 + v(rl) * tstop), t, "k-", lw=0.8)
for x0 in np.arange(0.0, 300.0, 1 / rr * 4):      # cars standing in the queue
    ax.plot([x0, x0], [0, 60], "k-", lw=0.8)
ax.set_xlim(-1200, 300); ax.set_ylim(0, 60)
ax.set_xlabel("x (m)"); ax.set_ylabel("t (s)"); ax.legend(loc="upper right")
plt.tight_layout()
plt.savefig("ch07_traffic_jam.pdf", bbox_inches="tight")
