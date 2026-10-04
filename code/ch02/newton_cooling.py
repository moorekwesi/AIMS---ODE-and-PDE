# Newton's law of cooling: time of death, and cocoa beans in a daily temperature cycle
# Applied ODE & PDE with Python, Ch. 2 | (c) 2026 Stephen E. Moore | MIT Licence
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# (a) Forensics: room at 22 C; body 30.0 C at 22:00 and 28.5 C at 23:00
Ta, T1, T2, Tbody = 22.0, 30.0, 28.5, 37.0
k = np.log((T1 - Ta) / (T2 - Ta))                  # per hour
t_dead = np.log((Tbody - Ta) / (T1 - Ta)) / k      # hours before 22:00
print(f"(a) k = {k:.4f} per hour, death {t_dead:.2f} h before 22:00")
hh = 22 - t_dead
print(f"    estimated time of death ~ {int(hh):02d}:{int(round(60*(hh % 1))):02d}")

# (b) Cocoa beans at 45 C spread out at 06:00; air temperature
#     Ta(t) = 27 + 5 sin(w (t - 9)),  w = 2 pi / 24, t in hours after midnight
kc, A, w = 0.5, 5.0, 2 * np.pi / 24


def air(t):
    return 27 + A * np.sin(w * (t - 9))


sol = solve_ivp(lambda t, T: -kc * (T - air(t)), (6, 54), [45.0],
                dense_output=True, rtol=1e-8, atol=1e-10)
amp = A * kc / np.sqrt(kc**2 + w**2)               # steady-state amplitude
lag = np.arctan(w / kc) / w                        # time lag in hours
print(f"(b) steady oscillation: amplitude {amp:.3f} C, lag {lag:.3f} h")
for th in (7, 8, 10, 15, 39):
    print(f"    t = {th:2d} h: beans {sol.sol(th)[0]:7.3f} C, air {air(th):7.3f} C")

tt = np.linspace(6, 54, 500)
plt.figure(figsize=(7, 3.8))
plt.plot(tt, sol.sol(tt)[0], "C3", lw=2, label="beans T(t)")
plt.plot(tt, air(tt), "C0--", label="air Ta(t)")
plt.xlabel("t (hours after midnight, day 1)")
plt.ylabel("temperature (C)")
plt.legend()
plt.tight_layout()
plt.savefig("ch02_cocoa_cooling.pdf", bbox_inches="tight")
plt.close()
