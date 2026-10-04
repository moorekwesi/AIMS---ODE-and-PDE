# Bifurcation diagram of y' = r y (1 - y/K) - H with the harvest H as parameter
# Applied ODE & PDE with Python, Ch. 2 | (c) 2026 Stephen E. Moore | MIT Licence
import numpy as np
import matplotlib.pyplot as plt

r, K = 0.5, 1000.0
Hc = r * K / 4                                   # saddle-node bifurcation value

H = np.linspace(0, Hc, 400)
disc = np.sqrt(1 - 4 * H / (r * K))
y_plus = 0.5 * K * (1 + disc)                    # stable branch
y_minus = 0.5 * K * (1 - disc)                   # unstable branch

for h in (0, 50, 100, 120, 124, 125):
    d = np.sqrt(max(1 - 4 * h / (r * K), 0.0))
    print(f"H = {h:5.1f}: y_minus = {0.5*K*(1-d):8.3f}, y_plus = {0.5*K*(1+d):8.3f},"
          f" gap = {K*d:8.3f}")

plt.figure(figsize=(7, 4))
plt.plot(H, y_plus, "C0", lw=2.5, label="stable equilibrium")
plt.plot(H, y_minus, "C3--", lw=2, label="unstable equilibrium")
plt.plot([Hc], [K / 2], "ko")
plt.annotate("saddle-node at H = rK/4", xy=(Hc, K / 2), xytext=(60, 420),
             arrowprops=dict(arrowstyle="->"))
# arrows: direction of motion for a few values of H
for h in (40, 90, 140):
    for y0 in np.linspace(50, 950, 7):
        fy = r * y0 * (1 - y0 / K) - h
        plt.annotate("", xy=(h, y0 + 40 * np.sign(fy)), xytext=(h, y0),
                     arrowprops=dict(arrowstyle="->", color="0.5"))
plt.xlabel("harvest rate H (tonnes/year)")
plt.ylabel("equilibrium stock y*")
plt.xlim(0, 150)
plt.legend(loc="upper right")
plt.tight_layout()
plt.savefig("ch02_harvest_bifurcation.pdf", bbox_inches="tight")
plt.close()
