# Animating a travelling wave with FuncAnimation
# Applied ODE & PDE with Python, App. A | (c) 2026 Stephen E. Moore | MIT Licence
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

c = 1.0                                         # wave speed
x = np.linspace(0, 10, 400)
u = lambda x, t: np.exp(-4 * (x - 2 - c * t)**2)   # u_t + c u_x = 0

fig, ax = plt.subplots(figsize=(7, 3))
line, = ax.plot(x, u(x, 0.0))
title = ax.set_title("t = 0.00")
ax.set_xlim(0, 10); ax.set_ylim(-0.1, 1.1); ax.set_xlabel("x"); ax.set_ylabel("u")

def update(n):
    """Redraw frame n (time t = 0.05 n) and return the changed artists."""
    tn = 0.05 * n
    line.set_ydata(u(x, tn))
    title.set_text(f"t = {tn:.2f}")
    return line, title

anim = FuncAnimation(fig, update, frames=121, interval=40, blit=False)
# anim.save("wave.gif", writer="pillow", fps=25)   # or "wave.mp4" if ffmpeg is installed
plt.close(fig)

# For a printed book we save a few snapshots in a single static figure instead
plt.figure(figsize=(7, 3))
for tn in (0, 2, 4, 6):
    plt.plot(x, u(x, tn), label=f"t = {tn}")
plt.xlabel("x"); plt.ylabel("u"); plt.legend(); plt.tight_layout()
plt.savefig("ch01_appA_snapshots.pdf", bbox_inches="tight")
plt.close()
print("animation with", 121, "frames created; snapshots saved")
