# Forced oscillations: beats, pure resonance and the amplitude response curve
# Applied ODE & PDE with Python, Ch. 3 | (c) 2026 Stephen E. Moore | MIT Licence
import numpy as np
import matplotlib.pyplot as plt

w0, F0 = 2.0, 1.0                       # natural frequency, force amplitude (per unit mass)
t = np.linspace(0, 60, 3000)

# (a) beats: y'' + w0^2 y = F0 cos(w t), y(0) = y'(0) = 0, w close to w0
w = 1.8
y_beats = F0 / (w0**2 - w**2) * (np.cos(w * t) - np.cos(w0 * t))
envelope = 2 * F0 / abs(w0**2 - w**2) * np.abs(np.sin(0.5 * (w0 - w) * t))
print(f"beats: carrier frequency {(w0 + w) / 2:.2f}, beat period "
      f"{2 * np.pi / abs(w0 - w):.4f}, max amplitude {2 * F0 / abs(w0**2 - w**2):.4f}")

# (b) pure resonance w = w0: y = F0 t sin(w0 t) / (2 w0)
y_res = F0 * t * np.sin(w0 * t) / (2 * w0)

# (c) steady-state amplitude with damping: y'' + g y' + w0^2 y = F0 cos(w t)
W = np.linspace(0.01, 4, 800)
fig, axs = plt.subplots(1, 3, figsize=(11, 3.4))
axs[0].plot(t, y_beats, lw=0.8)
axs[0].plot(t, envelope, "k--", lw=0.8)
axs[0].plot(t, -envelope, "k--", lw=0.8)
axs[0].set_title("beats (w = 1.8, w0 = 2)")
axs[1].plot(t, y_res, lw=0.8)
axs[1].plot(t, t / (2 * w0), "k--", lw=0.8)
axs[1].plot(t, -t / (2 * w0), "k--", lw=0.8)
axs[1].set_title("resonance (w = w0 = 2)")
for g in (0.1, 0.5, 1.0, 2.0):
    A = F0 / np.sqrt((w0**2 - W**2) ** 2 + (g * W) ** 2)
    axs[2].plot(W, A, label=f"gamma = {g}")
    if g**2 < 2 * w0**2:
        wmax = np.sqrt(w0**2 - g**2 / 2)
        Amax = F0 / (g * np.sqrt(w0**2 - g**2 / 4))
        print(f"gamma = {g}: peak at w = {wmax:.4f}, amplitude {Amax:.4f}, "
              f"gain over static {Amax * w0**2 / F0:.3f}")
axs[2].set_title("amplitude response A(w)")
axs[2].set_ylim(0, 5.5)
axs[2].legend(fontsize=8)
for ax, xl in zip(axs, ("t", "t", "forcing frequency w")):
    ax.set_xlabel(xl)
plt.tight_layout()
plt.savefig("ch03_beats_resonance.pdf", bbox_inches="tight")
plt.close()
