# Variable-coefficient characteristics
# Applied ODE & PDE with Python, Ch. 6 | (c) 2026 Stephen E. Moore | MIT Licence
import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 3, figsize=(11, 3.6))
# (a) u_x + y u_y = 0: dy/dx = y, so y = C e^x
xs = np.linspace(-2, 2, 200)
for C in np.linspace(-2, 2, 17):
    ax[0].plot(xs, C*np.exp(xs), "b", lw=0.8)
ax[0].set_ylim(-3, 3); ax[0].set_title(r"$u_x + y u_y = 0$:  $y = Ce^x$")
ax[0].set_xlabel("x"); ax[0].set_ylabel("y")

# (b) u_t + (1 + x^2) u_x = 0: dx/dt = 1 + x^2, so x = tan(t + C)
for x0 in np.linspace(-6, 6, 25):
    C = np.arctan(x0)
    tt = np.linspace(0, np.pi/2 - C - 1e-3, 400)        # until blow-up
    ax[1].plot(np.tan(tt + C), tt, "r", lw=0.8)
tt = np.linspace(1e-3, 1.5, 200)
ax[1].fill_betweenx(tt, -6, -1/np.tan(tt), color="0.85")  # x < -cot t
ax[1].set_xlim(-6, 6); ax[1].set_ylim(0, 1.5)
ax[1].set_title(r"$u_t + (1+x^2) u_x = 0$:  $x = \tan(t + C)$")
ax[1].set_xlabel("x"); ax[1].set_ylabel("t")

# (c) u_t + x u_x = 0: dx/dt = x, so x = C e^t
T = np.linspace(0, 1.5, 200)
for C in np.linspace(-2, 2, 17):
    ax[2].plot(C*np.exp(T), T, "g", lw=0.8)
ax[2].set_xlim(-4, 4); ax[2].set_title(r"$u_t + x u_x = 0$:  $x = Ce^t$")
ax[2].set_xlabel("x"); ax[2].set_ylabel("t")
plt.tight_layout()
plt.savefig("ch06_variable_chars.pdf", bbox_inches="tight")


# the exam problem: u = (cos t + x sin t)^2/(1 + x^2) is constant along x = tan(t + C)
def u_exam(x, t):
    return (np.cos(t) + x*np.sin(t))**2/(1 + x**2)


print("  x0    blow-up time   u along the characteristic at t=0, 0.2, 0.4")
for x0 in [-2.0, 0.0, 0.5, 2.0]:
    C = np.arctan(x0)
    vals = [u_exam(np.tan(t + C), t) for t in (0.0, 0.2, 0.4)]
    print(f"{x0:5.1f}   {np.pi/2 - C:9.4f}     " +
          "  ".join(f"{v:.6f}" for v in vals))
