# Free fall with and without air resistance
# Applied ODE & PDE with Python, Ch. 1 | (c) 2026 Stephen E. Moore | MIT Licence
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

g, m = 9.81, 80.0            # gravity (m/s^2), mass of the skydiver (kg)
b, c = 15.0, 0.25            # linear (kg/s) and quadratic (kg/m) drag coefficients

def rhs_quadratic(t, v):
    """m v' = m g - c v^2  (v > 0 downwards)."""
    return g - (c / m) * v**2

vT = np.sqrt(m * g / c)                              # terminal speed, quadratic drag
exact = lambda t: vT * np.tanh(g * t / vT)
t = np.linspace(0, 30, 301)
sol = solve_ivp(rhs_quadratic, (0, 30), [0.0], t_eval=t, rtol=1e-10, atol=1e-10)

print(f"terminal speed sqrt(m g / c) = {vT:.3f} m/s")
print("   t     no drag   linear   quadratic   exact")
for tk in (0, 2, 5, 10, 20, 30):
    i = np.argmin(abs(t - tk))
    lin = (m * g / b) * (1 - np.exp(-b * tk / m))
    print(f"{tk:4d}  {g*tk:9.2f}  {lin:7.2f}  {sol.y[0, i]:9.4f}  {exact(tk):8.4f}")
print(f"max |numeric - exact| = {np.max(abs(sol.y[0] - exact(t))):.2e}")

plt.figure(figsize=(7, 4))
plt.plot(t, g * t, ":", label="no air resistance")
plt.plot(t, (m * g / b) * (1 - np.exp(-b * t / m)), "--", label="linear drag")
plt.plot(t, sol.y[0], "-", label="quadratic drag")
plt.axhline(vT, color="gray", lw=0.8)
plt.ylim(0, 100); plt.xlabel("t (s)"); plt.ylabel("v (m/s)")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig("ch01_free_fall.pdf", bbox_inches="tight")
plt.close()
