# Half-line problem with inflow data
# Applied ODE & PDE with Python, Ch. 6 | (c) 2026 Stephen E. Moore | MIT Licence
import numpy as np
import matplotlib.pyplot as plt

c = 1.0                                      # speed (to the right)


def solve(f, g, x, t):
    """u_t + c u_x = 0 on x > 0, u(x,0) = f(x), u(0,t) = g(t)."""
    return np.where(x >= c*t, f(x - c*t), g(t - x/c))


cases = {
    "C1-compatible": (lambda x: np.exp(-x**2), lambda t: np.exp(-t**2)),
    "kink":          (lambda x: np.exp(-x),    lambda t: np.cos(2*t)),
    "jump":          (lambda x: np.exp(-x),    lambda t: 0.5 + 0*t),
}
t1, eps = 1.0, 1e-6                           # look across x = c t1
print("case             u(ct-,t)   u(ct+,t)   slope left  slope right")
for name, (f, g) in cases.items():
    xl, xr = c*t1 - eps, c*t1 + eps
    ul, ur = solve(f, g, xl, t1), solve(f, g, xr, t1)
    sl = (ul - solve(f, g, xl - eps, t1))/eps   # one-sided slopes
    sr = (solve(f, g, xr + eps, t1) - ur)/eps
    sl, sr = round(sl, 4) + 0.0, round(sr, 4) + 0.0   # avoid printing -0.0000
    print(f"{name:15s} {ul:9.5f}  {ur:9.5f}  {sl:10.4f}  {sr:10.4f}")

# signalling problem: quiet river, periodic release at x = 0
f0, gs = (lambda x: 0*x), (lambda t: np.sin(np.pi*t)**2*(t > 0))
x = np.linspace(0, 6, 601)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
f, g = cases["jump"]
for t in [0, 1, 2, 3]:
    a1.plot(x, solve(f, g, x, t), label=f"t = {t}")
a1.set_xlabel("x"); a1.set_ylabel("u"); a1.legend()
a1.set_title("Incompatible data: a jump travels along x = ct")
T = np.linspace(0, 4, 401)
XX, TT = np.meshgrid(x, T)
a2.contourf(XX, TT, solve(f0, gs, XX, TT), levels=20, cmap="viridis")
a2.plot(c*T, T, "w--", lw=1)
a2.set_xlabel("x"); a2.set_ylabel("t"); a2.set_title("Signalling problem")
plt.tight_layout()
plt.savefig("ch06_half_line.pdf", bbox_inches="tight")
