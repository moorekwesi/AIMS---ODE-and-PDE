# Free vibrations of a mass-spring-damper: under-, critically and over-damped motion
# Applied ODE & PDE with Python, Ch. 3 | (c) 2026 Stephen E. Moore | MIT Licence
import numpy as np
import matplotlib.pyplot as plt

m, k = 1.0, 4.0                     # mass (kg), spring constant (N/m)
y0, v0 = 1.0, 0.0                   # initial displacement (m) and velocity (m/s)
w0 = np.sqrt(k / m)                 # natural frequency (rad/s)


def free_motion(gamma, t):
    """Exact solution of m y'' + gamma y' + k y = 0, y(0)=y0, y'(0)=v0."""
    r = np.roots([m, gamma, k])
    disc = gamma**2 - 4 * m * k
    if abs(disc) < 1e-12:                         # critical damping: double root
        r0 = -gamma / (2 * m)
        return (y0 + (v0 - r0 * y0) * t) * np.exp(r0 * t), "critical"
    if disc > 0:                                  # overdamped: two real roots
        r1, r2 = r.real
        c2 = (v0 - r1 * y0) / (r2 - r1)
        return (y0 - c2) * np.exp(r1 * t) + c2 * np.exp(r2 * t), "overdamped"
    lam, mu = -gamma / (2 * m), abs(r[0].imag)    # underdamped: lam +- i mu
    kind = "undamped" if gamma == 0 else "underdamped"
    return np.exp(lam * t) * (y0 * np.cos(mu * t)
                              + (v0 - lam * y0) / mu * np.sin(mu * t)), kind


t = np.linspace(0, 8, 801)                # t[200] = 2
print(f"natural frequency w0 = {w0:.4f} rad/s, critical damping gamma_c = "
      f"{2 * np.sqrt(m * k):.4f} kg/s")
plt.figure(figsize=(7, 3.8))
for gamma in (0.0, 1.0, 4.0, 6.0):
    y, kind = free_motion(gamma, t)
    plt.plot(t, y, label=f"gamma = {gamma:g} ({kind})")
    if kind == "underdamped":
        mu = np.sqrt(4 * m * k - gamma**2) / (2 * m)
        print(f"gamma = {gamma:g}: quasi-frequency mu = {mu:.4f} rad/s, quasi-period "
              f"{2 * np.pi / mu:.4f} s, log decrement {gamma * np.pi / (m * mu):.4f}")
    else:
        print(f"gamma = {gamma:g}: {kind}, y(2) = {y[200]:.6f}")
plt.axhline(0, color="k", lw=0.6)
plt.xlabel("t (s)")
plt.ylabel("displacement y (m)")
plt.legend()
plt.tight_layout()
plt.savefig("ch03_damped_vibrations.pdf", bbox_inches="tight")
plt.close()
