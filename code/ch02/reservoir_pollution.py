# A pollutant in a well-mixed reservoir: discharge, then recovery
# Applied ODE & PDE with Python, Ch. 2 | (c) 2026 Stephen E. Moore | MIT Licence
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

V = 5.0e7        # volume of the reservoir (m^3)
Q = 1.0e6        # inflow = outflow (m^3/day)
k = 0.01         # first-order decay rate of the pollutant (1/day)
c_in0 = 20.0     # concentration in the polluted inflow (g/m^3 = mg/L)
T_stop = 100.0   # the discharge stops after T_stop days


def c_in(t):
    return c_in0 if t < T_stop else 0.0


def rhs(t, c):
    """dc/dt = (Q/V)(c_in - c) - k c."""
    return (Q / V) * (c_in(t) - c) - k * c


a = Q / V + k                         # total removal rate (1/day)
c_star = (Q / V) * c_in0 / a          # equilibrium during discharge


def exact(t):
    """Piecewise exact solution with c(0) = 0."""
    t = np.asarray(t, dtype=float)
    c1 = c_star * (1 - np.exp(-a * t))
    cT = c_star * (1 - np.exp(-a * T_stop))
    return np.where(t < T_stop, c1, cT * np.exp(-a * (t - T_stop)))


# integrate in two pieces so that the solver never steps across the jump
s1 = solve_ivp(rhs, (0, T_stop), [0.0], rtol=1e-9, atol=1e-12, dense_output=True)
s2 = solve_ivp(rhs, (T_stop, 300), [s1.y[0, -1]], rtol=1e-9, atol=1e-12,
               dense_output=True)
print(f"residence time V/Q        = {V / Q:.1f} days")
print(f"equilibrium c*            = {c_star:.4f} mg/L")
print(f"c(T_stop)  exact / solver = {exact(T_stop):.6f} / {s1.y[0, -1]:.6f}")
t_safe = T_stop + np.log(exact(T_stop) / 1.0) / a
print(f"c falls below 1 mg/L at t = {t_safe:.2f} days")
print(f"solver value there        = {s2.sol(t_safe)[0]:.6f}")

tt = np.linspace(0, 300, 601)
plt.figure(figsize=(7, 3.8))
plt.plot(tt, exact(tt), "C0", lw=2, label="c(t)")
plt.axhline(c_star, color="C1", ls="--", label="equilibrium c*")
plt.axhline(1.0, color="C3", ls=":", label="1 mg/L")
plt.axvline(T_stop, color="0.6", lw=0.8)
plt.xlabel("t (days)")
plt.ylabel("concentration (mg/L)")
plt.legend()
plt.tight_layout()
plt.savefig("ch02_reservoir_pollution.pdf", bbox_inches="tight")
plt.close()
