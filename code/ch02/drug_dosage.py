# Repeated drug doses in a one-compartment model: peaks, troughs and steady state
# Applied ODE & PDE with Python, Ch. 2 | (c) 2026 Stephen E. Moore | MIT Licence
import numpy as np
import matplotlib.pyplot as plt

t_half = 6.0               # elimination half-life (hours)
k = np.log(2) / t_half     # elimination rate constant (1/hour)
T = 8.0                    # dosing interval (hours)
c_dose = 10.0              # concentration jump produced by one dose, D/V (mg/L)
n_doses = 8
q = np.exp(-k * T)         # fraction left after one interval

print(f"k = {k:.5f} 1/h,  q = exp(-kT) = {q:.5f}")
print(" n   peak (mg/L)  trough (mg/L)")
peak = 0.0
for n in range(1, n_doses + 1):
    trough_before = peak * q if n > 1 else 0.0
    peak = trough_before + c_dose            # jump at the n-th dose
    print(f"{n:2d}   {peak:10.4f}   {peak * q:10.4f}")
c_max = c_dose / (1 - q)
print(f"steady state: peak {c_max:.4f}, trough {c_max * q:.4f} mg/L")
print(f"loading dose for immediate steady state: {c_max / c_dose:.3f} x normal dose")


def concentration(t):
    """c(t) = sum over doses given so far of c_dose * exp(-k (t - t_j))."""
    t = np.asarray(t, dtype=float)
    c = np.zeros_like(t)
    for j in range(n_doses):
        tj = j * T
        c += np.where(t >= tj, c_dose * np.exp(-k * (t - tj)), 0.0)
    return c


tt = np.linspace(0, n_doses * T, 1200, endpoint=False)
plt.figure(figsize=(7, 3.6))
plt.plot(tt, concentration(tt), "C0", lw=1.8, label="c(t)")
plt.axhline(c_max, color="C3", ls="--", label="steady-state peak")
plt.axhline(c_max * q, color="C2", ls="--", label="steady-state trough")
plt.xlabel("t (hours)")
plt.ylabel("concentration (mg/L)")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("ch02_drug_dosage.pdf", bbox_inches="tight")
plt.close()
