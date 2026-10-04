# Round-off versus truncation error in numerical differentiation
# Applied ODE & PDE with Python, Ch. 9 | (c) 2026 Stephen E. Moore | MIT Licence
import numpy as np
import matplotlib.pyplot as plt

f, x0 = np.exp, 1.0                 # f' = f'' = f''' = exp, so all M's are e
exact = np.exp(x0)
eps = np.finfo(float).eps           # machine epsilon, about 2.2e-16

hs = 10.0 ** np.linspace(-16, 0, 161)
fwd = np.abs((f(x0 + hs) - f(x0)) / hs - exact)
cen = np.abs((f(x0 + hs) - f(x0 - hs)) / (2 * hs) - exact)

# error models  E(h) = truncation + round-off  and their minimisers
model_f = exact * hs / 2 + 2 * eps * exact / hs
model_c = exact * hs**2 / 6 + eps * exact / hs
hopt_f = 2 * np.sqrt(eps)
hopt_c = (3 * eps) ** (1 / 3)
print(f"machine epsilon      = {eps:.3e}")
print(f"forward: h_opt ~ {hopt_f:.2e},  best observed h = {hs[fwd.argmin()]:.2e},"
      f"  min error = {fwd.min():.2e}")
print(f"central: h_opt ~ {hopt_c:.2e},  best observed h = {hs[cen.argmin()]:.2e},"
      f"  min error = {cen.min():.2e}")
print(f"{'h':>8} {'forward err':>12} {'central err':>12}")
for k in range(1, 16, 2):
    h = 10.0 ** -k
    ef = abs((f(x0 + h) - f(x0)) / h - exact)
    ec = abs((f(x0 + h) - f(x0 - h)) / (2 * h) - exact)
    print(f"{h:8.0e} {ef:12.3e} {ec:12.3e}")

plt.figure(figsize=(7, 4.2))
plt.loglog(hs, fwd, ".", ms=3, color="C0", label="forward, observed")
plt.loglog(hs, model_f, "-", color="C0", lw=1, label="forward, model")
plt.loglog(hs, cen, ".", ms=3, color="C3", label="central, observed")
plt.loglog(hs, model_c, "-", color="C3", lw=1, label="central, model")
plt.axvline(hopt_f, color="C0", ls=":")
plt.axvline(hopt_c, color="C3", ls=":")
plt.ylim(1e-12, 1e2)
plt.xlabel("step size h")
plt.ylabel("absolute error in f'(1)")
plt.legend(loc="upper center")
plt.tight_layout()
plt.savefig("ch09_roundoff.pdf", bbox_inches="tight")
plt.close()
