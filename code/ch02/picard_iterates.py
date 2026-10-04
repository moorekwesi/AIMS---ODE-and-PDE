# Picard iterates for y' = 1 + y^2, y(0) = 0 (exact solution tan t)
# Applied ODE & PDE with Python, Ch. 2 | (c) 2026 Stephen E. Moore | MIT Licence
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

t, s = sp.symbols("t s", real=True)


def picard(f, t0, y0, n):
    """Return the Picard iterates phi_0, ..., phi_n for y' = f(t, y), y(t0) = y0."""
    phis = [sp.Integer(y0)]
    for _ in range(n):
        integrand = f(s, phis[-1].subs(t, s))
        phis.append(sp.expand(y0 + sp.integrate(integrand, (s, t0, t))))
    return phis


phis = picard(lambda tt, yy: 1 + yy**2, 0, 0, 4)
for k in range(4):
    print(f"phi_{k} =", phis[k])
print("phi_4 has degree", sp.degree(phis[4], t))
print("Taylor of tan t:", sp.series(sp.tan(t), t, 0, 10).removeO())

# errors at t = 0.5 and t = 1
for tv in (0.5, 1.0):
    errs = [abs(float(p.subs(t, tv)) - np.tan(tv)) for p in phis]
    print(f"t = {tv}: |phi_k - tan t| =", " ".join(f"{e:.2e}" for e in errs))

# plot the iterates against tan t
tt = np.linspace(0, 1.45, 300)
plt.figure(figsize=(7, 4))
plt.plot(tt, np.tan(tt), "k", lw=2.5, label="tan t (exact)")
for k in range(1, 5):
    fk = sp.lambdify(t, phis[k], "numpy")
    plt.plot(tt, fk(tt), "--", label=f"phi_{k}")
plt.axvline(np.pi / 2, color="0.6", ls=":")
plt.ylim(0, 8)
plt.xlabel("t")
plt.ylabel("y")
plt.legend()
plt.tight_layout()
plt.savefig("ch02_picard_iterates.pdf", bbox_inches="tight")
plt.close()
