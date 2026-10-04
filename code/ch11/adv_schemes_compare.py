# Numerical diffusion and dispersion: square pulse and Gaussian after one period
# Applied ODE & PDE with Python, Ch. 11 | (c) 2026 Stephen E. Moore | MIT Licence
import numpy as np
import matplotlib.pyplot as plt

R = lambda U, k: np.roll(U, k)
STEPS = {
    "upwind": lambda U, nu: U - nu * (U - R(U, 1)),
    "Lax-Friedrichs": lambda U, nu: 0.5 * (R(U, -1) + R(U, 1)) - 0.5 * nu * (R(U, -1) - R(U, 1)),
    "Lax-Wendroff": lambda U, nu: (U - 0.5 * nu * (R(U, -1) - R(U, 1))
                                   + 0.5 * nu**2 * (R(U, -1) - 2 * U + R(U, 1))),
    "Beam-Warming": lambda U, nu: (U - 0.5 * nu * (3 * U - 4 * R(U, 1) + R(U, 2))
                                   + 0.5 * nu**2 * (U - 2 * R(U, 1) + R(U, 2))),
}
N, nu, T = 200, 0.8, 1.0
h = 1.0 / N; x = np.arange(N) * h; nsteps = int(round(T / (nu * h)))
data = {"square pulse": lambda x: np.where((x > 0.2) & (x < 0.4), 1.0, 0.0),
        "Gaussian": lambda x: np.exp(-((x - 0.5) / 0.05)**2)}

print(f"N = {N}, nu = {nu}, one period ({nsteps} steps)")
print("data           scheme           L1 error     min U     max U")
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for a, (dname, u0) in zip(ax, data.items()):
    a.plot(x, u0(x), "k-", lw=1.5, label="exact")
    for sname, step in STEPS.items():
        U = u0(x)
        for n in range(nsteps):
            U = step(U, nu)
        l1 = h * np.sum(np.abs(U - u0(x)))     # exact solution = initial data
        print(f"{dname:13s}  {sname:15s}  {l1:.3e}   {U.min():7.4f}   {U.max():7.4f}")
        a.plot(x, U, lw=1, label=sname)
    a.set_title(dname); a.set_xlabel("x"); a.legend(fontsize=8)
ax[0].set_xlim(0.05, 0.6); ax[1].set_xlim(0.25, 0.75)
plt.tight_layout(); plt.savefig("ch11_schemes_compare.pdf", bbox_inches="tight")
plt.close()
