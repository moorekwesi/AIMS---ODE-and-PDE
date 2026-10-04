# The McKendrick-von Foerster equation
# Applied ODE & PDE with Python, Ch. 6 | (c) 2026 Stephen E. Moore | MIT Licence
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

amax, da = 100.0, 0.25                    # ages 0..100 years; dt = da
a = np.arange(0, amax + da/2, da)
mu = lambda a: 0.005 + 0.0002*np.exp(0.08*a)          # mortality (1/year)
beta = lambda a: 0.15*np.exp(-((a - 28)/7)**2)        # fertility (1/year)
cum_mu = lambda a: 0.005*a + 0.0002/0.08*(np.exp(0.08*a) - 1)
S = np.exp(-(cum_mu(a + da) - cum_mu(a)))   # survival over one step, exact
w = np.full(a.size, da); w[[0, -1]] = da/2  # trapezoidal weights

u = 2.0*np.exp(-0.03*a)                     # initial age density
N, times, snaps = [np.sum(w*u)], [0.0], {0: u/np.sum(w*u)}
for n in range(1, 801):                     # 800 steps = 200 years
    u_new = np.empty_like(u)
    u_new[1:] = u[:-1]*S[:-1]               # every cohort ages by da
    u_new[0] = np.sum(w[1:]*beta(a[1:])*u_new[1:])   # births (beta(0) ~ 0)
    u = u_new
    times.append(n*da); N.append(np.sum(w*u))
    if n*da in (20, 50, 200):
        snaps[int(n*da)] = u/N[-1]
N = np.array(N)

# Lotka's characteristic equation  int beta(a) pi(a) exp(-r a) da = 1
pi_a = np.exp(-cum_mu(a))
lotka = lambda r: np.sum(w*beta(a)*pi_a*np.exp(-r*a)) - 1
r = brentq(lotka, -0.1, 0.2)
R0 = np.sum(w*beta(a)*pi_a)
r_est = np.log(N[800]/N[600])/50
print(f"net reproduction number R0 = {R0:.4f}")
print(f"Lotka growth rate r        = {r:.5f} per year")
print(f"growth rate, t in [150,200] = {r_est:.5f} per year")
for tt in [0, 25, 50, 100, 150, 200]:
    print(f"t = {tt:3d}  N = {N[int(tt/da)]:10.2f}")
stable = pi_a*np.exp(-r*a); stable /= np.sum(w*stable)
print(f"max |u/N - stable| at t = 200: {np.max(abs(snaps[200]-stable)):.2e}")

fig, ax = plt.subplots(figsize=(7, 3.8))
for key, prof in snaps.items():
    ax.plot(a, prof, label=f"t = {key}")
ax.plot(a, stable, "k--", label="stable age distribution")
ax.set_xlabel("age a (years)"); ax.set_ylabel("u(a,t)/N(t)"); ax.legend()
plt.tight_layout()
plt.savefig("ch06_mckendrick.pdf", bbox_inches="tight")
