# d'Alembert's formula: plucked and struck infinite strings
# Applied ODE & PDE with Python, Ch. 8 | (c) 2026 Stephen E. Moore | MIT Licence
import numpy as np
import matplotlib.pyplot as plt

c = 1.0
f = lambda x: np.maximum(0.0, 1 - np.abs(x))         # initial displacement
G = lambda x: np.clip(x, -1, 1)                      # antiderivative of g = 1 on [-1,1]


def plucked(x, t):
    """u_tt = c^2 u_xx, u(x,0) = f(x), u_t(x,0) = 0."""
    return 0.5 * (f(x - c * t) + f(x + c * t))


def struck(x, t):
    """u(x,0) = 0, u_t(x,0) = g(x):  (1/2c) * int_{x-ct}^{x+ct} g."""
    return (G(x + c * t) - G(x - c * t)) / (2 * c)


x = np.linspace(-6, 6, 12001)
print("   t   support of plucked   support of struck   predicted 1+ct   u_struck(0,t)")
for t in [0.5, 1.0, 2.0, 4.0]:
    sp = x[plucked(x, t) > 1e-12]
    ss = x[struck(x, t) > 1e-12]
    print(f"{t:4.1f}   [{sp[0]:5.2f}, {sp[-1]:5.2f}]      [{ss[0]:5.2f}, {ss[-1]:5.2f}]"
          f"        {1 + c * t:4.1f}          {struck(0.0, t):.4f}")

fig, ax = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
for t in [0, 0.5, 1.5, 3.0]:
    ax[0].plot(x, plucked(x, t), label=f"$t={t}$")
    ax[1].plot(x, struck(x, t), label=f"$t={t}$")
ax[0].set_title("plucked: $f$ = tent, $g=0$", fontsize=10)
ax[1].set_title("struck: $f=0$, $g=\\chi_{[-1,1]}$", fontsize=10)
for a in ax:
    a.set_xlabel("$x$"); a.set_xlim(-5, 5); a.legend(fontsize=8)
ax[0].set_ylabel("$u(x,t)$")
plt.tight_layout()
plt.savefig("ch08_dalembert.pdf", bbox_inches="tight")
