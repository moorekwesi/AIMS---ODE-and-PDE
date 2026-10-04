# Exact entropy solution of the Riemann problem for the inviscid Burgers equation
# Applied ODE & PDE with Python, Ch. 7 | (c) 2026 Stephen E. Moore | MIT Licence
import numpy as np
import matplotlib.pyplot as plt


def riemann_burgers(ul, ur, x, t):
    """Entropy solution of u_t + (u^2/2)_x = 0 with u = ul (x<0), ur (x>0).

    ul > ur: shock moving with the Rankine-Hugoniot speed s = (ul + ur)/2.
    ul < ur: rarefaction fan u = x/t for ul < x/t < ur.
    """
    x = np.asarray(x, dtype=float)
    if ul > ur:
        s = 0.5 * (ul + ur)
        return np.where(x < s * t, ul, ur)
    return np.clip(x / t, ul, ur)


# a few values, compared with what the formulas predict
xs = np.array([-0.6, -0.2, 0.2, 0.45, 0.55, 0.9])
print("                         x =  " + "  ".join(f"{v:6.2f}" for v in xs))
for ul, ur in [(1.0, 0.0), (-0.5, 1.0)]:
    vals = "  ".join(f"{v:6.3f}" for v in riemann_burgers(ul, ur, xs, 1.0))
    print(f"ul = {ul:5.2f}, ur = {ur:5.2f}, t = 1:  u = {vals}")

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, (ul, ur) in zip(axes, [(1.0, 0.0), (-0.5, 1.0)]):
    t = np.linspace(0, 2, 200)
    for x0 in np.linspace(-2, 2, 21):            # characteristics from t = 0
        speed = ul if x0 < 0 else ur
        xc = x0 + speed * t
        if ul > ur:                              # stop at the shock x = s t
            keep = (xc <= 0.5 * (ul + ur) * t) if x0 < 0 else (xc >= 0.5 * (ul + ur) * t)
            xc, tc = xc[keep], t[keep]
        else:
            tc = t
        ax.plot(xc, tc, "b-", lw=0.8)
    if ul > ur:
        ax.plot(0.5 * (ul + ur) * t, t, "r-", lw=2.5, label="shock x = s t")
    else:
        for c in np.linspace(ul, ur, 9):         # fan: straight lines x = c t
            ax.plot(c * t, t, "g-", lw=0.8)
        ax.plot([], [], "g-", label="fan x = c t")
    ax.set_xlim(-2, 2); ax.set_ylim(0, 2)
    ax.set_xlabel("x"); ax.set_ylabel("t"); ax.legend(loc="upper left")
    ax.set_title(f"u_l = {ul}, u_r = {ur}")
plt.tight_layout()
plt.savefig("ch07_riemann_burgers.pdf", bbox_inches="tight")
