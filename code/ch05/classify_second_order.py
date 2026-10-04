# Type of a second-order equation
# Applied ODE & PDE with Python, Ch. 5 | (c) 2026 Stephen E. Moore | MIT Licence
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

x, y = sp.symbols("x y", real=True)


def pde_type(A, B, C, point=None):
    """Return the discriminant B^2 - 4AC and, if a point (or constant
    coefficients) is given, the type there."""
    disc = sp.expand(B**2 - 4*A*C)
    d = disc.subs({x: point[0], y: point[1]}) if point else disc
    if not d.is_number:
        return disc, "depends on (x, y)"
    return disc, ("hyperbolic" if d > 0 else "parabolic" if d == 0
                  else "elliptic")


examples = {                                      # (A, B, C)
    "u_xx - 5u_xy + 6u_yy":       (1, -5, 6),
    "u_xx + 4u_xy + 4u_yy":       (1, 4, 4),
    "u_xx + 2u_xy + 5u_yy":       (1, 2, 5),
    "heat u_t = u_xx  (y = t)":   (1, 0, 0),
    "Tricomi y u_xx + u_yy":      (y, 0, 1),
    "u_xx + 2x u_xy + (1-y^2)u_yy": (1, 2*x, 1 - y**2),
}
for name, (A, B, C) in examples.items():
    disc, kind = pde_type(sp.sympify(A), sp.sympify(B), sp.sympify(C))
    print(f"{name:30s} B^2-4AC = {str(disc):20s} {kind}")
A, B, C = (y, 0, 1)
for pt in [(0, 1), (0, 0), (0, -1)]:
    print(f"Tricomi at (x,y) = {pt}: {pde_type(A, B, C, pt)[1]}")

# type maps: sign of the discriminant on a grid
fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
X, Y = np.meshgrid(np.linspace(-2, 2, 401), np.linspace(-2, 2, 401))
maps = [("Tricomi: y u_xx + u_yy", (y, 0, 1)),
        ("u_xx + 2x u_xy + (1-y^2) u_yy", (1, 2*x, 1 - y**2))]
for ax, (name, coeffs) in zip(axes, maps):
    disc = sp.lambdify((x, y), pde_type(*map(sp.sympify, coeffs))[0])
    Dval = disc(X, Y)*np.ones_like(X)
    ax.contourf(X, Y, np.sign(Dval), levels=[-1.5, -0.5, 0.5, 1.5],
                colors=["#9ecae1", "0.6", "#fdd0a2"])
    ax.contour(X, Y, Dval, levels=[0], colors="k")
    ax.set_title(name, fontsize=10); ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_aspect("equal")
axes[0].text(-1.6, 1.0, "elliptic"); axes[0].text(-1.6, -1.2, "hyperbolic")
axes[1].text(-0.6, 0.0, "elliptic"); axes[1].text(-1.9, 1.6, "hyperbolic")
plt.tight_layout()
plt.savefig("ch05_type_maps.pdf", bbox_inches="tight")
