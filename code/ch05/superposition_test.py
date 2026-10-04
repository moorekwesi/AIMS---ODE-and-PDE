# Testing linearity by superposition
# Applied ODE & PDE with Python, Ch. 5 | (c) 2026 Stephen E. Moore | MIT Licence
import sympy as sp

x, t = sp.symbols("x t", real=True)
a, b = sp.symbols("a b")                 # arbitrary constants
U, V = sp.Function("U")(x, t), sp.Function("V")(x, t)
D = sp.diff


def is_linear(L):
    """True if L[a U + b V] - a L[U] - b L[V] vanishes identically."""
    defect = L(a*U + b*V) - a*L(U) - b*L(V)
    return sp.simplify(defect) == 0


operators = {
    "heat      u_t - u_xx":          lambda u: D(u, t) - D(u, x, 2),
    "exam      u_t - x^2u_xx-2xu_x": lambda u: D(u, t) - x**2*D(u, x, 2)
                                               - 2*x*D(u, x),
    "exam      u_t - 5u_xxx - x^2u": lambda u: D(u, t) - 5*D(u, x, 3)
                                               - x**2*u,
    "sine-Gordon u_tt-u_xx+sin u":   lambda u: D(u, t, 2) - D(u, x, 2)
                                               + sp.sin(u),
    "Burgers   u_t + u u_x":         lambda u: D(u, t) + u*D(u, x),
    "eikonal   u_x^2 + u_t^2":       lambda u: D(u, x)**2 + D(u, t)**2,
    "affine    u_t - u_xx - x":      lambda u: D(u, t) - D(u, x, 2) - x,
}
for name, L in operators.items():
    print(f"{name:32s} linear: {is_linear(L)}")
