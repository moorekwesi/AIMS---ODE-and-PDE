# Reduction to canonical form
# Applied ODE & PDE with Python, Ch. 5 | (c) 2026 Stephen E. Moore | MIT Licence
import sympy as sp

x, y = sp.symbols("x y", positive=True)
X, Y = sp.symbols("xi eta", positive=True)        # the new variables
d = sp.diff


def transform(A, B, C, D, E, xi, eta):
    """Coefficients of the operator in the variables (xi, eta), returned as
    functions of (x, y): (A*, B*, C*, D*, E*)."""
    As = A*d(xi, x)**2 + B*d(xi, x)*d(xi, y) + C*d(xi, y)**2
    Bs = (2*A*d(xi, x)*d(eta, x) + B*(d(xi, x)*d(eta, y) + d(xi, y)*d(eta, x))
          + 2*C*d(xi, y)*d(eta, y))
    Cs = A*d(eta, x)**2 + B*d(eta, x)*d(eta, y) + C*d(eta, y)**2
    Ds = A*d(xi, x, 2) + B*d(xi, x, y) + C*d(xi, y, 2) + D*d(xi, x) + E*d(xi, y)
    Es = (A*d(eta, x, 2) + B*d(eta, x, y) + C*d(eta, y, 2) + D*d(eta, x)
          + E*d(eta, y))
    return [sp.simplify(c) for c in (As, Bs, Cs, Ds, Es)]


def check(A, B, C, D, E, xi, eta, new):
    """Verify the chain rule on the test function w = sin(xi) eta^3 + xi^2 eta."""
    w = sp.sin(X)*Y**3 + X**2*Y
    u = w.subs({X: xi, Y: eta})
    lhs = A*d(u, x, 2) + B*d(u, x, y) + C*d(u, y, 2) + D*d(u, x) + E*d(u, y)
    As, Bs, Cs, Ds, Es = new
    rhs = (As*d(w, X, 2) + Bs*d(w, X, Y) + Cs*d(w, Y, 2) + Ds*d(w, X)
           + Es*d(w, Y)).subs({X: xi, Y: eta})
    return sp.simplify(lhs - rhs) == 0


cases = {   # name: (A, B, C, D, E, xi, eta)
    "hyperbolic u_xx-5u_xy+6u_yy": (1, -5, 6, 0, 0, y + 2*x, y + 3*x),
    "parabolic  u_xx+4u_xy+4u_yy": (1, 4, 4, 0, 0, y - 2*x, x),
    "elliptic   u_xx+2u_xy+5u_yy": (1, 2, 5, 0, 0, y - x, 2*x),
    "variable   x^2u_xx-y^2u_yy":  (x**2, 0, -y**2, 0, 0, x*y, y/x),
}
for name, (A, B, C, D, E, xi, eta) in cases.items():
    new = transform(A, B, C, D, E, xi, eta)
    # express the new coefficients in terms of xi and eta
    inv = sp.solve([sp.Eq(X, xi), sp.Eq(Y, eta)], [x, y], dict=True)[0]
    new_xe = [sp.simplify(c.subs(inv)) for c in new]
    print(name, " chain rule ok:", check(A, B, C, D, E, xi, eta, new))
    print("    [A*, B*, C*, D*, E*] =", new_xe)

# general solution of the variable-coefficient example, verified
F, G = sp.Function("F"), sp.Function("G")
u = F(x*y) + sp.sqrt(x*y)*G(y/x)
print("x^2u_xx - y^2u_yy for F(xy)+sqrt(xy)G(y/x):",
      sp.simplify(x**2*d(u, x, 2) - y**2*d(u, y, 2)))
