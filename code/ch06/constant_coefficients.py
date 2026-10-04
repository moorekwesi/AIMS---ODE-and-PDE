# The coordinate method
# Applied ODE & PDE with Python, Ch. 6 | (c) 2026 Stephen E. Moore | MIT Licence
import sympy as sp

x, y, t, xi, eta = sp.symbols("x y t xi eta", real=True)
w = sp.Function("w")
d = sp.diff


def coordinate_method(a, b, c, f):
    """Solve a u_x + b u_y + c u = f(x, y) via xi = b x - a y, eta = a x + b y.
    In the new variables: (a^2 + b^2) w_eta + c w = f, a linear ODE in eta."""
    k = a**2 + b**2
    X = (b*xi + a*eta)/k                       # inverse transformation
    Y = (-a*xi + b*eta)/k
    ode = sp.Eq(k*d(w(eta), eta) + c*w(eta), f.subs({x: X, y: Y}))
    sol = sp.dsolve(ode, w(eta)).rhs           # contains a constant C1
    G = sp.Function("G")
    sol = sol.subs(sp.Symbol("C1"), G(xi))     # the constant is a function of xi
    return sp.simplify(sol.subs({xi: b*x - a*y, eta: a*x + b*y}))


# Example 1: 3 u_x - 2 u_y + u = x   (general solution)
u1 = coordinate_method(3, -2, 1, x)
print("general solution:", u1)
print("residual        :", sp.simplify(3*d(u1, x) - 2*d(u1, y) + u1 - x))
print("sympy pdsolve   :", sp.pdsolve(sp.Eq(3*d(sp.Function("u")(x, y), x)
      - 2*d(sp.Function("u")(x, y), y) + sp.Function("u")(x, y), x)).rhs)

# Example 2: damped transport 3 u_t + 4 u_x + u = 0, u(x,0) = x^2
u2 = coordinate_method(4, 3, 1, sp.Integer(0)).subs(y, t)   # (x, y) -> (x, t)
print("\ndamped transport, general:", u2)
G, g = sp.Function("G"), sp.Symbol("g")
s = sp.symbols("s", real=True)
# at t = 0 the solution is G(3x) exp(-4x/25); set it equal to x^2, put s = 3x
G_s = sp.solve(u2.subs(t, 0).subs(G(3*x), g) - x**2, g)[0].subs(x, s/3)
u2p = sp.simplify(u2.subs(G(3*x - 4*t), G_s.subs(s, 3*x - 4*t)))
print("with u(x,0)=x^2         :", sp.factor(u2p))
print("residual, u(x,0)        :", sp.simplify(3*d(u2p, t) + 4*d(u2p, x) + u2p),
      ",", sp.simplify(u2p.subs(t, 0)))
