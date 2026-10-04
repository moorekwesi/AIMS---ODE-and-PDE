# Solving initial value problems with the Laplace transform in SymPy
# Applied ODE & PDE with Python, Ch. 3 | (c) 2026 Stephen E. Moore | MIT Licence
import sympy as sp

t, s = sp.symbols("t s", positive=True)
Y = sp.symbols("Y")


def L(f):
    """Laplace transform of f(t) (convergence conditions dropped)."""
    return sp.laplace_transform(f, t, s, noconds=True)


def solve_by_laplace(a, b, c, g, y0, y1):
    """a y'' + b y' + c y = g(t), y(0) = y0, y'(0) = y1, via
    L[y'] = sY - y0 and L[y''] = s^2 Y - s y0 - y1."""
    eq = sp.Eq(a * (s**2 * Y - s * y0 - y1) + b * (s * Y - y0) + c * Y, L(g))
    Ys = sp.solve(eq, Y)[0]
    if Ys.has(sp.exp):                          # shifted inputs: keep exp(-c s)
        print("   Y(s) =", sp.factor(Ys))
    else:                                       # rational: partial fractions
        print("   Y(s) =", sp.apart(Ys, s))
    return sp.simplify(sp.inverse_laplace_transform(Ys, s, t))


print("(a) y'' + 3y' + 2y = exp(-3t), y(0) = 1, y'(0) = 0")
ya = solve_by_laplace(1, 3, 2, sp.exp(-3 * t), 1, 0)
print("   y(t) =", ya)

print("(b) y'' + 4y = u(t - pi) (switch on at t = pi), y(0) = y'(0) = 0")
yb = solve_by_laplace(1, 0, 4, sp.Heaviside(t - sp.pi), 0, 0)
print("   y(t) =", yb)

print("(c) y'' + 2y' + 5y = delta(t - 1) (hammer blow at t = 1), y(0) = 0, y'(0) = 0")
yc = solve_by_laplace(1, 2, 5, sp.DiracDelta(t - 1), 0, 0)
print("   y(t) =", yc)
