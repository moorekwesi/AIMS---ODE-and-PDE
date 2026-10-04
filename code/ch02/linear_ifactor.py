# The integrating factor method in SymPy, compared with dsolve
# Applied ODE & PDE with Python, Ch. 2 | (c) 2026 Stephen E. Moore | MIT Licence
import sympy as sp

t, C = sp.symbols("t C", real=True)


def linear_solve(p, g, t0=None, y0=None):
    """General (or IVP) solution of y' + p(t) y = g(t) by an integrating factor."""
    mu = sp.exp(sp.integrate(p, t))                 # mu = exp(int p dt)
    mu = sp.simplify(mu)
    ysol = (sp.integrate(sp.simplify(mu * g), t) + C) / mu
    if t0 is not None:
        cval = sp.solve(sp.Eq(ysol.subs(t, t0), y0), C)[0]
        ysol = ysol.subs(C, cval)
    return mu, sp.expand(ysol)


# Example 1: t y' + 2y = 4t^2, y(1) = 2   (divide by t: p = 2/t, g = 4t)
mu, ysol = linear_solve(2 / t, 4 * t, 1, 2)
print("Ex 1: mu =", mu, "  y =", ysol)

# Example 2: y' + y = sin t (general solution)
mu, ysol = linear_solve(1, sp.sin(t))
print("Ex 2: mu =", mu, "  y =", ysol)

# the same with dsolve
y = sp.Function("y")
print("dsolve 1:", sp.dsolve(t * y(t).diff(t) + 2 * y(t) - 4 * t**2, y(t),
                             ics={y(1): 2}))
print("dsolve 2:", sp.dsolve(y(t).diff(t) + y(t) - sp.sin(t), y(t)))
# a case where the antiderivative is not elementary: y' - 2t y = 1
print("dsolve 3:", sp.dsolve(y(t).diff(t) - 2 * t * y(t) - 1, y(t)))
