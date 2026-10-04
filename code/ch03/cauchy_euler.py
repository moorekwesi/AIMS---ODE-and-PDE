# Cauchy-Euler equations and reduction of order with SymPy
# Applied ODE & PDE with Python, Ch. 3 | (c) 2026 Stephen E. Moore | MIT Licence
import sympy as sp

x = sp.symbols("x", positive=True)
m = sp.symbols("m")
y = sp.Function("y")

# Indicial equation of a x^2 y'' + b x y' + c y = 0: a m(m-1) + b m + c = 0
for a, b, c in [(1, -2, -4), (1, -3, 4), (1, 1, 4)]:
    ind = sp.expand(a * m * (m - 1) + b * m + c)
    ode = a * x**2 * y(x).diff(x, 2) + b * x * y(x).diff(x) + c * y(x)
    print(f"{a}x^2y'' + ({b})xy' + ({c})y = 0: indicial {ind} = 0, roots",
          sp.roots(ind, m))
    print("    dsolve:", sp.dsolve(ode, y(x)).rhs)


def reduction_of_order(p, y1):
    """Second solution y2 = y1 * int( exp(-int p) / y1^2 ) for y'' + p y' + q y = 0."""
    v_prime = sp.exp(-sp.integrate(p, x)) / y1**2
    return sp.simplify(y1 * sp.integrate(sp.simplify(v_prime), x))


# Legendre's equation of order 1: (1 - x^2) y'' - 2x y' + 2y = 0, y1 = x, |x| < 1
p = -2 * x / (1 - x**2)
y2 = reduction_of_order(p, x)
print("Legendre n=1: y2 =", y2)
# for 0 < x < 1 the logarithm has a negative argument; a real second solution is
Q1 = x * sp.atanh(x) - 1          # = (x/2) ln((1+x)/(1-x)) - 1
for f in (y2, Q1):
    print("    residual of", f, ":",
          sp.simplify((1 - x**2) * f.diff(x, 2) - 2 * x * f.diff(x) + 2 * f))
