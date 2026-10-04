# Separable equations with SymPy: implicit solutions and the interval of existence
# Applied ODE & PDE with Python, Ch. 2 | (c) 2026 Stephen E. Moore | MIT Licence
import sympy as sp

x = sp.symbols("x", real=True)
y = sp.Function("y")

# (a) y' = x/y - x/(1+y), y(0) = 1   (separate: y(1+y) dy = x dx)
Y = sp.symbols("Y")
lhs = sp.integrate(Y * (1 + Y), (Y, 1, Y))        # int_1^Y s(1+s) ds
rhs = sp.integrate(x, (x, 0, x))                  # int_0^x s ds
implicit = sp.Eq(sp.expand(6 * lhs), sp.expand(6 * rhs))
print("(a) implicit solution :", implicit)
# check that the implicit relation really solves the ODE (implicit differentiation)
G = 6 * lhs - 6 * rhs
dYdx = -sp.diff(G, x) / sp.diff(G, Y)
print("    dy/dx from G=0     :", sp.simplify(dYdx))
print("    ODE right-hand side:", sp.simplify(x / Y - x / (1 + Y)))

# (b) y' = (1 + 3x^2)/(2y), y(0) = 1: dsolve gives an explicit solution
ode = sp.Eq(y(x).diff(x), (1 + 3 * x**2) / (2 * y(x)))
sol = sp.dsolve(ode, y(x), ics={y(0): 1})
print("(b) dsolve            :", sol)
print("    checkodesol       :", sp.checkodesol(ode, sol))
# the solution exists while 1 + x + x^3 > 0: find the left end point
r = sp.nsolve(1 + x + x**3, x, -0.7)
print(f"    interval of existence: ({float(r):.6f}, oo)")
