# Wronskians, linear independence and Abel's formula with SymPy
# Applied ODE & PDE with Python, Ch. 3 | (c) 2026 Stephen E. Moore | MIT Licence
import sympy as sp

x = sp.symbols("x", real=True)

# (a) sinh x and 3 sinh x - 2 cosh x for y'' - y = 0
y1, y2 = sp.sinh(x), 3 * sp.sinh(x) - 2 * sp.cosh(x)
for f in (y1, y2):
    print("residual of y'' - y:", sp.simplify(sp.diff(f, x, 2) - f))
W = sp.simplify(sp.wronskian([y1, y2], x))
print("W(sinh x, 3 sinh x - 2 cosh x) =", W)

# (b) e^x, e^(2x), e^(3x): a fundamental set for y''' - 6y'' + 11y' - 6y = 0
fs = [sp.exp(x), sp.exp(2 * x), sp.exp(3 * x)]
print("W(e^x, e^2x, e^3x) =", sp.simplify(sp.wronskian(fs, x)))

# (c) Abel's formula for x^2 y'' - 3x y' + 4y = 0 (solutions x^2, x^2 ln x)
u1, u2 = x**2, x**2 * sp.log(x)
Wu = sp.simplify(sp.wronskian([u1, u2], x))
p = -3 * x / x**2                                    # y'' + p y' + q y = 0
abel = sp.exp(-sp.integrate(p, x))                   # C exp(-int p dx), C = 1
print("W(x^2, x^2 ln x) =", Wu, "  Abel: C *", abel)

# (d) a zero Wronskian does not imply dependence for arbitrary functions
for name, s in (("x > 0", sp.symbols("s", positive=True)),
                ("x < 0", sp.symbols("s", negative=True))):
    W = sp.simplify(sp.wronskian([s**2, s * sp.Abs(s)], s))
    print(f"W(x^2, x|x|) for {name}:", W)
