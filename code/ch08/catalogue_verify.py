# Symbolic verification of the catalogue of exact solutions
# Applied ODE & PDE with Python, Ch. 8 | (c) 2026 Stephen E. Moore | MIT Licence
import sympy as sp

x, y, t = sp.symbols("x y t", real=True)
k, c, a, g = sp.symbols("kappa c a gamma", positive=True)
pi, E = sp.pi, sp.exp
mu = sp.sqrt(c**2 * pi**2 - g**2)
heat = lambda u, F=0: sp.diff(u, t) - k * sp.diff(u, x, 2) - F
wave = lambda u: sp.diff(u, t, 2) - c**2 * sp.diff(u, x, 2)
lapl = lambda u, F=0: -sp.diff(u, x, 2) - sp.diff(u, y, 2) - F

src = (k * pi**2 - 1) * E(-t) * sp.sin(pi * x)
w = x / 10 - 5 * x**3 / 3 + 8 * x**4 / 3 - 11 * x**5 / 10
cat = [
    ("H1 Dirichlet mode", heat(E(-k * pi**2 * t) * sp.sin(pi * x))),
    ("H2 Neumann mode", heat(1 + E(-k * pi**2 * t) * sp.cos(pi * x))),
    ("H3 shifted BCs", heat(x + E(-k * pi**2 * t) * sp.sin(pi * x))),
    ("H4 Gaussian", heat(E(-x**2 / (1 + 4 * k * t)) / sp.sqrt(1 + 4 * k * t))),
    ("H5 erfc step", heat(sp.erfc(x / sp.sqrt(4 * k * t)) / 2)),
    ("H6 manufactured", heat(E(-t) * sp.sin(pi * x), src)),
    ("H7 polynomial MMS", heat((1 + t) * x * (1 - x), x * (1 - x) + 2 * k * (1 + t))),
    ("S1 steady source", -sp.diff(w, x, 2) - x * (1 - x) * (10 - 22 * x)),
    ("W1 standing wave", wave(sp.sin(pi * x) * sp.cos(pi * c * t))),
    ("W2 dAlembert", wave((E(-(x - c * t)**2) + E(-(x + c * t)**2)) / 2)),
    ("W3 struck mode", wave(sp.sin(pi * x) * sp.sin(pi * c * t) / (pi * c))),
    ("W4 damped mode", wave(E(-g * t) * sp.cos(mu * t) * sp.sin(pi * x))
     + 2 * g * sp.diff(E(-g * t) * sp.cos(mu * t) * sp.sin(pi * x), t)),
    ("T1 transport", sp.diff(sp.sin(2 * pi * (x - a * t)), t)
     + a * sp.diff(sp.sin(2 * pi * (x - a * t)), x)),
    ("L1 exp*sin", lapl(E(pi * x) * sp.sin(pi * y))),
    ("L2 sinh mode", lapl(sp.sin(pi * x) * sp.sinh(pi * y) / sp.sinh(pi))),
    ("P1 sine bump", lapl(sp.sin(pi * x) * sp.sin(pi * y),
                          2 * pi**2 * sp.sin(pi * x) * sp.sin(pi * y))),
    ("P2 polynomial", lapl(x * (1 - x) * y * (1 - y),
                           2 * (x * (1 - x) + y * (1 - y)))),
]
for name, res in cat:
    print(f"{name:18s} residual = {sp.simplify(res)}")

print("\nBoundary checks:")
print("S1: w(0), w(1) =", w.subs(x, 0), w.subs(x, 1), "  w(1/2) =", w.subs(x, sp.Rational(1, 2)))
print("H2: u_x(0), u_x(1) =", [sp.diff(1 + E(-k * pi**2 * t) * sp.cos(pi * x), x).subs(x, s)
                             for s in (0, 1)])
print("H6 source F(x,t) =", sp.factor(src))
