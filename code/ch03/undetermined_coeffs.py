# Undetermined coefficients by hand-coded ansatz, checked against dsolve
# Applied ODE & PDE with Python, Ch. 3 | (c) 2026 Stephen E. Moore | MIT Licence
import sympy as sp

x, A, B, C = sp.symbols("x A B C")
y = sp.Function("y")


def L(u):
    """The operator L[u] = u'' - 3u' + 2u (characteristic roots 1 and 2)."""
    return sp.diff(u, x, 2) - 3 * sp.diff(u, x) + 2 * u


# (1) g = 6 e^{3x}: 3 is not a root, try A e^{3x}
yp1 = A * sp.exp(3 * x)
A1 = sp.solve(sp.simplify(L(yp1) / sp.exp(3 * x)) - 6, A)[0]
print("(1) y_p =", yp1.subs(A, A1))

# (2) g = e^{x}: 1 IS a root (resonance), so A e^x fails; try A x e^x
print("    L[A e^x] =", sp.simplify(L(A * sp.exp(x))))
yp2 = A * x * sp.exp(x)
A2 = sp.solve(sp.simplify(L(yp2) / sp.exp(x)) - 1, A)[0]
print("(2) y_p =", yp2.subs(A, A2))

# (3) g = 10 sin x: try A cos x + B sin x and match coefficients
yp3 = A * sp.cos(x) + B * sp.sin(x)
res = sp.expand(L(yp3) - 10 * sp.sin(x))
sol3 = sp.solve([res.coeff(sp.cos(x)), res.coeff(sp.sin(x))], [A, B])
print("(3) y_p =", yp3.subs(sol3))

# (4) g = 4x^2: try A x^2 + B x + C
yp4 = A * x**2 + B * x + C
sol4 = sp.solve(sp.Poly(L(yp4) - 4 * x**2, x).coeffs(), [A, B, C])
print("(4) y_p =", yp4.subs(sol4))

# dsolve for the exam problem and for the resonant case
ode = y(x).diff(x, 2) - 3 * y(x).diff(x) + 2 * y(x)
print("dsolve (1):", sp.dsolve(ode - 6 * sp.exp(3 * x), y(x)).rhs)
print("dsolve (2):", sp.dsolve(ode - sp.exp(x), y(x)).rhs)
