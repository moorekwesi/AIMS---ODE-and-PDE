# The transversality condition
# Applied ODE & PDE with Python, Ch. 6 | (c) 2026 Stephen E. Moore | MIT Licence
import sympy as sp

x, y, r = sp.symbols("x y r", real=True)


def jacobian(a, b, curve):
    """J(r) = x0'(r) b - y0'(r) a for a(x,y) u_x + b(x,y) u_y = ..., with the
    data given on the curve (x0(r), y0(r)).  J != 0 means non-characteristic."""
    x0, y0 = curve
    on = {x: x0, y: y0}
    return sp.simplify(sp.diff(x0, r)*b.subs(on) - sp.diff(y0, r)*a.subs(on))


cases = [  # (description, a, b, initial curve)
    ("u_x + u_y = u,  data on y = 0",         1, 1, (r, 0)),
    ("u_x + u_y = 0,  data on y = x",         1, 1, (r, r)),
    ("x u_x + y u_y = u, unit circle",        x, y, (sp.cos(r), sp.sin(r))),
    ("y u_x - x u_y = 0, unit circle",        y, -x, (sp.cos(r), sp.sin(r))),
    ("y u_x - x u_y = 0, x-axis",             y, -x, (r, 0)),
    ("u_x + 2x u_y = 0, parabola y = x^2",    1, 2*x, (r, r**2)),
    ("u_x + 2x u_y = 0, data on x = 0",       1, 2*x, (0, r)),
]
for text, a, b, curve in cases:
    J = jacobian(sp.sympify(a), sp.sympify(b), curve)
    verdict = "characteristic" if J == 0 else "non-characteristic where J != 0"
    print(f"{text:38s} J(r) = {str(J):4s} {verdict}")

# the solution of x u_x + y u_y = u with u = f(theta) on the unit circle
f = sp.Function("f")
rho, theta = sp.sqrt(x**2 + y**2), sp.atan2(y, x)
u = rho*f(theta)
print("residual x u_x + y u_y - u =", sp.simplify(x*sp.diff(u, x)
                                                 + y*sp.diff(u, y) - u))
