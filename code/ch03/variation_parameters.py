# Variation of parameters: y_p = -y1 int(y2 g / W) + y2 int(y1 g / W)
# Applied ODE & PDE with Python, Ch. 3 | (c) 2026 Stephen E. Moore | MIT Licence
import sympy as sp

x = sp.symbols("x")
y = sp.Function("y")


def var_params(y1, y2, g):
    """Particular solution of y'' + p y' + q y = g (leading coefficient 1!)
    from a fundamental set {y1, y2} of the homogeneous equation."""
    W = sp.simplify(sp.wronskian([y1, y2], x))
    u1 = sp.integrate(sp.simplify(-y2 * g / W), x)
    u2 = sp.integrate(sp.simplify(y1 * g / W), x)
    return sp.simplify(u1 * y1 + u2 * y2), W


# (a) y'' + y = sec x  (undetermined coefficients cannot handle sec x)
yp, W = var_params(sp.cos(x), sp.sin(x), sp.sec(x))
print("(a) W =", W, "  y_p =", yp)
print("    residual:", sp.simplify(yp.diff(x, 2) + yp - sp.sec(x)))

# (b) y'' - 2y' + y = e^x / x  (repeated root 1, fundamental set e^x, x e^x)
yp, W = var_params(sp.exp(x), x * sp.exp(x), sp.exp(x) / x)
print("(b) W =", W, "  y_p =", yp)

# (c) variable coefficients: x^2 y'' - 2x y' + 2y = x^3 on x > 0, with y1 = x, y2 = x^2;
#     divide by x^2 first to get the standard form with g = x
yp, W = var_params(x, x**2, x)
print("(c) W =", W, "  y_p =", sp.expand(yp))

# compare (a) with dsolve
print("dsolve (a):", sp.dsolve(y(x).diff(x, 2) + y(x) - sp.sec(x), y(x),
      hint="nth_linear_constant_coeff_variation_of_parameters").rhs)
