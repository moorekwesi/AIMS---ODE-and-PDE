# Classifying PDEs by linearity with SymPy
# Applied ODE & PDE with Python, Ch. 5 | (c) 2026 Stephen E. Moore | MIT Licence
import itertools
import sympy as sp


def classify(expr, u):
    """Classify the PDE 'expr = 0' for the unknown function u = u(x, y, ...)."""
    derivs = sorted(expr.atoms(sp.Derivative), key=sp.default_sort_key)
    derivs = [d for d in derivs if d.expr == u]
    order = max(d.derivative_count for d in derivs)
    # replace u and each derivative by a plain symbol p0, p1, p2, ...
    syms = sp.symbols(f"p0:{len(derivs) + 1}")
    e = expr.xreplace(dict(zip(derivs, syms[1:]))).xreplace({u: syms[0]})
    top = [s for d, s in zip(derivs, syms[1:]) if d.derivative_count == order]

    def affine(e, vars_):            # all second partials w.r.t. vars_ vanish
        return all(sp.simplify(sp.diff(e, p, q)) == 0
                   for p, q in itertools.combinations_with_replacement(vars_, 2))

    if affine(e, syms):
        homog = sp.simplify(e.subs({s: 0 for s in syms})) == 0
        kind = "linear, " + ("homogeneous" if homog else "inhomogeneous")
    elif affine(e, top) and all(not sp.diff(e, p).has(*syms) for p in top):
        kind = "semilinear"
    elif affine(e, top):
        kind = "quasilinear"
    else:
        kind = "fully nonlinear"
    return order, kind


x, y, t = sp.symbols("x y t")
u = sp.Function("u")(x, t)
w = sp.Function("u")(x, y)
D = sp.diff
examples = {
    "u_t = x^2 u_xx + 2x u_x":   (D(u, t) - x**2*D(u, x, 2) - 2*x*D(u, x), u),
    "-u_xx - u_yy = sin u":      (-D(w, x, 2) - D(w, y, 2) - sp.sin(w), w),
    "u_t = 5u_xxx + x^2 u + x":  (D(u, t) - 5*D(u, x, 3) - x**2*u - x, u),
    "u_t + u u_x = 0":           (D(u, t) + u*D(u, x), u),
    "u_t + u u_x = nu u_xx":     (D(u, t) + u*D(u, x) - D(u, x, 2)/10, u),
    "u_t + 6u u_x + u_xxx = 0":  (D(u, t) + 6*u*D(u, x) + D(u, x, 3), u),
    "u_t = (u u_x)_x":           (D(u, t) - D(u*D(u, x), x), u),
    "u_xx u_yy - u_xy^2 = 1":    (D(w, x, 2)*D(w, y, 2) - D(w, x, y)**2 - 1, w),
    "u_x^2 + u_y^2 = 1":         (D(w, x)**2 + D(w, y)**2 - 1, w),
}
for name, (e, f) in examples.items():
    order, kind = classify(e, f)
    print(f"{name:26s} order {order}: {kind}")
