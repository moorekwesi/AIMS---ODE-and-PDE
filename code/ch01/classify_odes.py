# Classifying ODEs with SymPy's classify_ode
# Applied ODE & PDE with Python, Ch. 1 | (c) 2026 Stephen E. Moore | MIT Licence
import sympy as sp

t = sp.symbols("t")
y = sp.Function("y")
yp, ypp = y(t).diff(t), y(t).diff(t, 2)

equations = {
    "y' = -2 t y + t":         sp.Eq(yp, -2*t*y(t) + t),
    "y' = y (1 - y)":          sp.Eq(yp, y(t)*(1 - y(t))),
    "y' = (t + y)^2":          sp.Eq(yp, (t + y(t))**2),
    "y'' + 4 y' + 13 y = 0":   sp.Eq(ypp + 4*yp + 13*y(t), 0),
    "t^2 y'' + t y' - y = 0":  sp.Eq(t**2*ypp + t*yp - y(t), 0),
    "y'' + sin(y) = 0":        sp.Eq(ypp + sp.sin(y(t)), 0),
}

for text, eq in equations.items():
    hints = sp.classify_ode(eq, y(t))
    order = sp.ode_order(eq, y(t))
    main = [h for h in hints
            if not h.endswith("_Integral") and h != "factorable"][:4]
    print(f"{text:24s} order {order}")
    print("    ", ", ".join(main) if main else "(no method found)")
