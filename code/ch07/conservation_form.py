# Checking conservation forms and characteristic speeds with SymPy
# Applied ODE & PDE with Python, Ch. 7 | (c) 2026 Stephen E. Moore | MIT Licence
import sympy as sp

x, t, c, g, vmax, rhomax = sp.symbols("x t c g v_max rho_max", positive=True)
u = sp.Function("u")(x, t)

# 1. Candidate fluxes phi for equations  u_t + (non-conservative terms) = 0
tests = {
    "u_t + c u_xx = 0":         (c * u.diff(x, 2),               c * u.diff(x)),
    "u_t + u^3 u_x + u_xx = 0": (u**3 * u.diff(x) + u.diff(x, 2), u**4 / 4 + u.diff(x)),
    "u_t + u u_x = 0":          (u * u.diff(x),                  u**2 / 2),
}
for name, (terms, phi) in tests.items():
    check = sp.simplify(sp.diff(phi, x) - terms)
    print(f"{name:26s} flux phi = {str(phi):28s} check: {check}")

# 2. Characteristic speed f'(u) of scalar fluxes
w, rho = sp.symbols("w rho")
print("Burgers     f'(u)   =", sp.diff(w**2 / 2, w))
f_traffic = vmax * rho * (1 - rho / rhomax)
print("Traffic     f'(rho) =", sp.expand(sp.diff(f_traffic, rho)))

# 3. Shallow water: conserved variables q = (h, m) with m = h v
h, m = sp.symbols("h m", positive=True)
F = sp.Matrix([m, m**2 / h + g * h**2 / 2])
J = F.jacobian([h, m])
v = sp.symbols("v")
eigs = [sp.simplify(e.subs(m, h * v)) for e in J.eigenvals()]
print("Shallow water Jacobian =", J.subs(m, h * v).applyfunc(sp.simplify))
print("eigenvalues            =", eigs)
