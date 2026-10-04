# Verifying exact solutions of famous PDEs with SymPy
# Applied ODE & PDE with Python, Ch. 5 | (c) 2026 Stephen E. Moore | MIT Licence
import sympy as sp

x, y, t, S = sp.symbols("x y t S", real=True)
c, kappa, nu, k, sigma, r, K, T = sp.symbols("c kappa nu k sigma r K T",
                                              positive=True)
F, G = sp.Function("F"), sp.Function("G")
D = sp.diff                      # short name for partial derivatives


def residual(lhs):
    """Simplify the left-hand side of 'PDE = 0' after substitution."""
    return sp.simplify(lhs)


gallery = {}
# 1. transport u_t + c u_x = 0
u = F(x - c*t)
gallery["transport"] = D(u, t) + c*D(u, x)
# 2. heat u_t = kappa u_xx  (heat kernel)
u = sp.exp(-x**2/(4*kappa*t))/sp.sqrt(4*sp.pi*kappa*t)
gallery["heat"] = D(u, t) - kappa*D(u, x, 2)
# 3. wave u_tt = c^2 u_xx  (d'Alembert)
u = F(x - c*t) + G(x + c*t)
gallery["wave"] = D(u, t, 2) - c**2*D(u, x, 2)
# 4. Laplace u_xx + u_yy = 0 (fundamental solution in 2D)
u = sp.log(x**2 + y**2)
gallery["Laplace"] = D(u, x, 2) + D(u, y, 2)
# 5. viscous Burgers u_t + u u_x = nu u_xx (travelling front)
u = 1 - sp.tanh((x - t)/(2*nu))
gallery["Burgers"] = D(u, t) + u*D(u, x) - nu*D(u, x, 2)
# 6. KdV u_t + 6 u u_x + u_xxx = 0 (soliton)
u = c/2*sp.sech(sp.sqrt(c)/2*(x - c*t))**2
gallery["KdV"] = D(u, t) + 6*u*D(u, x) + D(u, x, 3)
# 7. Schrodinger i psi_t = -psi_xx (plane wave)
psi = sp.exp(sp.I*(k*x - k**2*t))
gallery["Schrodinger"] = sp.I*D(psi, t) + D(psi, x, 2)
# 8. Black-Scholes (forward contract V = S - K exp(-r(T-t)))
V = S - K*sp.exp(-r*(T - t))
gallery["Black-Scholes"] = (D(V, t) + sigma**2*S**2/2*D(V, S, 2)
                            + r*S*D(V, S) - r*V)
# 9. Fisher-KPP u_t = u_xx + u(1-u) (Ablowitz-Zeppetella wave)
u = 1/(1 + sp.exp(x/sp.sqrt(6) - sp.Rational(5, 6)*t))**2
gallery["Fisher-KPP"] = D(u, t) - D(u, x, 2) - u*(1 - u)
# 10. 2D Navier-Stokes (Taylor-Green vortex, density 1)
E = sp.exp(-2*nu*t)
u1, u2 = -sp.cos(x)*sp.sin(y)*E, sp.sin(x)*sp.cos(y)*E
p = -(sp.cos(2*x) + sp.cos(2*y))/4*E**2
gallery["NS x-momentum"] = (D(u1, t) + u1*D(u1, x) + u2*D(u1, y) + D(p, x)
                            - nu*(D(u1, x, 2) + D(u1, y, 2)))
gallery["NS y-momentum"] = (D(u2, t) + u1*D(u2, x) + u2*D(u2, y) + D(p, y)
                            - nu*(D(u2, x, 2) + D(u2, y, 2)))
gallery["NS div u"] = D(u1, x) + D(u2, y)

for name, lhs in gallery.items():
    print(f"{name:15s} residual = {residual(lhs)}")
