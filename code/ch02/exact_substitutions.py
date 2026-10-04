# Exact equations, Bernoulli, homogeneous and Riccati equations in SymPy
# Applied ODE & PDE with Python, Ch. 2 | (c) 2026 Stephen E. Moore | MIT Licence
import sympy as sp

x, y, t = sp.symbols("x y t", real=True)

# 1. Exactness test and potential for M dx + N dy = 0
M = 3 * x**2 + 2 * x * y**2
N = 2 * x**2 * y + 4 * y**3
print("M_y - N_x =", sp.simplify(sp.diff(M, y) - sp.diff(N, x)))
F = sp.integrate(M, x)                              # F = int M dx + h(y)
h_prime = sp.simplify(N - sp.diff(F, y))            # must depend on y only
F = F + sp.integrate(h_prime, y)
print("potential F(x, y) =", F)

# 2. Integrating factor mu(x) for (3xy + y^2) dx + (x^2 + xy) dy = 0
M2, N2 = 3 * x * y + y**2, x**2 + x * y
ratio = sp.simplify((sp.diff(M2, y) - sp.diff(N2, x)) / N2)
print("(M_y - N_x)/N =", ratio, " -> mu =", sp.exp(sp.integrate(ratio, x)))
mu = x
print("exact after mu?", sp.simplify(sp.diff(mu * M2, y) - sp.diff(mu * N2, x)) == 0)

# 3. Bernoulli (logistic): P' = r P (1 - P/K)
r, K, P0 = sp.symbols("r K P_0", positive=True)
P = sp.Function("P")
sol = sp.dsolve(P(t).diff(t) - r * P(t) * (1 - P(t) / K), P(t), ics={P(0): P0})
print("logistic:", sp.simplify(sol.rhs))

# 4. Homogeneous: y' = (x^2 + y^2)/(x y)
Y = sp.Function("Y")
sol = sp.dsolve(Y(x).diff(x) - (x**2 + Y(x)**2) / (x * Y(x)), Y(x))
print("homogeneous:", sol)

# 5. Riccati with known particular solution y1 = t: y' = 1 + t^2 - 2 t y + y^2
u = sp.Function("u")
riccati = lambda yy: sp.diff(yy, t) - (1 + t**2 - 2 * t * yy + yy**2)
print("y1 = t solves it?", sp.simplify(riccati(t)) == 0)
eq_u = sp.simplify(riccati(t + u(t)))               # equation for u = y - y1
print("equation for u:", sp.Eq(eq_u, 0))
print("u =", sp.dsolve(eq_u, u(t)).rhs)
