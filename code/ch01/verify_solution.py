# Verifying solutions of differential equations with SymPy
# Applied ODE & PDE with Python, Ch. 1 | (c) 2026 Stephen E. Moore | MIT Licence
import sympy as sp

t, x, C, c1, c2, k = sp.symbols("t x C c1 c2 k")
y = sp.Function("y")

# 1. First-order linear ODE  y' + 2y = 4t  with candidate family  2t - 1 + C e^{-2t}
ode1 = sp.Eq(y(t).diff(t) + 2*y(t), 4*t)
cand1 = 2*t - 1 + C*sp.exp(-2*t)
residual = sp.simplify(cand1.diff(t) + 2*cand1 - 4*t)
print("1. residual of y' + 2y - 4t     :", residual)
print("   checkodesol                   :", sp.checkodesol(ode1, sp.Eq(y(t), cand1)))

# 2. A function that is NOT a solution: y = t^2 for y' = 2y
ode2 = sp.Eq(y(t).diff(t), 2*y(t))
print("2. y = t^2 in y' = 2y            :", sp.checkodesol(ode2, sp.Eq(y(t), t**2)))

# 3. Second-order ODE  y'' + 4y = 0  with two arbitrary constants
ode3 = sp.Eq(y(t).diff(t, 2) + 4*y(t), 0)
cand3 = c1*sp.cos(2*t) + c2*sp.sin(2*t)
print("3. y'' + 4y = 0                  :", sp.checkodesol(ode3, sp.Eq(y(t), cand3)))

# 4. Implicit solution x^2 + y^2 = 25 of y' = -x/y (implicit differentiation)
F = x**2 + y(x)**2 - 25
dydx = sp.solve(sp.diff(F, x), y(x).diff(x))[0]
print("4. implicit differentiation, y' =", dydx)

# 5. A PDE: u = exp(-k^2 t) sin(k x) satisfies the heat equation u_t = u_xx
u = sp.exp(-k**2*t)*sp.sin(k*x)
print("5. u_t - u_xx                    :", sp.simplify(u.diff(t) - u.diff(x, 2)))
