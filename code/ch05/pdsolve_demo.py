# Solving first-order linear PDEs with sympy.pdsolve
# Applied ODE & PDE with Python, Ch. 5 | (c) 2026 Stephen E. Moore | MIT Licence
import sympy as sp

x, y, t = sp.symbols("x y t", real=True)
u = sp.Function("u")
D = sp.diff

# (a) constant coefficients, homogeneous
eq_a = sp.Eq(2*D(u(x, y), x) + 3*D(u(x, y), y), 0)
sol_a = sp.pdsolve(eq_a)
print("(a)", sol_a, " check:", sp.checkpdesol(eq_a, sol_a))

# (b) constant coefficients with a lower-order term and a source
eq_b = sp.Eq(D(u(x, y), x) + D(u(x, y), y) + u(x, y), sp.exp(x + 2*y))
sol_b = sp.pdsolve(eq_b)
print("(b)", sp.simplify(sol_b.rhs), " check:", sp.checkpdesol(eq_b, sol_b))

# (c) variable coefficients: the 2024 exam equation
eq_c = sp.Eq(D(u(x, t), t) + (1 + x**2)*D(u(x, t), x), 0)
sol_c = sp.pdsolve(eq_c)
print("(c)", sol_c)

# (d) fix the arbitrary function F from u(x, 0) = 1/(1 + x^2)
s = sp.symbols("s", real=True)
F = sp.Function("F")
# on t = 0 the argument of F is -atan(x) = s, i.e. x = -tan(s)
F_expr = (1/(1 + x**2)).subs(x, -sp.tan(s))           # F(s) = cos(s)^2
arg = sol_c.rhs.args[0]                                # t - atan(x)
u_c = sp.simplify(sp.expand_trig(sp.trigsimp(F_expr).subs(s, arg)))
print("(d) F(s) =", sp.trigsimp(F_expr), "  u(x,t) =", u_c)
print("    residual:", sp.simplify(D(u_c, t) + (1 + x**2)*D(u_c, x)),
      "  u(x,0) =", sp.simplify(u_c.subs(t, 0)))

# (e) pdsolve is limited to first order
try:
    sp.pdsolve(sp.Eq(D(u(x, t), t), D(u(x, t), x, 2)))
except NotImplementedError as err:
    print("(e) heat equation:", type(err).__name__)
