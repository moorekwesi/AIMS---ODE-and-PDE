# Automatic local truncation errors of finite difference schemes with SymPy
# Applied ODE & PDE with Python, Ch. 12 | (c) 2026 Stephen E. Moore | MIT Licence
import sympy as sp

h, dt, s = sp.symbols("h dt s", positive=True)
kap, c, r, nu = sp.symbols("kappa c r nu", positive=True)
X, T = sp.symbols("X T")          # placeholders for the operators d/dx and d/dt


def truncation_error(stencil, pde, t0=0, order=2, wt=1, eliminate=True):
    """Local truncation error of the linear scheme  sum c_ab U_{j+a}^{n+b} = 0.

    stencil : {(a, b): coefficient}, scaled so that the scheme approximates P(u)
    pde     : the PDE as a polynomial P(X, T) in X = d/dx, T = d/dt
    t0      : expand about t^{n+t0}  (t0 = 1/2 for Crank-Nicolson)
    order   : keep terms with  wt*deg(dt) + deg(h) <= order  (wt = 2: dt ~ h^2)
    eliminate: if True use P(u) = 0 to remove time derivatives (T^k)
    """
    tau = 0
    for (a, b), coef in stencil.items():
        z = a * h * X + (b - t0) * dt * T       # u(x+ah, t+(b-t0)dt) = exp(z) u
        tau += coef * sum(z**q / sp.factorial(q) for q in range(order + 2*wt + 3))
    tau = sp.expand(tau.subs({h: s * h, dt: s**wt * dt}))
    tau = sum(t for t in sp.Add.make_args(tau)
              if t.as_coeff_exponent(s)[1] <= order).subs(s, 1)
    if eliminate:
        return sp.rem(sp.expand(tau), pde, T)    # e.g. T -> kappa X^2
    return sp.expand(tau - pde)


def as_derivatives(expr, subs=None):
    """Write X^p T^q as the symbol u_t..tx..x and factor each coefficient."""
    out = 0
    for (p, q), coef in sp.Poly(sp.expand(expr), X, T).terms():
        coef = sp.factor(coef.subs(subs or {}))
        out += coef * sp.Symbol("u_" + "t" * q + "x" * p)
    return out


r2, th = 1 / h**2, 1 / dt
heat, transport = T - kap * X**2, T + c * X
R, NU = {dt: r * h**2 / kap}, {dt: nu * h / c}       # r = kappa dt/h^2, nu = c dt/h
half = sp.Rational(1, 2)
schemes = {   # name: (stencil, PDE, t0, order, wt, substitution)
 "FTCS": ({(0, 1): th, (0, 0): -th + 2*kap*r2, (1, 0): -kap*r2,
           (-1, 0): -kap*r2}, heat, 0, 2, 2, R),
 "BTCS": ({(0, 1): th + 2*kap*r2, (1, 1): -kap*r2, (-1, 1): -kap*r2,
           (0, 0): -th}, heat, 1, 2, 2, R),
 "Crank-Nicolson": ({(0, 1): th + kap*r2, (1, 1): -kap*r2/2, (-1, 1): -kap*r2/2,
                     (0, 0): -th + kap*r2, (1, 0): -kap*r2/2, (-1, 0): -kap*r2/2},
                    heat, half, 2, 1, None),
 "upwind": ({(0, 1): th, (0, 0): -th + c/h, (-1, 0): -c/h}, transport, 0, 1, 1, NU),
 "Lax-Friedrichs": ({(0, 1): th, (1, 0): -th/2 + c/(2*h),
                     (-1, 0): -th/2 - c/(2*h)}, transport, 0, 1, 1, NU),
 "Lax-Wendroff": ({(0, 1): th, (0, 0): -th + c**2*dt*r2,
                   (1, 0): c/(2*h) - c**2*dt*r2/2,
                   (-1, 0): -c/(2*h) - c**2*dt*r2/2}, transport, 0, 2, 1, NU),
 "leapfrog (transport)": ({(0, 1): th/2, (0, -1): -th/2, (1, 0): c/(2*h),
                           (-1, 0): -c/(2*h)}, transport, 0, 2, 1, NU),
 "leapfrog (wave)": ({(0, 1): th**2, (0, -1): th**2, (0, 0): -2*th**2 + 2*c**2*r2,
                      (1, 0): -c**2*r2, (-1, 0): -c**2*r2},
                     T**2 - c**2*X**2, 0, 2, 1, NU),
}
for name, (st, pde, t0, order, wt, sub) in schemes.items():
    tau = truncation_error(st, pde, t0, order, wt)
    print(f"{name:20s} tau = {sp.sstr(as_derivatives(tau, sub))}")

# Exam problem: 3u_t + 4u_x + u = 0, centred in time and in space
damped = {(0, 1): 3*th/2, (0, -1): -3*th/2, (1, 0): 4/(2*h), (-1, 0): -4/(2*h),
          (0, 0): 1}
tau = truncation_error(damped, 3*T + 4*X + 1, 0, order=2, eliminate=False)
print("3u_t+4u_x+u=0 (CTCS) tau =", sp.sstr(as_derivatives(tau)))
