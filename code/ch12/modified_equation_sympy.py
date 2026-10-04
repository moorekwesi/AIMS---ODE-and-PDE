# Modified equations of upwind, Lax-Friedrichs and Lax-Wendroff with SymPy
# Applied ODE & PDE with Python, Ch. 12 | (c) 2026 Stephen E. Moore | MIT Licence
import sympy as sp

h, c, nu, s = sp.symbols("h c nu s", positive=True)
X, T = sp.symbols("X T")              # X = d/dx, T = d/dt acting on v
dt = nu * h / c                       # Courant number nu = c dt / h
K = 3                                 # keep terms up to O(h^K)


def ex(z, n=K + 3):
    """Truncated exponential series: the shift operator exp(z)."""
    return sum(z**q / sp.factorial(q) for q in range(n + 1))


def trunc(expr, K):
    """Drop all terms of order > K in the bookkeeping parameter s."""
    expr = sp.expand(expr)
    return sum(expr.coeff(s, k) * s**k for k in range(K + 1))


def modified_equation(L):
    """L(T, X) = 0 is the scheme in operator form (h -> s h). Solve for
    T = P(X) by fixed-point iteration T <- T - L(T, X)/(dL/dT at s=0)."""
    P = -c * X                        # leading order: the PDE itself
    for _ in range(K + 1):
        P = trunc(P - L.subs(T, P), K)    # dL/dT = 1 at leading order
    return sp.expand(P.subs(s, 1))


Es = lambda a: ex(a * s * h * X)          # spatial shift E^a
Et = ex(s * dt * T)                       # time shift
schemes = {
    "upwind": (Et - 1) / (s * dt) + c * (1 - Es(-1)) / (s * h),
    "Lax-Friedrichs": (Et - (Es(1) + Es(-1)) / 2) / (s * dt)
                      + c * (Es(1) - Es(-1)) / (2 * s * h),
    "Lax-Wendroff": (Et - 1) / (s * dt) + c * (Es(1) - Es(-1)) / (2 * s * h)
                    - c**2 * dt * (Es(1) - 2 + Es(-1)) / (2 * s * h**2),
}
for name, L in schemes.items():
    P = modified_equation(sp.expand(L))
    print(f"{name}:  v_t + c v_x = a2 v_xx + a3 v_xxx + a4 v_xxxx + ...")
    for p in range(2, K + 2):
        print(f"    a{p} = {sp.sstr(sp.factor(P.coeff(X, p)))}")
