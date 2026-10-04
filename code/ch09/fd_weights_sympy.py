# Finite difference weights by undetermined coefficients and by Fornberg
# Applied ODE & PDE with Python, Ch. 9 | (c) 2026 Stephen E. Moore | MIT Licence
import sympy as sp

h = sp.symbols("h", positive=True)
d = sp.symbols("f0:12")          # d[q] stands for the q-th derivative f^(q)(x)


def fd_weights(offsets, m):
    """Weights w_k with  sum_k w_k f(x + s_k h) = h^m f^(m)(x) + ...
    found by matching Taylor coefficients (undetermined coefficients)."""
    n = len(offsets)
    w = sp.symbols(f"w0:{n}")
    # moment equations: sum_k w_k s_k^q / q! = delta_{q,m},  q = 0..n-1
    eqs = [sum(w[k] * sp.Integer(offsets[k])**q for k in range(n))
           / sp.factorial(q) - (1 if q == m else 0) for q in range(n)]
    sol = sp.solve(eqs, w)
    return [sol[wk] for wk in w]


def leading_error(offsets, weights, m, qmax=10):
    """Leading term of  (1/h^m) sum_k w_k f(x + s_k h) - f^(m)(x)."""
    taylor = lambda s: sum((s * h)**q / sp.factorial(q) * d[q]
                           for q in range(qmax + 1))
    err = sp.expand(sum(wk * taylor(sk) for wk, sk in zip(weights, offsets))
                    / h**m - d[m])
    p = min(sp.Poly(err, h).monoms())[0]          # lowest power of h
    return p, sp.factor(err.coeff(h, p)) * h**p


cases = [("f'  central  ", [-1, 0, 1], 1),
         ("f'  one-sided", [0, 1, 2], 1),
         ("f'' central  ", [-1, 0, 1], 2),
         ("f'' five-pt  ", [-2, -1, 0, 1, 2], 2),
         ("f'' one-sided", [0, 1, 2, 3], 2)]
for name, s, m in cases:
    w = fd_weights(s, m)
    p, lead = leading_error(s, w, m)
    print(f"{name} offsets {s}")
    print(f"    weights {w},  order {p},  error ~ {sp.sstr(lead)}")

# The same weights from Fornberg's algorithm (all derivatives at once)
W = sp.finite_diff_weights(2, [-2, -1, 0, 1, 2], 0)
print("Fornberg, 5 points, 1st derivative:", W[1][-1])
print("Fornberg, 5 points, 2nd derivative:", W[2][-1])
