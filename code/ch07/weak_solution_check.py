# Testing candidate weak solutions of Riemann problems against a test function
# Applied ODE & PDE with Python, Ch. 7 | (c) 2026 Stephen E. Moore | MIT Licence
import numpy as np

L, T = 8.0, 6.0                                  # truncated domain [-L, L] x [0, T]
gx, gw = np.polynomial.legendre.leggauss(60)     # Gauss-Legendre rule on [-1, 1]


def phi(x, t):
    """Smooth test function, negligible outside a neighbourhood of (0.5, 1)."""
    return np.exp(-((x - 0.5)**2 + (t - 1.0)**2) / 0.5)


def phi_x(x, t):
    return -2 * (x - 0.5) / 0.5 * phi(x, t)


def phi_t(x, t):
    return -2 * (t - 1.0) / 0.5 * phi(x, t)


def integrate_x(g, breaks):
    """Integral of g over [-L, L], split at the points where g is not smooth."""
    pts = [-L] + sorted(breaks) + [L]
    total = 0.0
    for a, b in zip(pts[:-1], pts[1:]):
        x = 0.5 * (b - a) * gx + 0.5 * (a + b)
        total += 0.5 * (b - a) * np.dot(gw, g(x))
    return total


def weak_residual(u, breaks, q, f):
    """R = int int (q(u) phi_t + f(u) phi_x) dx dt + int q(u(x,0)) phi(x,0) dx."""
    tn = 0.5 * T * (np.polynomial.legendre.leggauss(200)[0] + 1)
    tw = 0.5 * T * np.polynomial.legendre.leggauss(200)[1]
    R = sum(w * integrate_x(lambda x: q(u(x, t)) * phi_t(x, t)
                            + f(u(x, t)) * phi_x(x, t), breaks(t))
            for t, w in zip(tn, tw))
    return R + integrate_x(lambda x: q(u(x, 0.0)) * phi(x, 0.0), [0.0])


# Burgers in the form  u_t + (u^2/2)_x = 0
q1, f1 = (lambda u: u), (lambda u: u**2 / 2)
# the "same" equation multiplied by 2u:  (u^2)_t + (2u^3/3)_x = 0
q2, f2 = (lambda u: u**2), (lambda u: 2 * u**3 / 3)


def shock(ul, ur, s):
    return (lambda x, t: np.where(x < s * t, ul, ur)), (lambda t: [s * t])


fan = (lambda x, t: np.clip(x / max(t, 1e-300), 0.0, 1.0)), (lambda t: [0.0, t])

print("Riemann data u_l = 0, u_r = 1   (form u_t + (u^2/2)_x = 0)")
for name, (u, br) in [("rarefaction fan       ", fan),
                      ("shock with s = 1/2    ", shock(0, 1, 0.5)),
                      ("shock with s = 0.3    ", shock(0, 1, 0.3))]:
    print(f"   {name} residual = {weak_residual(u, br, q1, f1): .2e}")

print("Riemann data u_l = 1, u_r = 0")
for name, s in [("shock with s = 1/2", 0.5), ("shock with s = 2/3", 2 / 3)]:
    u, br = shock(1, 0, s)
    print(f"   {name}: residual form 1 = {weak_residual(u, br, q1, f1): .2e},"
          f"  form 2 = {weak_residual(u, br, q2, f2): .2e}")
