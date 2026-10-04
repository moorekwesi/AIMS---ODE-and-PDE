# Von Neumann analysis with SymPy: amplification factors of eight schemes
# Applied ODE & PDE with Python, Ch. 12 | (c) 2026 Stephen E. Moore | MIT Licence
import sympy as sp

r, nu, S = sp.symbols("r nu S", positive=True)    # S = sin^2(theta/2)
C, Sn = sp.symbols("cos(theta) sin(theta)", real=True)
g = sp.symbols("g")
E = C + sp.I * Sn                                 # e^{i theta}; 1/E = conj(E)


def symbol(stencil):
    """Substitute U_j^n = g^n e^{i j theta} into sum c_ab U_{j+a}^{n+b}."""
    return sum(cf * g**b * (E if a > 0 else sp.conjugate(E))**abs(a)
               for (a, b), cf in stencil.items())


def in_S(expr):
    """Use sin^2 = 1 - cos^2 and cos(theta) = 1 - 2S; then factor."""
    num, den = sp.fraction(sp.together(sp.expand(expr)))
    red = lambda p: sp.rem(sp.expand(p), Sn**2 + C**2 - 1, Sn).subs(C, 1 - 2*S)
    return sp.factor(red(num) / red(den))


# Two-level schemes, multiplied through by dt (r = kappa dt/h^2, nu = c dt/h)
two_level = {
    "FTCS": {(0, 1): 1, (0, 0): -1 + 2*r, (1, 0): -r, (-1, 0): -r},
    "BTCS": {(0, 1): 1 + 2*r, (1, 1): -r, (-1, 1): -r, (0, 0): -1},
    "Crank-Nicolson": {(0, 1): 1 + r, (1, 1): -r/2, (-1, 1): -r/2,
                       (0, 0): -1 + r, (1, 0): -r/2, (-1, 0): -r/2},
    "upwind": {(0, 1): 1, (0, 0): -1 + nu, (-1, 0): -nu},
    "Lax-Friedrichs": {(0, 1): 1, (1, 0): (nu - 1)/2, (-1, 0): (-nu - 1)/2},
    "Lax-Wendroff": {(0, 1): 1, (0, 0): -1 + nu**2, (1, 0): nu/2 - nu**2/2,
                     (-1, 0): -nu/2 - nu**2/2},
}
for name, st in two_level.items():
    G = sp.solve(symbol(st), g)[0]
    print(f"{name:15s} g = {sp.sstr(in_S(sp.re(G)) + sp.I*sp.factor(sp.im(G)))}")
    print(f"{'':15s} 1 - |g|^2 = {sp.sstr(in_S(1 - G * sp.conjugate(G)))}")

# Three-level schemes give a quadratic equation for g
lf_t = {(0, 1): 1, (0, -1): -1, (1, 0): nu, (-1, 0): -nu}            # transport
lf_w = {(0, 1): 1, (0, -1): 1, (0, 0): -2 + 2*nu**2, (1, 0): -nu**2,
        (-1, 0): -nu**2}                                             # wave
for name, st in [("leapfrog (transport)", lf_t), ("leapfrog (wave)", lf_w)]:
    q = sp.expand(symbol(st) * g).subs(C, 1 - 2*S)
    print(f"{name}: {sp.sstr(sp.collect(sp.expand(q), g))} = 0")
