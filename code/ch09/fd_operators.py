# Operator calculus: derivatives as power series in difference operators
# Applied ODE & PDE with Python, Ch. 9 | (c) 2026 Stephen E. Moore | MIT Licence
import sympy as sp

z, E = sp.symbols("z E")        # z = Delta, nabla or delta;  E = shift operator

# hD as a series in each difference operator (E = exp(hD)):
#   Delta = E - 1 ,  nabla = 1 - 1/E ,  delta = E^(1/2) - E^(-1/2) = 2 sinh(hD/2)
series = {
    "hD     in Delta": sp.series(sp.log(1 + z), z, 0, 5).removeO(),
    "hD     in nabla": sp.series(-sp.log(1 - z), z, 0, 5).removeO(),
    "hD     in delta": sp.series(2 * sp.asinh(z / 2), z, 0, 6).removeO(),
    "(hD)^2 in delta": sp.series((2 * sp.asinh(z / 2))**2, z, 0, 7).removeO(),
}
for name, s in series.items():
    print(f"{name}: {sp.sstr(sp.Poly(s, z).as_expr(), order='rev-lex')}")


def stencil(poly_in_E):
    """Coefficients c_k of sum_k c_k E^k, i.e. weights of f(x + k h)."""
    w = {}
    for term in sp.Add.make_args(sp.expand(poly_in_E)):
        c, k = term.as_coeff_exponent(E)
        w[int(k)] = w.get(int(k), 0) + c
    return dict(sorted(w.items()))


# Truncate (hD)^2 = delta^2 - delta^4/12 + ... and replace delta^2 = E - 2 + 1/E
d2 = E - 2 + 1 / E
print("h^2 f'' ~ delta^2 f              :", stencil(d2))
print("h^2 f'' ~ (delta^2 - delta^4/12) f:", stencil(d2 - d2**2 / 12))
# Truncate hD = Delta - Delta^2/2 (second-order one-sided formula)
print("h f'   ~ (Delta - Delta^2/2) f    :", stencil((E - 1) - (E - 1)**2 / 2))
