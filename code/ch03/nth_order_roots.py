# Characteristic roots of n-th order equations with numpy.roots, and their multiplicities
# Applied ODE & PDE with Python, Ch. 3 | (c) 2026 Stephen E. Moore | MIT Licence
import numpy as np


def group_roots(coeffs, tol=1e-4):
    """Roots of the characteristic polynomial, grouped into (root, multiplicity).
    Roots closer than tol are treated as one multiple root (see the Caution box)."""
    rts = np.roots(coeffs)
    groups = []
    for z in rts:
        for g in groups:
            if abs(z - g[0]) < tol:
                g[1].append(z)
                break
        else:
            groups.append([z, [z]])
    return [(np.mean(g[1]), len(g[1]), max(abs(np.array(g[1]) - np.mean(g[1]))))
            for g in groups]


def basis(groups):
    """Real fundamental set from grouped roots (one entry per conjugate pair)."""
    terms = []
    for z, m, _ in groups:
        a, b = round(z.real, 6) + 0.0, round(z.imag, 6) + 0.0
        if b < 0:
            continue                                  # use the partner with b > 0
        for k in range(m):
            tk = "" if k == 0 else ("t" if k == 1 else f"t^{k}")
            e = "" if a == 0 else f"exp({a:g} t)"
            if b == 0:
                terms.append(" ".join(w for w in (tk, e) if w) or "1")
            else:
                for trig in ("cos", "sin"):
                    terms.append(" ".join(w for w in (tk, e, f"{trig}({b:g} t)") if w))
    return terms


examples = {
    "y''' - y = 0": [1, 0, 0, -1],
    "y'''' - 2y''' + 5y'' - 8y' + 4y = 0": [1, -2, 5, -8, 4],
    "y'''' + 8y'' + 16y = 0": [1, 0, 8, 0, 16],
    "y^(5) - 3y'''' + 3y''' - y'' = 0": [1, -3, 3, -1, 0, 0],
}
for name, c in examples.items():
    print(name)
    groups = group_roots(c)
    for z, m, spread in groups:
        re, im = round(z.real, 6) + 0.0, round(z.imag, 6) + 0.0   # avoid "-0.000000"
        print(f"   root {re:+.6f}{im:+.6f}i  multiplicity {m}"
              f"  (spread of computed roots {spread:.1e})")
    print("   basis:", ", ".join(basis(groups)))
