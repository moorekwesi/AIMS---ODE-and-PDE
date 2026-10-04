# Burgers' equation: conservative and non-conservative schemes, shocks and fans
# Applied ODE & PDE with Python, Ch. 11 | (c) 2026 Stephen E. Moore | MIT Licence
import numpy as np
import matplotlib.pyplot as plt

f = lambda u: 0.5 * u**2
xa, xb, N = -1.0, 3.0, 200
h = (xb - xa) / N
x = xa + (np.arange(N) + 0.5) * h               # cell centres


def flux(UL, UR, scheme, lam):
    """Numerical flux F_{j+1/2} from the states left (UL) and right (UR) of the face."""
    if scheme == "upwind":                      # valid only when u >= 0
        return f(UL)
    if scheme == "Roe":                         # upwind by the sign of the mean speed
        return np.where(UL + UR >= 0, f(UL), f(UR))
    if scheme == "Lax-Friedrichs":
        return 0.5 * (f(UL) + f(UR)) - 0.5 / lam * (UR - UL)
    if scheme == "Godunov":                     # exact Riemann solution at the face
        return np.maximum(f(np.maximum(UL, 0)), f(np.minimum(UR, 0)))


def solve(uL, uR, T, scheme, conservative=True):
    U = np.where(x < 0, uL, uR).astype(float)
    dt = 0.8 * h / max(abs(uL), abs(uR)); nsteps = int(np.ceil(T / dt)); dt = T / nsteps
    lam = dt / h
    for n in range(nsteps):
        Ue = np.r_[U[0], U, U[-1]]              # constant extrapolation (ghost cells)
        if conservative:
            F = flux(Ue[:-1], Ue[1:], scheme, lam)
            U = U - lam * (F[1:] - F[:-1])
        else:                                   # u_t + u u_x = 0 discretised directly
            U = U - lam * U * (U - Ue[:-2])
    return U


def shock_position(U, level):
    j = np.argmax(U < level)
    return x[j - 1] + h * (U[j - 1] - level) / (U[j - 1] - U[j])


print("shock tests: position of the shock at time T")
print("scheme                     uL=1,uR=0,T=2   uL=2,uR=1,T=1")
print(f"{'exact (Rankine-Hugoniot)':26s}   1.0000          1.5000")
for scheme, cons in (("upwind", True), ("Lax-Friedrichs", True), ("Godunov", True),
                     ("upwind", False)):
    pA = shock_position(solve(1, 0, 2.0, scheme, cons), 0.5)
    pB = shock_position(solve(2, 1, 1.0, scheme, cons), 1.5)
    name = scheme + ("" if cons else " (non-conservative)")
    print(f"{name:26s}   {round(pA, 8) + 0.0:.4f}          {pB:.4f}")

print("transonic rarefaction uL=-1, uR=1, T=0.8: max error")
ex = np.clip(x / 0.8, -1, 1)
for scheme in ("Roe", "Lax-Friedrichs", "Godunov"):
    print(f"  {scheme:15s} {np.max(np.abs(solve(-1, 1, 0.8, scheme) - ex)):.3f}")

fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
ax[0].plot(x, np.where(x < 1, 1.0, 0.0), "k-", label="exact")
for scheme, cons in (("Lax-Friedrichs", True), ("Godunov", True), ("upwind", False)):
    ax[0].plot(x, solve(1, 0, 2.0, scheme, cons), ".-", ms=3,
               label=scheme + ("" if cons else ", non-conservative"))
ax[0].set_title("shock, uL = 1, uR = 0, t = 2"); ax[0].set_xlim(-0.5, 2)
ax[1].plot(x, ex, "k-", label="exact")
for scheme in ("Roe", "Lax-Friedrichs", "Godunov"):
    ax[1].plot(x, solve(-1, 1, 0.8, scheme), ".-", ms=3, label=scheme)
ax[1].set_title("transonic rarefaction, t = 0.8"); ax[1].set_xlim(-1, 2)
for a in ax:
    a.set_xlabel("x"); a.legend(fontsize=8)
plt.tight_layout(); plt.savefig("ch11_burgers_schemes.pdf", bbox_inches="tight")
plt.close()
