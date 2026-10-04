# The backward heat equation amplifies noise
# Applied ODE & PDE with Python, Ch. 5 | (c) 2026 Stephen E. Moore | MIT Licence
import numpy as np
import matplotlib.pyplot as plt

T, J = 0.02, 200                          # final time, number of grid intervals
x = np.linspace(0, 1, J + 1)
f_true = np.sin(np.pi*x) + 0.5*np.sin(3*np.pi*x) + 0.2*np.sin(6*np.pi*x)


def sine_coeffs(v, nmax):
    """b_n = 2 int_0^1 v(x) sin(n pi x) dx, n = 1..nmax (trapezoidal rule)."""
    n = np.arange(1, nmax + 1)[:, None]
    return 2*np.trapz(v*np.sin(n*np.pi*x), x, axis=1)


def synth(b):
    n = np.arange(1, len(b) + 1)[:, None]
    return np.sum(b[:, None]*np.sin(n*np.pi*x), axis=0)


# forward problem: u(x,T) from u(x,0) = f_true   (well posed)
b0 = sine_coeffs(f_true, 40)
lam = (np.arange(1, 41)*np.pi)**2
g = synth(b0*np.exp(-lam*T))
rng = np.random.default_rng(0)
g_noisy = g + 1e-6*rng.standard_normal(x.size)      # measurement noise

# backward problem: undo the decay mode by mode, keeping M modes
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(x, f_true, "k", lw=2, label="true u(x,0)")
print("  M   amplification e^(M^2 pi^2 T)   max error in u(x,0)")
for M in [3, 6, 7, 8, 10, 12]:
    bT = sine_coeffs(g_noisy, M)
    f_rec = synth(bT*np.exp(lam[:M]*T))
    err = np.max(np.abs(f_rec - f_true))
    print(f"{M:3d}   {np.exp(lam[M-1]*T):12.3e}                {err:.3e}")
    if M in (6, 10):
        ax.plot(x, f_rec, "--", label=f"reconstruction, M = {M}")
ax.set_ylim(-2, 2.5); ax.set_xlabel("x"); ax.set_ylabel("u(x,0)")
ax.legend(); plt.tight_layout()
plt.savefig("ch05_backward_heat.pdf", bbox_inches="tight")
