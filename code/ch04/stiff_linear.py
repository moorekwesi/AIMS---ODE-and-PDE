# A stiff scalar problem: explicit versus implicit Euler and the trapezoidal rule
# Applied ODE & PDE with Python, Ch. 4 | (c) 2026 Stephen E. Moore | MIT Licence
import numpy as np

lam = 1000.0                       # y' = -lam (y - cos t), y(0) = 0


def exact(t):
    """Exact solution: y = lam/(1+lam^2) (lam cos t + sin t) - lam^2/(1+lam^2) exp(-lam t)."""
    return lam / (1 + lam**2) * (lam * np.cos(t) + np.sin(t) - lam * np.exp(-lam * t))


def integrate(method, h, T=2.0):
    """Explicit Euler, backward Euler or trapezoidal rule (implicit steps solved exactly)."""
    N = int(round(T / h))
    y = 0.0
    for n in range(N):
        t, t1 = n * h, (n + 1) * h
        if method == "explicit Euler":
            y = y - h * lam * (y - np.cos(t))
        elif method == "backward Euler":          # y1 = y + h(-lam (y1 - cos t1))
            y = (y + h * lam * np.cos(t1)) / (1 + h * lam)
        else:                                     # trapezoidal rule
            y = ((1 - h * lam / 2) * y + h * lam / 2 * (np.cos(t) + np.cos(t1))) \
                / (1 + h * lam / 2)
    return y


print("    N    h*lam   explicit Euler   backward Euler   trapezoidal  (errors at t=2)")
for N in [20, 200, 950, 1050, 2000]:
    h = 2.0 / N
    e = [abs(integrate(m, h) - exact(2.0)) for m in
         ["explicit Euler", "backward Euler", "trapezoidal"]]
    print(f"{N:5d}  {h * lam:6.3f}   {e[0]:12.3e}    {e[1]:12.3e}   {e[2]:12.3e}")
