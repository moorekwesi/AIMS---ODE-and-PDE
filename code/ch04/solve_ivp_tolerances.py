# Accuracy and cost of solve_ivp for different tolerances and methods
# Applied ODE & PDE with Python, Ch. 4 | (c) 2026 Stephen E. Moore | MIT Licence
import numpy as np
from scipy.integrate import solve_ivp

# y' = y cos t, y(0) = 1 on [0, 20]; exact solution exp(sin t)
f = lambda t, y: y * np.cos(t)
T = 20.0
print("method   rtol    nfev   steps   error at T")
for method in ["RK23", "RK45", "DOP853"]:
    for rtol in [1e-3, 1e-6, 1e-9]:
        sol = solve_ivp(f, (0, T), [1.0], method=method, rtol=rtol, atol=1e-3 * rtol)
        err = abs(sol.y[0, -1] - np.exp(np.sin(T)))
        print(f"{method:7s} {rtol:6.0e}  {sol.nfev:5d}  {sol.t.size - 1:5d}    {err:.2e}")

# dense output: a continuous interpolant at no extra cost
sol = solve_ivp(f, (0, T), [1.0], rtol=1e-8, atol=1e-11, dense_output=True)
tt = np.linspace(0, T, 1001)
print("RK45 with rtol=1e-8: max error of dense output on 1001 points:",
      f"{np.max(abs(sol.sol(tt)[0] - np.exp(np.sin(tt)))):.2e}")
