# Linear versus nonlinear pendulum: periods
# Applied ODE & PDE with Python, Ch. 4 | (c) 2026 Stephen E. Moore | MIT Licence
import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import ellipk

g, L = 9.81, 1.0
w0 = np.sqrt(g / L)                       # linear angular frequency


def nonlinear(t, z):                      # z = (theta, omega)
    return [z[1], -w0**2 * np.sin(z[0])]


def downward_crossing(t, z):              # theta = 0 while moving to negative theta
    return z[0]
downward_crossing.direction = -1


print("theta0(deg)  T_linear   T_numerical   T_exact(elliptic)")
for th0 in [10, 45, 90, 135, 170]:
    a = np.radians(th0)
    sol = solve_ivp(nonlinear, (0, 30), [a, 0.0], events=downward_crossing,
                    rtol=1e-10, atol=1e-12)
    T_num = np.diff(sol.t_events[0]).mean()          # time between crossings
    T_ex = 4 / w0 * ellipk(np.sin(a / 2)**2)           # scipy uses parameter m = k^2
    print(f"{th0:8d}     {2 * np.pi / w0:.5f}    {T_num:.5f}      {T_ex:.5f}")
