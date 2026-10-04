# Good practice: a documented solver tested against an exact solution
# Applied ODE & PDE with Python, App. A | (c) 2026 Stephen E. Moore | MIT Licence
import numpy as np

def euler(f, t0, y0, T, N):
    """Solve y' = f(t, y), y(t0) = y0 on [t0, T] with N steps of Euler's method.

    Parameters
    ----------
    f : callable f(t, y) returning a float or a NumPy array
    t0, T : float, initial and final time
    y0 : float or array, initial value
    N : int, number of steps (step size h = (T - t0) / N)

    Returns
    -------
    t : array of shape (N + 1,), y : array of shape (N + 1, ...)
    """
    t = np.linspace(t0, T, N + 1)
    h = (T - t0) / N
    y = np.zeros((N + 1,) + np.shape(y0))
    y[0] = y0
    for n in range(N):
        y[n + 1] = y[n] + h * f(t[n], y[n])
    return t, y

def test_euler_order():
    """Euler's method is first order: halving h should halve the error."""
    f = lambda t, y: -2 * t * y + t                 # exact y = 1/2 + 3/2 exp(-t^2)
    exact = lambda t: 0.5 + 1.5 * np.exp(-t**2)
    errors = []
    for N in (100, 200, 400, 800):
        t, y = euler(f, 0.0, 2.0, 3.0, N)
        errors.append(np.max(np.abs(y - exact(t))))
        print(f"N = {N:4d}   max error = {errors[-1]:.4e}")
    rates = np.log2(np.array(errors[:-1]) / np.array(errors[1:]))
    print("observed orders:", np.round(rates, 3))
    assert np.all(np.abs(rates - 1.0) < 0.05), "Euler should be first order"

if __name__ == "__main__":
    test_euler_order()
    print("test passed")
