# Common pitfalls: views, integer arrays and off-by-one grids
# Applied ODE & PDE with Python, App. A | (c) 2026 Stephen E. Moore | MIT Licence
import numpy as np

# 1. Slices are VIEWS: changing the slice changes the original array
u = np.zeros(5)
inner = u[1:-1]
inner[:] = 7.0
print("1. after editing a slice, u =", u)
w = u.copy(); w[:] = 0.0                       # a copy is independent
print("   after editing a copy,  u =", u)

# 2. Assignment does not copy: unew = u makes two names for ONE array
u = np.array([0.0, 1.0, 0.0]); unew = u
unew[1] = 0.5
print("2. u =", u, " (u changed as well!)")

# 3. Integer arrays truncate floating-point values silently
a = np.array([1, 2, 3])                        # dtype int
a[0] = 0.9
print("3. integer array after a[0] = 0.9:", a)
print("   a / 2 =", a / 2, "  but a // 2 =", a // 2)

# 4. Off-by-one: N subintervals need N + 1 points
N = 4
print("4. np.linspace(0, 1, N)     ->", np.linspace(0, 1, N), " (h = 1/3, wrong)")
print("   np.linspace(0, 1, N + 1) ->", np.linspace(0, 1, N + 1), " (h = 1/4)")
print("   np.arange(0, 1, 0.25)    ->", np.arange(0, 1, 0.25), " (endpoint missing)")

# 5. Floating-point comparison of times
t = 0.0
for _ in range(10):
    t += 0.1
print(f"5. t == 1.0 ? {t == 1.0}   t = {t:.17f}   np.isclose(t, 1.0) = {np.isclose(t, 1.0)}")
