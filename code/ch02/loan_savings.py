# Loans and savings with continuous compounding: A' = r A + s
# Applied ODE & PDE with Python, Ch. 2 | (c) 2026 Stephen E. Moore | MIT Licence
import numpy as np


def balance(t, A0, r, s):
    """Solution of A' = r A + s, A(0) = A0 (s > 0 deposit, s < 0 repayment)."""
    return (A0 + s / r) * np.exp(r * t) - s / r


# (a) Loan of GHS 50,000 at 25% per year, repaid continuously over 5 years
A0, r, T = 50_000.0, 0.25, 5.0
p = r * A0 / (1 - np.exp(-r * T))            # repayment rate per year
print(f"(a) repayment rate   = {p:10.2f} GHS/year = {p / 12:9.2f} GHS/month")
print(f"    total repaid     = {p * T:10.2f} GHS, interest = {p * T - A0:9.2f} GHS")
print(f"    balance at T     = {balance(T, A0, r, -p):10.2e} GHS")
# compare: discrete monthly payments with monthly rate r/12
i, n = r / 12, 12 * T
m = A0 * i / (1 - (1 + i) ** (-n))
print(f"    monthly instalment (discrete formula) = {m:9.2f} GHS")

# (b) Savings: how much per year for GHS 100,000 after 10 years at 12%?
r2, T2, target = 0.12, 10.0, 100_000.0
s = target * r2 / (np.exp(r2 * T2) - 1)
print(f"(b) deposit rate     = {s:10.2f} GHS/year = {s / 12:9.2f} GHS/month")
print(f"    check A(10)      = {balance(T2, 0.0, r2, s):10.2f} GHS")
# (c) the "rule of 70": doubling time ln 2 / r
for rr in (0.05, 0.12, 0.25):
    print(f"(c) r = {rr:4.2f}: doubling time {np.log(2) / rr:6.3f} years "
          f"(rule of 70: {0.70 / rr:6.3f})")
