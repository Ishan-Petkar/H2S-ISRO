"""Computes the bivariate normal tail probability for Lemma 1."""
import numpy as np
from scipy import integrate
import argparse

def bivariate_normal_pdf(x, y, r):
    """Standard bivariate normal PDF with correlation r."""
    denom = 2 * np.pi * np.sqrt(1 - r**2)
    exponent = -(x**2 - 2*r*x*y + y**2) / (2*(1-r**2))
    return np.exp(exponent) / denom

def compute_tail_probability(r, threshold=2.0):
    """P(X >= threshold, Y <= -threshold) for standard bivariate normal."""
    # Integrate x from threshold to infinity, and y from -infinity to -threshold
    # Using scipy.integrate.dblquad:
    # dblquad(func, gfun, hfun, qfun, rfun)
    # where func is func(y, x), gfun/hfun are limits of y as a function of x,
    # and qfun/rfun are constant limits of x.
    result, err = integrate.dblquad(
        lambda y, x: bivariate_normal_pdf(x, y, r),
        threshold, np.inf,              # x limits: [threshold, inf]
        lambda x: -np.inf,              # y lower limit: -inf
        lambda x: -threshold            # y upper limit: -threshold
    )
    return result, err

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--r", type=float, default=0.225, help="Correlation coefficient")
    parser.add_argument("--threshold", type=float, default=2.0, help="Standardized threshold")
    args = parser.parse_args()

    prob, err = compute_tail_probability(args.r, args.threshold)
    print(f"P_FP for r={args.r} and threshold={args.threshold}: {prob:.6f} (error: {err:.2e})")
