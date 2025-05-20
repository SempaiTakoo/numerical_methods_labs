import math


def is_equal_with_accuracy(a, b, rel_tol=1e-9, abs_tol=1e-9) -> bool:
    return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
