"""Demos for Math for ML Lab. Uses ml_toolbox.math_foundations (run from repo root)."""
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def math_dot():
    from ml_toolbox.math_foundations.linear_algebra import Vector
    import numpy as np
    v, w = np.array([1.0, 2.0]), np.array([3.0, 4.0])
    dot = Vector.dot(v, w)
    norm_v = Vector.norm(v)
    out = f"v = [1, 2], w = [3, 4]\ndot(v,w) = {dot}\nnorm(v) = {norm_v}"
    return {"ok": True, "output": out}


def math_svd():
    from ml_toolbox.math_foundations.linear_algebra import svd
    import numpy as np
    A = np.array([[1.0, 2.0], [2.0, 1.0]])
    U, S, Vt = svd(A)
    s = np.diag(S).flatten()
    out = f"A = [[1,2],[2,1]]\nSVD singular values: {s}"
    return {"ok": True, "output": out}


def math_derivative():
    from ml_toolbox.math_foundations.calculus import derivative
    f = lambda x: x ** 2
    d = derivative(f, 3.0)
    out = f"f(x) = x^2\nderivative at x=3: {d}  (expected 6.0)"
    return {"ok": True, "output": out}


def math_gradient():
    from ml_toolbox.math_foundations.calculus import gradient
    import numpy as np
    f = lambda x: np.sum(x ** 2)
    grad = gradient(f, np.array([1.0, 2.0]))
    out = f"f(x) = x1^2 + x2^2\ngradient at [1,2]: {grad}  (expected [2, 4])"
    return {"ok": True, "output": out}


def math_gaussian():
    from ml_toolbox.math_foundations.probability_statistics import Gaussian
    import numpy as np
    g = Gaussian(mean=0, std=1)
    pdf0 = float(g.pdf(np.array([0.0]))[0])
    samples = g.sample(3)
    out = f"Gaussian(0, 1)\npdf(0) ≈ {pdf0:.4f}\nsample(3) = {samples}"
    return {"ok": True, "output": out}


def math_gd():
    from ml_toolbox.math_foundations.optimization import gradient_descent
    import numpy as np
    f = lambda x: np.sum(x ** 2)
    grad_f = lambda x: 2 * x
    x_opt, hist = gradient_descent(f, grad_f, np.array([5.0, 5.0]), learning_rate=0.1, max_iter=200)
    out = f"minimize sum(x^2), x0 = [5, 5]\nminimizer: {x_opt}\niterations: {len(hist)}"
    return {"ok": True, "output": out}


DEMO_HANDLERS = {
    "math_dot": math_dot,
    "math_svd": math_svd,
    "math_derivative": math_derivative,
    "math_gradient": math_gradient,
    "math_gaussian": math_gaussian,
    "math_gd": math_gd,
}


def run_demo(demo_id: str):
    if demo_id not in DEMO_HANDLERS:
        return {"ok": False, "output": "", "error": f"Unknown demo: {demo_id}"}
    try:
        return DEMO_HANDLERS[demo_id]()
    except Exception as e:
        return {"ok": False, "output": "", "error": str(e)}
