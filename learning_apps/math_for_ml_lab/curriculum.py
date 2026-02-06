"""
Curriculum: Basic Math for Machine Learning.
Linear algebra, calculus, probability & statistics, optimization. Uses ml_toolbox.math_foundations.
"""
from typing import Dict, Any, List

LEVELS = ["basics", "intermediate", "advanced"]

BOOKS = [
    {"id": "linear_algebra", "name": "Linear Algebra", "short": "Linear Alg", "color": "#2563eb"},
    {"id": "calculus", "name": "Calculus", "short": "Calculus", "color": "#059669"},
    {"id": "probability", "name": "Probability & Statistics", "short": "Prob & Stats", "color": "#7c3aed"},
    {"id": "optimization", "name": "Optimization", "short": "Optimization", "color": "#dc2626"},
]

CURRICULUM: List[Dict[str, Any]] = [
    {"id": "math_dot_norm", "book_id": "linear_algebra", "level": "basics", "title": "Vectors: Dot Product & Norm",
     "learn": "Dot product v·w; L2 norm ||v||. Used everywhere in ML (similarity, loss). Vector.dot, Vector.norm, Vector.normalize.",
     "try_code": "from ml_toolbox.math_foundations.linear_algebra import Vector\nimport numpy as np\nv, w = np.array([1., 2.]), np.array([3., 4.])\nprint('dot:', Vector.dot(v, w), 'norm:', Vector.norm(v))",
     "try_demo": "math_dot"},
    {"id": "math_matrix", "book_id": "linear_algebra", "level": "basics", "title": "Matrices: Multiply, Transpose, Inverse",
     "learn": "Matrix multiply A@B, transpose A.T, inverse for solving linear systems. Matrix.multiply, Matrix.transpose, Matrix.inverse.",
     "try_code": "from ml_toolbox.math_foundations.linear_algebra import Matrix\nimport numpy as np\nA = np.array([[1., 2.], [3., 4.]])\nprint('det:', Matrix.determinant(A), 'rank:', Matrix.rank(A))",
     "try_demo": None},
    {"id": "math_svd", "book_id": "linear_algebra", "level": "intermediate", "title": "SVD (Singular Value Decomposition)",
     "learn": "A = U S V^T. Used in PCA, recommendation, low-rank approx. svd(A) returns U, S, Vt.",
     "try_code": "from ml_toolbox.math_foundations.linear_algebra import svd\nimport numpy as np\nA = np.array([[1., 2.], [2., 1.]])\nU, S, Vt = svd(A)\nprint('singular values:', np.diag(S).flatten())",
     "try_demo": "math_svd"},
    {"id": "math_eigen", "book_id": "linear_algebra", "level": "intermediate", "title": "Eigendecomposition",
     "learn": "A v = λ v. Eigenvalues and eigenvectors; used in PCA, spectral methods. eigendecomposition(A).",
     "try_code": "from ml_toolbox.math_foundations.linear_algebra import eigendecomposition\nimport numpy as np\nA = np.array([[1., 2.], [2., 1.]])\neigenvalues, eigenvectors = eigendecomposition(A)\nprint('eigenvalues:', eigenvalues)",
     "try_demo": None},
    {"id": "math_derivative", "book_id": "calculus", "level": "basics", "title": "Derivatives (Numerical)",
     "learn": "f'(x) ≈ (f(x+h)-f(x-h))/(2h). Foundation for gradient-based learning.",
     "try_code": "from ml_toolbox.math_foundations.calculus import derivative\nf = lambda x: x**2\nprint('d/dx x^2 at x=3:', derivative(f, 3.0))",
     "try_demo": "math_derivative"},
    {"id": "math_gradient", "book_id": "calculus", "level": "basics", "title": "Gradient (Multivariate)",
     "learn": "∇f(x) = [∂f/∂x1, ...]. Numerical gradient for any f: R^n → R. Used in backprop.",
     "try_code": "from ml_toolbox.math_foundations.calculus import gradient\nimport numpy as np\nf = lambda x: np.sum(x**2)\nprint('gradient of sum(x^2) at [1,2]:', gradient(f, np.array([1., 2.])))",
     "try_demo": "math_gradient"},
    {"id": "math_hessian", "book_id": "calculus", "level": "intermediate", "title": "Jacobian & Hessian",
     "learn": "Jacobian: matrix of first partials. Hessian: matrix of second partials (curvature). Second-order optimization.",
     "try_code": "from ml_toolbox.math_foundations.calculus import hessian\nimport numpy as np\nf = lambda x: np.sum(x**2)\nprint(hessian(f, np.array([1., 2.])))",
     "try_demo": None},
    {"id": "math_gaussian", "book_id": "probability", "level": "basics", "title": "Gaussian Distribution",
     "learn": "Normal distribution: mean μ, std σ. pdf, cdf, sample. Ubiquitous in ML (noise, priors, many loss assumptions).",
     "try_code": "from ml_toolbox.math_foundations.probability_statistics import Gaussian\nimport numpy as np\ng = Gaussian(mean=0, std=1)\nprint('pdf(0):', g.pdf(np.array([0.]))[0], 'sample(3):', g.sample(3))",
     "try_demo": "math_gaussian"},
    {"id": "math_bayes", "book_id": "probability", "level": "intermediate", "title": "Bayes' Rule",
     "learn": "P(A|B) = P(B|A)P(A)/P(B). Bayesian inference: update prior with likelihood. BayesianInference.bayes_rule.",
     "try_code": "from ml_toolbox.math_foundations.probability_statistics import BayesianInference\np = BayesianInference.bayes_rule(prior=0.5, likelihood=0.8, evidence=0.6)\nprint('posterior:', p)",
     "try_demo": None},
    {"id": "math_mle", "book_id": "probability", "level": "intermediate", "title": "Maximum Likelihood Estimation (MLE)",
     "learn": "Choose θ that maximizes likelihood of data. MLE.fit for Gaussian; general principle for many models.",
     "try_code": "from ml_toolbox.math_foundations.probability_statistics import MLE\nimport numpy as np\nX = np.random.normal(2, 1.5, 100)\nmle = MLE(); mle.fit(X)\nprint('estimated mean, std:', mle.mean_, mle.std_)",
     "try_demo": None},
    {"id": "math_gd", "book_id": "optimization", "level": "basics", "title": "Gradient Descent",
     "learn": "x ← x - η ∇f(x). Minimize f by following the negative gradient. learning_rate, tol, history.",
     "try_code": "from ml_toolbox.math_foundations.optimization import gradient_descent\nimport numpy as np\nf = lambda x: np.sum(x**2)\ngrad_f = lambda x: 2*x\nx_opt, hist = gradient_descent(f, grad_f, np.array([5., 5.]), learning_rate=0.1)\nprint('minimizer:', x_opt)",
     "try_demo": "math_gd"},
    {"id": "math_sgd", "book_id": "optimization", "level": "intermediate", "title": "Stochastic Gradient Descent (SGD)",
     "learn": "Update from mini-batches; faster per epoch, noisier. Default in many deep learning trainers.",
     "try_code": "from ml_toolbox.math_foundations.optimization import stochastic_gradient_descent\nimport numpy as np\n# f, grad_f take (data_batch); data = (X, y) style",
     "try_demo": None},
    {"id": "math_adam", "book_id": "optimization", "level": "intermediate", "title": "Adam Optimizer",
     "learn": "Adaptive learning rates (momentum + RMSprop). Adam optimizer in math_foundations.optimization.",
     "try_code": "from ml_toolbox.math_foundations.optimization import adam_optimizer\nimport numpy as np\nf = lambda x: np.sum(x**2)\ngrad_f = lambda x: 2*x\nx_opt, _ = adam_optimizer(f, grad_f, np.array([1., 1.]))\nprint('Adam minimizer:', x_opt)",
     "try_demo": None},
]


def get_curriculum(): return list(CURRICULUM)
def get_books(): return list(BOOKS)
def get_levels(): return list(LEVELS)
def get_by_book(book_id: str): return [c for c in CURRICULUM if c["book_id"] == book_id]
def get_by_level(level: str): return [c for c in CURRICULUM if c["level"] == level]
def get_item(item_id: str):
    for c in CURRICULUM:
        if c["id"] == item_id: return c
    return None
