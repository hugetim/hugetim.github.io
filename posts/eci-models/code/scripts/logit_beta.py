"""Logit-centred beta (Nov 13, 2026, Tim): Beta(a, b) with a + b = kappa and psi(a) - psi(b) = logit(mu), so that
logit(mu) = E[logit X] and a single observation x has maximum-likelihood location mu-hat = x (mu is the "best-fit"
score; in the XB, the best-fit latent score). No closed form: a is found by Newton's method in v = logit(a / kappa),
where g(v) = psi(a) - psi(kappa - a) - logit(mu) is increasing, with g'(v) = (psi1(a) + psi1(b)) a b / kappa.
Start: v = logit(mu) (the mean parameterization's a = mu kappa). Steps are capped at +-3 in v.

solve_a: numpy version (scoring scripts, and inside the model as a forward-only Op). pt_solve_a: PyTensor version for
the model: solve_a gives the root without gradients, then one more Newton step carries the exact implicit-function
gradient (g = 0 at the solution, so d a = -(dg / d theta) / (dg / d a) for theta = mu, kappa).
"""
import numpy as np

N_ITER = 25  # 20 already reach 1e-15 over mu in [1e-6, 1 - 1e-6], kappa in [0.3, 3000]
MAX_STEP = 3.0


def np_trigamma(x):
    """psi1(x) as pt_trigamma below, in numpy (scipy's polygamma(1, .) is several times slower)."""
    s = 0.0
    for i in range(6):
        s = s + 1 / (x + i) ** 2
    z = x + 6; iz = 1 / z; iz2 = iz * iz
    return s + iz + iz2 / 2 + iz * iz2 * (1 / 6 - iz2 * (1 / 30 - iz2 * (1 / 42 - iz2 / 30)))


def solve_a(mu, kappa, n_iter=N_ITER, tol=1e-13, v0=None, return_v=False):
    """Newton iterations, stopping early once every step in v is below tol. v0: optional start (e.g. the previous
    call's solution, as the model's Op does); any start converges, it only changes how many iterations it takes."""
    from scipy.special import digamma, logit, expit
    mu, kappa = np.broadcast_arrays(np.asarray(mu, float), np.asarray(kappa, float))
    L = logit(mu)
    v = L.copy() if v0 is None or np.shape(v0) != L.shape else np.array(v0, float)
    for _ in range(n_iter):
        a = kappa * expit(v); b = kappa - a
        g = digamma(a) - digamma(b) - L
        gp = (np_trigamma(a) + np_trigamma(b)) * a * b / kappa
        step = np.clip(g / gp, -MAX_STEP, MAX_STEP)
        v = v - step
        if np.abs(step).max() < tol:
            break
    a = kappa * expit(v)
    return (a, v) if return_v else a


def pt_trigamma(x):
    """psi1(x) from elementwise arithmetic (PyTensor's tri_gamma runs ~1,000x slower than digamma): recurrence
    psi1(x) = psi1(x + 1) + 1 / x^2 six times, then the asymptotic series at z = x + 6 >= 6 (relative error < 1e-10)."""
    s = 0.0
    for i in range(6):
        s = s + 1 / (x + i) ** 2
    z = x + 6; iz = 1 / z; iz2 = iz * iz
    return s + iz + iz2 / 2 + iz * iz2 * (1 / 6 - iz2 * (1 / 30 - iz2 * (1 / 42 - iz2 / 30)))


_SOLVE_OP = None


def _solve_op():
    """solve_a as a PyTensor Op (forward only, numpy/scipy inside). Unrolling the Newton loop in the graph instead made
    PyTensor's fused graph ~3,000x slower than the plain XB per gradient (1.8 s vs 0.6 ms)."""
    global _SOLVE_OP
    if _SOLVE_OP is None:
        import pytensor.tensor as pt
        from pytensor.graph.basic import Apply
        from pytensor.graph.op import Op

        class SolveA(Op):
            """Root a of psi(a) - psi(kappa - a) = logit(mu). Its gradient is set to zero: PyMC's graph rewrites can strip
            disconnected_grad, and zero is exact for the way pt_solve_a uses it (the Newton step after it has derivative
            1 - g'(a) / g'(a) = 0 in its starting point at the root)."""
            def make_node(self, m, k):
                m, k = pt.as_tensor_variable(m), pt.as_tensor_variable(k)
                return Apply(self, [m, k], [pt.dvector()])

            last_v = None   # warm start from the previous call (sampler steps are close); n_iter caps a bad start

            def perform(self, node, inputs, outputs):
                a, v = solve_a(inputs[0], inputs[1], n_iter=60, v0=self.last_v, return_v=True)
                self.last_v = v
                outputs[0][0] = a

            def grad(self, inputs, output_grads):
                return [pt.zeros_like(inputs[0]), pt.zeros_like(inputs[1])]

            def L_op(self, inputs, outputs, output_grads):
                return self.grad(inputs, output_grads)

        class Digamma(Op):
            """digamma (scipy) whose gradient is pt_trigamma: PyTensor's own digamma differentiates through its slow
            tri_gamma (~65 ms per 2,787 values), which dominated the gradient of the Newton step in pt_solve_a."""
            def make_node(self, x):
                x = pt.as_tensor_variable(x)
                return Apply(self, [x], [pt.dvector()])

            def perform(self, node, inputs, outputs):
                from scipy.special import digamma
                outputs[0][0] = digamma(inputs[0])

            def L_op(self, inputs, outputs, output_grads):
                return [output_grads[0] * pt_trigamma(inputs[0])]

        _SOLVE_OP = (SolveA(), Digamma())
    return _SOLVE_OP


def pt_solve_a(mu, kappa):
    import pytensor.tensor as pt
    from pytensor.gradient import disconnected_grad as sg
    L = pt.log(mu) - pt.log1p(-mu)
    solve, digamma = _solve_op()
    k0 = sg(kappa)
    a_s = sg(solve(pt.cast(sg(mu), "float64"), pt.cast(k0, "float64")))
    # one Newton step in a, with gradients through g (mu via L, and kappa) but not through the slope
    g = digamma(a_s) - digamma(pt.cast(kappa, "float64") - a_s) - L
    ga = sg(pt_trigamma(a_s) + pt_trigamma(k0 - a_s))
    return a_s - g / ga
