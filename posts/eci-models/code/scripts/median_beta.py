"""Median-centred beta (Nov 14, 2026, Tim): Beta(a, b) with a + b = kappa and median = mu, i.e. I_mu(a, kappa - a) = 1/2
(I = regularized incomplete beta, scipy.special.betainc). The median rises monotonically with a, so the root is unique.
In the XB, mu is then the median latent score: the score a model beats half the time (before the clamp to [0, 1]).

solve_median: numpy root-finder (scoring scripts, and inside the model as a forward-only Op). Newton's method in
v = logit(a / kappa), started at the logit-centred solution (logit_beta.solve_a, close to the median one) or a warm
start, with dF/dv from central differences of betainc; elements that have not converged after the Newton steps are
finished by bisection.

pt_solve_median: PyTensor version: the root without gradients, then one Newton step a - F / F_a that carries the exact
implicit-function gradient. dI/da and dI/db come from central differences of betainc (relative step 1e-5; stable to ~8
digits across steps 1e-4 to 1e-6), dI/dx is the beta density in closed form; the solver Op returns them with the root
(the last Newton step computes them anyway), so each evaluation costs about five betainc calls.
"""
import numpy as np

H = 1e-5  # relative step for the central differences of betainc in a and b


def inc_beta_parts(a, b, x):
    """I_x(a, b) and its derivatives in a, b (central differences) and x (the beta density)."""
    from scipy.special import betainc, betaln
    I = betainc(a, b, x)
    Ia = (betainc(a * (1 + H), b, x) - betainc(a * (1 - H), b, x)) / (2 * H * a)
    Ib = (betainc(a, b * (1 + H), x) - betainc(a, b * (1 - H), x)) / (2 * H * b)
    Ix = np.exp((a - 1) * np.log(x) + (b - 1) * np.log1p(-x) - betaln(a, b))
    return I, Ia, Ib, Ix


def solve_median(mu, kappa, v0=None, return_v=False, n_newton=30, n_bisect=200, tol=1e-12, return_parts=False):
    from scipy.special import betainc, expit, logit
    mu, kappa = np.broadcast_arrays(np.asarray(mu, float), np.asarray(kappa, float))
    if v0 is None or np.shape(v0) != mu.shape:
        from logit_beta import solve_a
        v = logit(solve_a(mu, kappa) / kappa)
    else:
        v = np.array(v0, float)
    parts = None
    for _ in range(n_newton):
        a = kappa * expit(v); b = kappa - a
        parts = inc_beta_parts(a, b, mu)
        F = parts[0] - 0.5
        if np.abs(F).max() < tol:
            break
        dF = (parts[1] - parts[2]) * a * b / kappa            # dF/dv < 0
        v = v - np.clip(F / np.where(dF < 0, dF, -1e-300), -2.0, 2.0)
        parts = None
    a = kappa * expit(v)
    if parts is None:
        parts = inc_beta_parts(a, kappa - a, mu)
    bad = np.abs(parts[0] - 0.5) >= tol
    if bad.any():  # bisection in v for the rest (F decreases in v)
        m, k = mu[bad], kappa[bad]
        lo, hi = np.full(m.shape, -40.0), np.full(m.shape, 40.0)
        for _ in range(n_bisect):
            mid = (lo + hi) / 2; f = betainc(k * expit(mid), k - k * expit(mid), m) - 0.5
            lo, hi = np.where(f > 0, mid, lo), np.where(f > 0, hi, mid)
        v = v.copy(); v[bad] = (lo + hi) / 2; a = kappa * expit(v)
        parts = inc_beta_parts(a, kappa - a, mu)
    if return_parts:
        return a, v, parts
    return (a, v) if return_v else a


_OPS = None


def _ops():
    global _OPS
    if _OPS is None:
        import pytensor.tensor as pt
        from pytensor.graph.basic import Apply
        from pytensor.graph.op import Op

        class SolveMedian(Op):
            """Root a of I_mu(a, kappa - a) = 1/2, plus F = I - 1/2, dI/da, dI/db and dI/dx there (computed anyway by
            the last Newton step). Zero gradient (exact as used in pt_solve_median, see logit_beta)."""
            last_v = None  # warm start from the previous call

            def make_node(self, m, k):
                return Apply(self, [pt.as_tensor_variable(m), pt.as_tensor_variable(k)], [pt.dvector() for _ in range(5)])

            def perform(self, node, inputs, outputs):
                a, v, (I, Ia, Ib, Ix) = solve_median(inputs[0], inputs[1], v0=self.last_v, return_parts=True)
                self.last_v = v
                for o, val in zip(outputs, (a, I - 0.5, Ia, Ib, Ix)):
                    o[0] = val

            def L_op(self, inputs, outputs, output_grads):
                return [pt.zeros_like(inputs[0]), pt.zeros_like(inputs[1])]

        class LinF(Op):
            """Value F0 = I_x(a, b) - 1/2 at the root (from SolveMedian); gradient dI/da, dI/db, dI/dx with respect to
            a, b, x (also from SolveMedian), none with respect to the passed-in values."""
            def make_node(self, a, b, x, F0, Ia, Ib, Ix):
                ins = [pt.as_tensor_variable(v) for v in (a, b, x, F0, Ia, Ib, Ix)]
                return Apply(self, ins, [pt.dvector()])

            def perform(self, node, inputs, outputs):
                outputs[0][0] = np.array(inputs[3], float)

            def L_op(self, inputs, outputs, output_grads):
                g = output_grads[0]
                return [g * inputs[4], g * inputs[5], g * inputs[6]] + [pt.zeros_like(v) for v in inputs[3:]]

        _OPS = (SolveMedian(), LinF())
    return _OPS


def pt_solve_median(mu, kappa):
    import pytensor.tensor as pt
    from pytensor.gradient import disconnected_grad as sg
    solve, linF = _ops()
    mu, kappa = pt.cast(mu, "float64"), pt.cast(kappa, "float64")
    a_s, F0, Ia, Ib, Ix = solve(sg(mu), sg(kappa))
    a_s = sg(a_s)
    F = linF(a_s, kappa - a_s, mu, F0, Ia, Ib, Ix)       # = I - 1/2 at the root; gradients dI/db via kappa, density via mu
    return a_s - F / sg(Ia - Ib)


# ---- Spline table for a(mu, kappa) (Nov 14): cheap enough to evaluate at every quadrature node (capability noise) ----
# z = logit(a / kappa) is smooth in u = logit(mu) and w = log(kappa); a bicubic spline on a fine grid (exact roots from
# solve_median at the knots) gives a(mu, kappa) and its derivatives in a few elementwise operations. The model then uses
# the spline itself (its value and exact spline derivatives), a smooth stand-in for the exact median-centring whose
# accuracy is checked against solve_median (see TABLE_* and spline_check in the brief).
# logit(mu): to +-27.7 since Nov 18, 2026 (XCI v0.1's clip on mu, 1e-12, is at logit +-27.63; Epoch's 1e-6 at +-13.82).
# Same 0.05 spacing as the first table (+-13.9, 557 knots), whose knots are all knots here.
TABLE_U = (-27.7, 27.7, 1109)
TABLE_W = (np.log(0.3), np.log(5000.0), 197)         # log(kappa)
_SPLINE = None


def _spline():
    global _SPLINE
    if _SPLINE is None:
        from scipy.interpolate import RectBivariateSpline
        from scipy.special import expit, logit
        u = np.linspace(*TABLE_U); w = np.linspace(*TABLE_W)
        U, W = np.meshgrid(u, w, indexing="ij")
        a = solve_median(expit(U.ravel()), np.exp(W.ravel())).reshape(U.shape)
        _SPLINE = RectBivariateSpline(u, w, logit(a / np.exp(W)), kx=3, ky=3, s=0)
    return _SPLINE


def spline_a(mu, kappa, derivs=False):
    """a(mu, kappa) from the spline (numpy); with derivs, also dz/du and dz/dw (z = logit(a / kappa))."""
    from scipy.special import expit, logit
    mu, kappa = np.broadcast_arrays(np.asarray(mu, float), np.asarray(kappa, float))
    u, w = logit(mu).ravel(), np.log(kappa).ravel()
    s = _spline()
    z = s.ev(u, w).reshape(mu.shape)
    a = kappa * expit(z)
    if not derivs:
        return a
    return a, z, s.ev(u, w, dx=1).reshape(mu.shape), s.ev(u, w, dy=1).reshape(mu.shape)


_SPLINE_OP = None


def pt_spline_a(mu, kappa):
    """PyTensor a(mu, kappa) = kappa sigmoid(z(logit mu, log kappa)) with z from the spline: an Op gives z and its two
    partial derivatives, a linearization Op attaches them as the gradient (value z, d/du = z_u, d/dw = z_w)."""
    global _SPLINE_OP
    import pytensor.tensor as pt
    if _SPLINE_OP is None:
        from pytensor.graph.basic import Apply
        from pytensor.graph.op import Op

        class SplineZ(Op):
            def make_node(self, u, w):
                return Apply(self, [pt.as_tensor_variable(u), pt.as_tensor_variable(w)], [pt.dvector() for _ in range(3)])

            def perform(self, node, inputs, outputs):
                s = _spline(); u, w = inputs
                outputs[0][0], outputs[1][0], outputs[2][0] = s.ev(u, w), s.ev(u, w, dx=1), s.ev(u, w, dy=1)

            def L_op(self, inputs, outputs, output_grads):
                return [pt.zeros_like(v) for v in inputs]

        class Lin2(Op):
            """Value z (given); gradient z_u, z_w with respect to u, w; none with respect to the given values."""
            def make_node(self, u, w, z, zu, zw):
                return Apply(self, [pt.as_tensor_variable(v) for v in (u, w, z, zu, zw)], [pt.dvector()])

            def perform(self, node, inputs, outputs):
                outputs[0][0] = np.array(inputs[2], float)

            def L_op(self, inputs, outputs, output_grads):
                g = output_grads[0]
                return [g * inputs[3], g * inputs[4]] + [pt.zeros_like(v) for v in inputs[2:]]

        _SPLINE_OP = (SplineZ(), Lin2())
    zop, lin = _SPLINE_OP
    _spline()  # build the table now, not inside a sampler thread
    from pytensor.gradient import disconnected_grad as sg
    mu, kappa = pt.cast(mu, "float64"), pt.cast(kappa, "float64")
    u, w = pt.log(mu) - pt.log1p(-mu), pt.log(kappa)
    z, zu, zw = zop(sg(u), sg(w))
    return kappa * pt.sigmoid(lin(u, w, z, zu, zw))


# ---- Bicubic Hermite interpolation of z on the same grid (Nov 14): plain array arithmetic, so it runs inside PyTensor's
# compiled graph (scipy's spline evaluation took ~50 ms for 24 x 2,787 points) and PyTensor differentiates it exactly.
# Knot values and derivatives (z, z_u, z_w, z_uw) come from the spline above. herm_a works on numpy or PyTensor arrays.
_HERM = None


def _herm_tables():
    global _HERM
    if _HERM is None:
        s = _spline()
        u = np.linspace(*TABLE_U); w = np.linspace(*TABLE_W)
        T = [s(u, w, dx=dx, dy=dy) for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1))]
        _HERM = dict(u0=u[0], w0=w[0], hu=u[1] - u[0], hw=w[1] - w[0], nu=len(u), nw=len(w),
                     Z=T[0].ravel(), Zu=T[1].ravel(), Zw=T[2].ravel(), Zuw=T[3].ravel())
    return _HERM


def herm_z(u, w, xp=np):
    """z(u, w) by bicubic Hermite interpolation; u, w outside the table are clamped to its edges."""
    H = _herm_tables()
    if xp is np:
        floor, clip, take, cast = np.floor, np.clip, (lambda T, i: T[i]), (lambda x: x.astype(int))
    else:
        import pytensor.tensor as pt
        floor, clip, cast = pt.floor, pt.clip, (lambda x: pt.cast(x, "int64"))
        take = lambda T, i: pt.as_tensor_variable(T)[i]
    hu, hw = H["hu"], H["hw"]
    u = clip(u, H["u0"], H["u0"] + hu * (H["nu"] - 1)); w = clip(w, H["w0"], H["w0"] + hw * (H["nw"] - 1))
    fi = clip(floor((u - H["u0"]) / hu), 0, H["nu"] - 2); fj = clip(floor((w - H["w0"]) / hw), 0, H["nw"] - 2)
    t = (u - H["u0"]) / hu - fi; s = (w - H["w0"]) / hw - fj
    i, j = cast(fi), cast(fj)
    h0 = lambda x: (2 * x - 3) * x * x + 1; h1 = lambda x: ((x - 2) * x + 1) * x     # value / slope basis at x = 0
    g0 = lambda x: (3 - 2 * x) * x * x; g1 = lambda x: (x - 1) * x * x                 # value / slope basis at x = 1
    bu = ((h0(t), h1(t)), (g0(t), g1(t))); bw = ((h0(s), h1(s)), (g0(s), g1(s)))
    z = 0
    for di in (0, 1):
        for dj in (0, 1):
            k = (i + di) * H["nw"] + (j + dj)
            z = z + (take(H["Z"], k) * bu[di][0] * bw[dj][0] + hu * take(H["Zu"], k) * bu[di][1] * bw[dj][0]
                     + hw * take(H["Zw"], k) * bu[di][0] * bw[dj][1] + hu * hw * take(H["Zuw"], k) * bu[di][1] * bw[dj][1])
    return z


def herm_a(mu, kappa, xp=np):
    """Median-centred a(mu, kappa) from the Hermite table (numpy or PyTensor)."""
    if xp is np:
        from scipy.special import expit
        mu, kappa = np.asarray(mu, float), np.asarray(kappa, float)
        return kappa * expit(herm_z(np.log(mu) - np.log1p(-mu), np.log(kappa)))
    import pytensor.tensor as pt
    return kappa * pt.sigmoid(herm_z(pt.log(mu) - pt.log1p(-mu), pt.log(kappa), xp=pt))
