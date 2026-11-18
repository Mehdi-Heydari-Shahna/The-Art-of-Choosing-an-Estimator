"""Independent reference computations for linear estimation (batch, non-recursive).

Model and timing convention used by every function in this file::

    x_{k+1} = A x_k + B u_k + w_k,     y_k = C x_k + v_k,     k = 0..N
    w_k ~ (0, Qd),  v_k ~ (0, R),  x_0 ~ (x0_hat, P0),  all mutually uncorrelated, white.

The "filtered" quantity at step k uses y_0..y_k and u_0..u_{k-1} (y_k is taken BEFORE
u_k acts).  The "prior"/"predicted" quantity at step k uses y_0..y_{k-1}.

Only numpy/scipy are used.  No Kalman recursion and no DARE/Lyapunov library solver is
called anywhere: filtering is done by explicit conditioning of the stacked joint Gaussian,
steady states by plain fixed-point iteration or a Kronecker linear solve, discretisation
by Gauss-Legendre quadrature of matrix exponentials.
"""
import numpy as np
from scipy.linalg import LinAlgError, cho_factor, cho_solve, expm


def _mat(a):
    """Float array with at least two dimensions (scalars become 1x1)."""
    return np.atleast_2d(np.asarray(a, dtype=float))


def _sym(P):
    return 0.5 * (P + P.T)


def _spd_solve(S, rhs, method="chol"):
    """Solve S X = rhs for symmetric PSD S: Cholesky, falling back to lstsq."""
    if method == "chol":
        try:
            return cho_solve(cho_factor(S, lower=True, check_finite=True), rhs)
        except LinAlgError:
            pass
    return np.linalg.lstsq(S, rhs, rcond=None)[0]


def _joint_moments(A, B, C, Qd, R, x0_hat, P0, u, N):
    """Mean and covariance blocks of X = (x_0..x_N) and Y = (y_0..y_N).

    X = Phi z + Phi d,  z = (x_0 - x0_hat, w_0..w_{N-1}),  d = (x0_hat, B u_0..B u_{N-1}),
    with block Phi[k, 0] = A^k and Phi[k, j+1] = A^(k-1-j) for j < k, and Y = (I (x) C) X + V.
    Returns mX, SX = Cov(X), mY, SXY = Cov(X, Y), SY = Cov(Y).
    """
    n, ny = A.shape[0], C.shape[0]
    Apow = [np.eye(n)]
    for _ in range(N):
        Apow.append(A @ Apow[-1])
    Phi = np.zeros(((N + 1) * n, (N + 1) * n), dtype=A.dtype)   # float or object (mpf)
    Dg = np.zeros_like(Phi)
    Dg[:n, :n] = P0
    for k in range(N + 1):
        Phi[k * n:(k + 1) * n, :n] = Apow[k]
        for j in range(k):
            Phi[k * n:(k + 1) * n, (j + 1) * n:(j + 2) * n] = Apow[k - 1 - j]
        if k < N:
            Dg[(k + 1) * n:(k + 2) * n, (k + 1) * n:(k + 2) * n] = Qd
    drive = np.concatenate([x0_hat] + [B @ u[j] for j in range(N)])
    mX = Phi @ drive
    SX = _sym(Phi @ Dg @ Phi.T)
    Cb = np.kron(np.eye(N + 1), C)
    mY = Cb @ mX
    SXY = SX @ Cb.T
    SY = _sym(Cb @ SXY + np.kron(np.eye(N + 1), R))
    return mX, SX, mY, SXY, SY


def batch_filtered(A, B, C, Qd, R, x0_hat, P0, y, u=None, method="chol", dps=None):
    """Filtered means/covariances E[x_k | y_0..y_k], Cov[x_k | y_0..y_k] by batch conditioning.

    y : (N+1, ny) (a 1-D array is accepted when ny == 1).
    u : (N or N+1, m) or None (= zero input); u_k acts on the step k -> k+1, so u_N is unused.
    B may be None when u is None.
    Returns means (N+1, n) and covs (N+1, n, n).

    Algorithm: the joint mean/covariance of (x_0..x_N, y_0..y_N) is assembled explicitly;
    for every k a SEPARATE dense solve with Cov(y_0..y_k) gives
    mean_k = m_xk + S_xy S_yy^{-1} (y - m_y),  cov_k = S_xx - S_xy S_yy^{-1} S_yx.
    Cost O(N^4 ny^3); intended for N <= 80.  `method` is "chol" (lstsq fallback if the
    output covariance is not numerically positive definite) or "lstsq".
    ACCURACY: covariance-form conditioning subtracts large numbers; in float64 the
    relative error is roughly eps * cond(Cov(y)) * max|Cov(x_k)| / |cov_k| (1e-12 for
    benign models, useless for |eig(A)| >~ 1.4 at N = 40).  Pass dps=<digits> (e.g. 50)
    to run the SAME batch formulas in mpmath arithmetic (inputs taken as exact doubles,
    outputs rounded to float64); needs Cov(y_0..y_N) positive definite.
    For non-Gaussian noise the result is the linear-minimum-variance estimate and its
    error covariance (not the conditional moments).
    """
    A, C, Qd, R, P0 = _mat(A), _mat(C), _mat(Qd), _mat(R), _mat(P0)
    n, ny = A.shape[0], C.shape[0]
    x0_hat = np.asarray(x0_hat, dtype=float).reshape(n)
    y = np.asarray(y, dtype=float)
    y = y.reshape(-1, ny) if y.ndim < 2 else y
    N = y.shape[0] - 1
    if u is None:
        B, u = np.zeros((n, 1)), np.zeros((N, 1))
    B = _mat(B).reshape(n, -1)
    u = np.asarray(u, dtype=float)
    u = u.reshape(-1, B.shape[1]) if u.ndim < 2 else u
    if u.shape[0] < N:
        raise ValueError("u must have at least N rows (u_k applied on step k -> k+1)")
    if dps is not None:
        return _batch_filtered_mp(A, B, C, Qd, R, x0_hat, P0, y, u, N, int(dps))
    mX, SX, mY, SXY, SY = _joint_moments(A, B, C, Qd, R, x0_hat, P0, u, N)
    resid = y.reshape(-1) - mY
    means, covs = np.empty((N + 1, n)), np.empty((N + 1, n, n))
    for k in range(N + 1):
        m, rows = (k + 1) * ny, slice(k * n, (k + 1) * n)
        G = SXY[rows, :m]                                   # Cov(x_k, (y_0..y_k))
        W = _spd_solve(SY[:m, :m], np.column_stack([G.T, resid[:m]]), method)
        means[k] = mX[rows] + G @ W[:, n]
        covs[k] = _sym(SX[rows, rows] - G @ W[:, :n])
    return means, covs


def _batch_filtered_mp(A, B, C, Qd, R, x0_hat, P0, y, u, N, dps):
    """batch_filtered in mpmath arithmetic (object arrays of mpf, `dps` digits).

    Same joint moments; one Cholesky factor L of Cov(y_0..y_N) is used for all k because
    its leading (k+1)ny block is the Cholesky factor of Cov(y_0..y_k):
    Z = L^{-1} [Cov(Y, X), y - m_y],  mean_k = m_xk + Z_k' z,  cov_k = S_xx - Z_k' Z_k
    with Z_k, z restricted to the first (k+1)ny rows.
    """
    import mpmath as mp
    n, ny = A.shape[0], C.shape[0]
    with mp.workdps(dps):
        tomp = lambda a: np.vectorize(mp.mpf, otypes=[object])(np.asarray(a, dtype=float))
        mX, SX, mY, SXY, SY = _joint_moments(*(tomp(q) for q in (A, B, C, Qd, R, x0_hat, P0, u)), N)
        L = np.array(mp.cholesky(mp.matrix(SY.tolist())).tolist(), dtype=object)
        rhs = np.column_stack([SXY.T, tomp(y).reshape(-1) - mY])
        Z = np.empty_like(rhs)
        for i in range(rhs.shape[0]):                        # forward substitution
            Z[i] = (rhs[i] - (L[i, :i] @ Z[:i] if i else 0)) / L[i, i]
        means, covs = np.empty((N + 1, n)), np.empty((N + 1, n, n))
        for k in range(N + 1):
            m, rows = (k + 1) * ny, slice(k * n, (k + 1) * n)
            Zk = Z[:m, rows]
            means[k] = (mX[rows] + Zk.T @ Z[:m, -1]).astype(float)
            covs[k] = _sym(SX[rows, rows] - Zk.T @ Zk).astype(float)
    return means, covs


def batch_gain_first(A, C, Qd, R, P0):
    """K0 (n x ny) with E[x_0 | y_0] = x0_hat + K0 (y_0 - C x0_hat), from the batch formulas.

    Uses the N = 0 joint covariance: K0 = Cov(x_0, y_0) Cov(y_0)^{-1}.  A and Qd do not
    enter (kept in the signature for uniformity).
    """
    A, C, Qd, R, P0 = _mat(A), _mat(C), _mat(Qd), _mat(R), _mat(P0)
    n = A.shape[0]
    _, _, _, SXY, SY = _joint_moments(A, np.zeros((n, 1)), C, Qd, R, np.zeros(n), P0,
                                      np.zeros((0, 1)), 0)
    return _spd_solve(SY, SXY.T).T


def riccati_fixed_point(A, C, Qd, R, tol=None, itmax=200, dps=50):
    """Steady-state PRIOR covariance of the filter Riccati equation, independently of the library DARE solver.

    Structure-preserving doubling in `dps`-digit arithmetic (mpmath): with the control-form data A_0 = A', G_0 =
    C' R^{-1} C, H_0 = Qd, iterate A_{k+1} = A_k (I + G_k H_k)^{-1} A_k, G_{k+1} = G_k + A_k (I + G_k H_k)^{-1} G_k A_k',
    H_{k+1} = H_k + A_k' H_k (I + G_k H_k)^{-1} A_k; H_k is the Riccati iterate after 2^k steps from zero and converges
    quadratically to the stabilising solution.  Stop when every entry of H changes by less than 10^-(dps-12) of its
    own scale sqrt(H_ii H_jj).  (A float iteration of the covariance map with a per-entry rule, used before, could stop
    early or at its start when the contraction factor is near 1: second recheck A.)
    Returns (P_prior, P_filtered, gain) in float64 with gain M = P C'(C P C' + R)^{-1} (the UPDATE gain) and
    P_filtered = (I - M C) P (I - M C)' + M R M' (Joseph form), both computed in the same arithmetic.
    Raises RuntimeError if the doubling does not converge in itmax steps (2^itmax Riccati steps), or if the result is
    not stabilizing (spectral radius of A (I - M C) at least 1: undetectable, or a unit-circle mode that Qd does not
    excite, whose variance keeps its starting value).  `tol` is accepted for compatibility and ignored."""
    import mpmath as mp
    A, C, Qd, R = _mat(A), _mat(C), _mat(Qd), _mat(R)
    n = A.shape[0]
    with mp.workdps(dps):
        M_ = lambda X: mp.matrix(np.asarray(X, float).tolist())
        Ak, Ck, Qm, Rm, I = M_(A).T, M_(C), M_(Qd), M_(R), mp.eye(n)
        Gk = Ck.T * mp.inverse(Rm) * Ck
        Hk = Qm
        floor, ok = mp.mpf(10) ** (-(dps - 12)), False
        for _ in range(int(itmax)):
            W = mp.inverse(I + Gk * Hk)
            A1 = Ak * W * Ak
            G1 = Gk + Ak * W * Gk * Ak.T
            H1 = Hk + Ak.T * Hk * W * Ak
            G1, H1 = (G1 + G1.T) / 2, (H1 + H1.T) / 2
            ch = max(abs(H1[i, j] - Hk[i, j]) / mp.sqrt(abs(H1[i, i] * H1[j, j]) + mp.mpf(10) ** (-300))
                     for i in range(n) for j in range(n))
            Ak, Gk, Hk = A1, G1, H1
            if ch <= floor:
                ok = True
                break
        if not ok:
            raise RuntimeError("riccati_fixed_point: the doubling iteration did not converge")
        # A_k is (up to the converged factors) the closed-loop matrix raised to the power 2^k: it vanishes exactly when the
        # solution is stabilizing; otherwise the fixed point depends on the start (second recheck B, point 3)
        if max(abs(Ak[i, j]) for i in range(n) for j in range(n)) > mp.mpf(10) ** (-20):
            raise RuntimeError("riccati_fixed_point: the fixed point reached is not stabilizing (undetectable or unexcited unit-circle mode)")
        Pm = Hk
        Mm = Pm * Ck.T * mp.inverse(Ck * Pm * Ck.T + Rm)
        J = I - Mm * Ck
        Pf = J * Pm * J.T + Mm * Rm * Mm.T
        tof = lambda X: np.array([[float(X[i, j]) for j in range(X.cols)] for i in range(X.rows)])
        P, Pf, M = _sym(tof(Pm)), _sym(tof(Pf)), tof(Mm)
    return P, Pf, M


def _gauss_legendre(Ac, Ts, nodes, panels):
    """Composite Gauss-Legendre nodes/weights on [0, Ts]; panels are increased so that
    ||Ac||_1 * panel_width <= 2 (capped at 4096 panels)."""
    if nodes < 20 or panels < 8:
        raise ValueError("use at least 20 nodes on at least 8 panels")
    panels = int(max(panels, min(4096, np.ceil(np.linalg.norm(Ac, 1) * abs(Ts) / 2.0))))
    xg, wg = np.polynomial.legendre.leggauss(int(nodes))
    edges = np.linspace(0.0, Ts, panels + 1)
    half = 0.5 * np.diff(edges)
    s = (edges[:-1, None] + half[:, None] * (xg[None, :] + 1.0)).ravel()
    return s, (half[:, None] * wg[None, :]).ravel()


def zoh_quadrature(Ac, Bc, Ts, nodes=20, panels=8):
    """Zero-order-hold discretisation: Ad = expm(Ac Ts), Bd = (int_0^Ts expm(Ac s) ds) Bc.

    The integral is a composite Gauss-Legendre sum of scipy.linalg.expm values
    (not the augmented-matrix exponential).  Returns (Ad, Bd) with Bd of shape (n, m).
    """
    Ac = _mat(Ac)
    n = Ac.shape[0]
    Bc = _mat(Bc).reshape(n, -1) if np.ndim(Bc) < 2 else _mat(Bc)
    s, w = _gauss_legendre(Ac, Ts, nodes, panels)
    integral = np.zeros((n, n))
    for si, wi in zip(s, w):
        integral += wi * expm(Ac * si)
    return expm(Ac * Ts), integral @ Bc


def process_cov_quadrature(Ac, G, Qc, Ts, nodes=20, panels=8):
    """Qd = int_0^Ts expm(Ac s) G Qc G' expm(Ac' s) ds by composite Gauss-Legendre.

    Qc is the two-sided spectral density (intensity) of the continuous white noise w_c in
    x' = Ac x + G w_c, i.e. E[w_c(t) w_c(s)'] = Qc delta(t - s).
    """
    Ac = _mat(Ac)
    n = Ac.shape[0]
    G = _mat(G).reshape(n, -1) if np.ndim(G) < 2 else _mat(G)
    W = G @ _mat(Qc) @ G.T
    s, w = _gauss_legendre(Ac, Ts, nodes, panels)
    Qd = np.zeros((n, n))
    for si, wi in zip(s, w):
        E = expm(Ac * si)
        Qd += wi * (E @ W @ E.T)
    return _sym(Qd)


def _psd_factor(S):
    """L with L L' = S (Cholesky if positive definite, else eigen-factor of the PSD part)."""
    try:
        return np.linalg.cholesky(S)
    except np.linalg.LinAlgError:
        lam, V = np.linalg.eigh(_sym(S))
        return V * np.sqrt(np.clip(lam, 0.0, None))


def fixed_gain_mse_montecarlo(A, B, C, Qd, R, M, x0_hat, P0, N, n_runs, seed,
                              dist="gaussian", estimate="filtered"):
    """Empirical mean-square error (N+1, n) of the constant-gain estimator, u = 0.

    xp_k = xm_k + M (y_k - C xm_k),  xm_{k+1} = A xp_k,  xm_0 = x0_hat,  x_0 ~ (x0_hat, P0).
    Row k is mean over runs of (x_k - xp_k)**2 component-wise (estimate="filtered",
    default) or of (x_k - xm_k)**2 (estimate="predicted").  B is unused because u = 0.
    Noise vectors are L z with L L' = covariance and z i.i.d. zero-mean unit-variance:
    standard normal (dist="gaussian") or uniform on [-sqrt(3), sqrt(3)] (dist="uniform";
    the components of L z are themselves uniform only when the covariance is diagonal).
    Statistical accuracy: relative standard error about sqrt(2 / n_runs) (Gaussian).
    """
    A, C, Qd, R, P0 = _mat(A), _mat(C), _mat(Qd), _mat(R), _mat(P0)
    n, ny = A.shape[0], C.shape[0]
    M = np.asarray(M, dtype=float).reshape(n, ny)
    rng = np.random.default_rng(seed)
    if dist == "gaussian":
        draw = lambda L: rng.standard_normal((n_runs, L.shape[0])) @ L.T
    elif dist == "uniform":
        draw = lambda L: rng.uniform(-np.sqrt(3.0), np.sqrt(3.0), (n_runs, L.shape[0])) @ L.T
    else:
        raise ValueError("dist must be 'gaussian' or 'uniform'")
    LQ, LR, LP = _psd_factor(Qd), _psd_factor(R), _psd_factor(P0)
    xm = np.tile(np.asarray(x0_hat, dtype=float).reshape(1, n), (n_runs, 1))
    x = xm + draw(LP)
    mse = np.empty((N + 1, n))
    for k in range(N + 1):
        yk = x @ C.T + draw(LR)
        xp = xm + (yk - xm @ C.T) @ M.T
        err = x - (xp if estimate == "filtered" else xm)
        mse[k] = np.mean(err * err, axis=0)
        x = x @ A.T + draw(LQ)
        xm = xp @ A.T
    return mse


def lyapunov_mse_of_gain(A, C, Qd, R, M):
    """Steady-state covariance P+ of the FILTERED error of the constant-gain estimator.

    Error dynamics e+_k = (I - M C) e-_k - M v_k, e-_{k+1} = A e+_k + w_k give
    P+ = F P+ F' + (I - M C) Qd (I - M C)' + M R M',  F = (I - M C) A,
    solved as (I - F (x) F) vec(P+) = vec(W) with numpy.linalg.solve (row-major vec).
    Requires spectral radius of F < 1 (ValueError otherwise).  The prior-error covariance
    is A P+ A' + Qd.
    """
    A, C, Qd, R = _mat(A), _mat(C), _mat(Qd), _mat(R)
    n, ny = A.shape[0], C.shape[0]
    M = np.asarray(M, dtype=float).reshape(n, ny)
    ImMC = np.eye(n) - M @ C
    F = ImMC @ A
    if np.max(np.abs(np.linalg.eigvals(F))) >= 1.0:
        raise ValueError("(I - M C) A is not Schur stable: no steady-state covariance")
    W = ImMC @ Qd @ ImMC.T + M @ R @ M.T
    P = np.linalg.solve(np.eye(n * n) - np.kron(F, F), W.reshape(-1)).reshape(n, n)
    return _sym(P)
