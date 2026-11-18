"""Numerical kernels reused VERBATIM from the supplied Kalman package (development baseline).

Origin: heros_kalman/kalman_agent/methods/kf.py   sha256 50766e849c3fd2f000a08bedc50da9f3af5210b5eeab412e782c553dbfd2b9e5
        heros_kalman/kalman_agent/methods/lqr.py  sha256 cd2bd961fb1d8832dd7945657b141c4ccb03fa88f470f8465b63d387b572b872
Functions copied without modification: sym, kalman_gains, fixed_gain_cov, run_filter, discretize_noise (kf.py);
riccati_lti is the body of lqr.riccati specialised to a time-invariant pair (same recursion, same symmetrisation).
The connection of these kernels to the new public interface is validated independently in
independent/ref_kalman.py and tests/test_kalman.py.
"""
from __future__ import annotations
import numpy as np
from scipy.linalg import cho_factor, cho_solve


def sym(M):
    return 0.5 * (M + M.T)


def _seq(X, T):
    return X if isinstance(X, list) else [np.atleast_2d(np.asarray(X, float))] * T


def kalman_gains(A, C, W, V, S0, T):
    """Returns (M, S, Sf): update gains M[t] (t=0..T-1), prior covariances S[t] (t=0..T),
    filtered covariances Sf[t] (t=0..T-1)."""
    A = _seq(A, T); C = _seq(C, T); W = _seq(W, T); V = _seq(V, T)
    S = [sym(np.asarray(S0, float))]; M = []; Sf = []
    for t in range(T):
        G = sym(C[t] @ S[t] @ C[t].T + V[t])
        c = cho_factor(G)
        Mt = cho_solve(c, C[t] @ S[t]).T
        I_MC = np.eye(S[t].shape[0]) - Mt @ C[t]
        Sft = sym(I_MC @ S[t] @ I_MC.T + Mt @ V[t] @ Mt.T)
        M.append(Mt); Sf.append(Sft)
        S.append(sym(A[t] @ Sft @ A[t].T + W[t]))
    return M, S, Sf


def fixed_gain_cov(A, C, W, V, S0, M):
    """Prior/filtered error covariances of the (generally suboptimal) fixed gain sequence M.
    Affine in (W, V, S0)."""
    T = len(M)
    A = _seq(A, T); C = _seq(C, T); W = _seq(W, T); V = _seq(V, T)
    S = [sym(np.asarray(S0, float))]; Sf = []
    for t in range(T):
        I_MC = np.eye(S[t].shape[0]) - M[t] @ C[t]
        Sft = sym(I_MC @ S[t] @ I_MC.T + M[t] @ V[t] @ M[t].T)
        Sf.append(Sft); S.append(sym(A[t] @ Sft @ A[t].T + W[t]))
    return S, Sf


def run_filter(A, B, C, M, y, u, m0):
    """Measurement update then time update. Returns filtered estimates xf[t] and priors xp[t]."""
    T = len(M); A = _seq(A, T); C = _seq(C, T); B = _seq(B, T)
    xp = [np.asarray(m0, float)]; xf = []
    for t in range(T):
        xf.append(xp[t] + M[t] @ (y[t] - C[t] @ xp[t]))
        xp.append(A[t] @ xf[t] + B[t] @ u[t])
    return xf, xp


def discretize_noise(Ac, Gc, Wc, dt):
    """Van Loan (1978) discretisation of continuous white process noise (not from Kalman's papers)."""
    from scipy.linalg import expm
    n = Ac.shape[0]
    M = np.zeros((2 * n, 2 * n))
    M[:n, :n] = -Ac; M[:n, n:] = Gc @ Wc @ Gc.T; M[n:, n:] = Ac.T
    E = expm(M * dt)
    Ad = E[n:, n:].T
    return sym(Ad @ E[:n, n:])


def riccati_lti(A, B, Q, R, QN, N):
    """Backward Riccati recursion of the finite-horizon LQR (legacy lqr.riccati for a time-invariant pair).
    Returns (K, P): u_t = -K[t] x_t, cost-to-go x' P[t] x."""
    A = np.atleast_2d(np.asarray(A, float)); B = np.atleast_2d(np.asarray(B, float))
    P = [None] * (N + 1); K = [None] * N
    P[N] = sym(np.asarray(QN, float))
    for t in range(N - 1, -1, -1):
        G = sym(R + B.T @ P[t + 1] @ B)
        c = cho_factor(G)
        K[t] = cho_solve(c, B.T @ P[t + 1] @ A)
        Ak = A - B @ K[t]
        P[t] = sym(Q + K[t].T @ R @ K[t] + Ak.T @ P[t + 1] @ Ak)
    return K, P
