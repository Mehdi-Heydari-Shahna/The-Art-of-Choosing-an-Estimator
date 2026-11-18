# Independent reference library (estimation cross-checks)

Written only from the mathematical problem statements; uses numpy, scipy and (optionally)
mpmath. No Kalman recursion, no DARE/Lyapunov library solver, no augmented-matrix
exponential, no error-recursion or closed-form observer formulas are used.
Run `python selfcheck.py` (about 5 s): prints PASS lines, exit code 0 when all pass.

## ref_kalman.py

Convention: `x_{k+1} = A x_k + B u_k + w_k`, `y_k = C x_k + v_k`, k = 0..N; the filtered
estimate at k uses `y_0..y_k` and `u_0..u_{k-1}`.

| function | computes | algorithm | accuracy / limits |
|---|---|---|---|
| `batch_filtered(A,B,C,Qd,R,x0_hat,P0,y,u, method="chol", dps=None)` | `E[x_k\|y_0..y_k]`, `Cov[x_k\|y_0..y_k]` for all k | explicit joint mean/covariance of `(x_0..x_N, y_0..y_N)`, then one dense solve per k (Cholesky, lstsq fallback) | float64: about 1e-12 relative for benign models; degrades as `eps * cond(Cov y) * max\|Cov x_k\| / \|cov_k\|` (1e-6 for `P0 = 1e6 I` or `eig(A) = 1.5`, N = 25; useless for `eig(A) = 1.5`, N >= 40). `dps=50` runs the same formulas in mpmath and removes the problem (about 12 s at N = 80, n = 2). O(N^4), N <= 80. |
| `batch_gain_first(A,C,Qd,R,P0)` | `K0` with `E[x_0\|y_0] = x0_hat + K0 (y_0 - C x0_hat)` | N = 0 case of the same joint covariance | machine precision |
| `riccati_fixed_point(A,C,Qd,R,tol,itmax)` | steady-state prior covariance, filtered covariance, update gain `M = P C'(C P C'+R)^-1` | plain iteration of the covariance map from `Qd + I`; stops when the step AND the estimated remaining error `d rho/(1-rho)` are below `tol * max\|P\|` | 1e-13 relative observed; raises `RuntimeError` when convergence is sublinear (unit-circle modes not excited by `Qd`) or absent |
| `zoh_quadrature(Ac,Bc,Ts)` | `Ad = expm(Ac Ts)`, `Bd = int_0^Ts expm(Ac s) ds Bc` | composite Gauss-Legendre, 20 nodes x 8 panels (panels auto-increased so `\|\|Ac\|\| * width <= 2`, cap 4096) | 1e-13 or better in tests, including `\|\|Ac\|\| Ts ~ 600` |
| `process_cov_quadrature(Ac,G,Qc,Ts)` | `Qd = int_0^Ts expm(Ac s) G Qc G' expm(Ac' s) ds` | same quadrature | same |
| `fixed_gain_mse_montecarlo(...)` | empirical per-component MSE (N+1, n) of `xp_k = xm_k + M(y_k - C xm_k)`, `xm_{k+1} = A xp_k`, `u = 0`, `xm_0 = x0_hat` | vectorised over runs, `numpy.random.default_rng(seed)`; noise = `L z`, `z` i.i.d. normal or uniform on `[-sqrt3, sqrt3]` | statistical: relative standard error about `sqrt(2/n_runs)`; `estimate="predicted"` returns the `xm_k` error instead |
| `lyapunov_mse_of_gain(A,C,Qd,R,M)` | steady-state filtered-error covariance of that estimator | `(I - F(x)F) vec P = vec W`, `F = (I-MC)A`, `numpy.linalg.solve` | machine precision times `cond(I - F(x)F)`; `ValueError` if `F` is not Schur stable |

## ref_hgo.py

Plant `x1' = x2`, `x2' = a(t)`, `y_k = x1(kT) + v_k`; observer `xp_k = xm_k + M(y_k - xm_k[0])`,
`xm_{k+1} = Ad xp_k + Gam a0_k`. Bounds `|x_i(0) - xm_0[i]| <= E_i`, `|v_k| <= V`,
`|a(t) - a0_k| <= D`. "current" = `xp_k`, "predictor" = `xm_k`; error = true - estimate.

| function | computes | algorithm | accuracy / limits |
|---|---|---|---|
| `simulate_observer(T,M,x0,xm0,v,delta_fn,a0_seq,N,substeps)` | true states, `xp`, `xm`, `y`, `t` at the sample instants | state and estimate propagated separately; plant integrated exactly for acceleration held over each sub-step (`delta_fn` sampled at sub-step midpoints) | exact (rounding only) when `delta` is constant on the sub-steps; otherwise a midpoint-rule approximation of the signal, O(1/substeps^2). `delta_fn` is tried as a vectorised call first, then point by point |
| `worst_case_by_enumeration(T,M,c,E,D,V,N,form,sub)` | `max \|c.(x_N - estimate_N)\|` with `delta` constant on `sub` sub-intervals per sample: a LOWER bound of the true worst case | coefficients of the linear map obtained from `2 + (N+1) + N*sub` unit-input simulations; result `\|ce\|.E + V sum\|cv\| + D sum\|cd\|` | relative gap to the true value O(1/sub^2) (4e-7 at sub = 64 in the test case); 1.2 s for N = 80, sub = 64 |
| `worst_case_by_enumeration_bounds(...)` | `(lower, upper)` bracketing the worst case over measurable `delta` | `upper = lower + D * sum_k \|cd[k,1]-cd[k,0]\|/4` over sample intervals whose kernel changes sign; rigorous in exact arithmetic because the kernel is affine in each sample interval (checked numerically in selfcheck) | bracket width O(D N T^2 / sub^2) |
| `worst_case_exact(...)` | the worst case over measurable `delta` | lower bound plus the exactly evaluated defect of the one sign-change sub-interval per sample | independent of `sub` to 2e-16 in tests |
| `worst_case_signals(...)` | `(x0err, v, delta_fn)` attaining the lower bound | signs of the coefficients | reproduces the bound to 1e-14 relative, PROVIDED `simulate_observer` is called with `substeps` a multiple of `sub` |
| `ct_observer_worst_case(g,E,D,V,t_end,target,n)` | pointwise worst case `W(t)` of `\|e_target(t)\|` for the continuous observer with double pole at `-g` | `scipy.linalg.expm` of `[[-2g,1],[-g^2,0]]` at spacing `dt/2`; Simpson per cell, quadratic root split in sign-change cells (4th order) | 1e-13 relative at n = 20001 (g t_end = 50); error scales as n^-4 (4e-9 at n = 801) |
| `lower_limit_two_signals(D,V)` | two admissible signals with zero measurement and velocity gap `4 sqrt(DV)` at `t_star = 2 sqrt(V/D)`, plus numerical verification | explicit construction (hold at `-V`, then parabola of curvature `D`), finite-difference checks | functions are defined for `t <= t_star` only (nan beyond) |

## Notes on the conventions (where the specification needed interpretation)

* `u_N` (if supplied) is unused. For non-Gaussian noise `batch_filtered` returns the best
  linear estimate and its error covariance, not conditional moments.
* `riccati_fixed_point` returns the update gain `M`; the predictor gain is `A M`.
* `dist="uniform"` draws `L z` with uniform `z`: only for diagonal covariances are the
  noise components themselves uniform.
* `form="predictor"`: `estimate_N = xm_N` does not depend on `v_N` (coefficient 0).
* Default `substeps=200` of `simulate_observer` is not a multiple of the default `sub=64`;
  pass `substeps=sub` (or a multiple) when replaying `worst_case_signals`.
* The two-signal construction proves the `2 sqrt(DV)` limit for CAUSAL estimates (data up to
  the estimation time). The signals cannot be extended beyond `t_star` with `|x| <= V`; with
  data on the whole time axis the corresponding limit is `sqrt(2DV)`. With bounded initial
  error the construction also needs `E_1 >= V` (or enough lead time to drift apart).
