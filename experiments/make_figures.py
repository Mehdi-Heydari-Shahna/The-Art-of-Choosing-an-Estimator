"""Result figures of the article, generated ONLY from saved results (results/*.json, *.npz).

    python experiments/make_figures.py            # all figures whose inputs exist
Vector PDFs go to paper/figures/.  Colours: a validated categorical palette (blue = Kalman filter, orange = high-gain
observer, aqua = conventional / third design, violet = reference); every series is also distinguished by line style and
named in a legend, so no figure relies on colour alone.
"""
import json
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, FancyArrowPatch, Rectangle, Circle
from common import ROOT, RES

FIG = ROOT / "paper" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
BLUE, ORANGE, AQUA, VIOLET, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#0b0b0b", "#52514e", "#d9d8d2"
plt.rcParams.update({"font.size": 8.5, "font.family": "DejaVu Sans", "axes.linewidth": 0.6, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.8, "legend.frameon": False,
                     "lines.linewidth": 1.2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.4, "pdf.fonttype": 42, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.titlesize": 7.5, "axes.titleweight": "bold", "axes.titlelocation": "left", "figure.dpi": 150})
W2 = 7.1          # double-column width [in]


def tag(ax, s):
    ax.text(-0.02, 1.03, s, transform=ax.transAxes, fontsize=9.5, fontweight="bold", va="bottom", ha="right", color=INK)


def fig_fidelity():
    J = json.loads((RES / "joint.json").read_text()); jt = np.load(RES / "joint_traces.npz")
    bt = np.load(RES / "boom_traces.npz"); L = json.loads((RES / "linear_stochastic.json").read_text())
    fig, ax = plt.subplots(2, 2, figsize=(W2, 4.5))
    a = ax[0, 0]; t = bt["t"]; n = t <= 1.5
    a.fill_between(t[n], bt["xf"][n, 1] - 2 * bt["sd"][n, 1], bt["xf"][n, 1] + 2 * bt["sd"][n, 1], color=BLUE, alpha=0.2, lw=0, label="estimate ± 2 standard deviations")
    a.plot(t[n], bt["x"][n, 1], color=INK, lw=0.9, label="true boom rate (simulated)")
    a.plot(t[n], bt["xf"][n, 1], color=BLUE, ls="--", label="Kalman estimate")
    a.set_xlabel("time [s]"); a.set_ylabel("boom rate [rad/s]"); a.set_ylim(-0.06, 0.085); a.legend(loc="upper right", ncol=1); tag(a, "(a)")
    a = ax[0, 1]
    a.semilogy(t, bt["rep_std_gaussian"], color=BLUE, label="reported $\\sqrt{P_{k|k}}$")
    a.semilogy(t, bt["emp_std_gaussian"], color=INK, lw=0.8, ls=":", label="empirical, Gaussian noise")
    a.semilogy(t, bt["emp_std_bounded"], color=AQUA, lw=0.8, ls="-.", label="empirical, uniform noise")
    a.set_ylim(1.5e-3, 0.08)                   # legend between the initial transient and the steady curves (review D, recheck)
    a.set_xlabel("time [s]"); a.set_ylabel("boom-rate error std. dev. [rad/s]"); a.legend(loc="center right"); tag(a, "(b)")
    a = ax[1, 0]; t = jt["t"]; n = t <= 0.4
    a.semilogy(t[n], jt["envelope"][n], color=ORANGE, label="exact worst case (first-reading start)")
    a.plot([0.4], [J["certified_post_deadline"]], marker="<", ms=5, mfc=ORANGE, mec="white", mew=0.6, ls="none", clip_on=False,
           label="certified upper bound after the deadline")
    a.semilogy(t[n], np.maximum(jt["ct_envelope"][n], 1e-9), color=VIOLET, lw=0.9, ls="--", label="continuous-time exact worst case, same $\\varepsilon$")
    a.semilogy(t[n], np.maximum(jt["err_max"][n], 1e-6), color=INK, lw=0.8, label="largest simulated error (100 records)")
    a.semilogy(t[n], np.maximum(jt["err_q50"][n], 1e-6), color=MUTED, lw=0.8, ls=":", label="median simulated error")
    a.axvline(J["deadline"], color=MUTED, lw=0.6, ls=(0, (4, 2))); a.axhline(J["tolerance"], color=MUTED, lw=0.6, ls=(0, (1, 2)))
    a.text(J["deadline"] + 0.004, 2.2, "deadline", fontsize=7.5, color=MUTED); a.text(0.25, J["tolerance"] * 1.2, "requested limit", fontsize=7.5, color=MUTED)
    a.set_ylim(1e-7, 6); a.set_xlabel("time [s]"); a.set_ylabel("velocity error [rad/s]")
    a.legend(loc="lower right", ncol=1, fontsize=7, frameon=True, facecolor="white", edgecolor="none", framealpha=1.0); tag(a, "(c)")
    a = ax[1, 1]; tr = J["tradeoff"]; e = np.array(tr["eps"]) * 1e3
    a.loglog(e, tr["ultimate"], color=ORANGE, label="exact worst case after the transient (sum)")
    a.loglog(e, tr["disturbance_part"], color=VIOLET, ls="--", lw=0.9, label="disturbance and mismatch part")
    a.loglog(e, tr["noise_part"], color=AQUA, ls="-.", lw=0.9, label="measurement-error part")
    a.plot([J["eps"] * 1e3], [J["certified_ultimate"]], marker="o", ms=5, mfc=ORANGE, mec="white", mew=1.0, ls="none", label="released design, certified upper bound")
    a.set_xlabel("observer parameter $\\varepsilon$ [ms]"); a.set_ylabel("worst-case velocity error [rad/s]"); a.legend(loc="upper left")
    a.set_ylim(2e-3, 30); tag(a, "(d)")
    fig.tight_layout(h_pad=1.4, w_pad=1.6); fig.savefig(FIG / "fig_fidelity.pdf"); plt.close(fig)


def fig_common():
    C = json.loads((RES / "common_domain.json").read_text()); ct = np.load(RES / "common_traces.npz")
    fig, ax = plt.subplots(1, 3, figsize=(W2, 3.3), gridspec_kw={"width_ratios": [1.1, 1.1, 1.0]})
    t = ct["t"] * 1e3; n = t <= 250
    a = ax[0]
    a.semilogy(t[n], ct["rms_kalman_steady_state"][n], color=BLUE, label="Kalman, constant gain")
    a.semilogy(t[n], ct["rms_kalman_time_varying"][n], color=BLUE, ls=":", lw=0.9, label="Kalman, time-varying")
    a.semilogy(t[n], ct["rms_hgo_worst_case_tuned"][n], color=ORANGE, ls="--", label="observer")
    a.set_ylim(5e-3, 5); a.set_xlabel("time [ms]"); a.set_ylabel("RMS speed error [rad/s]"); a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), handlelength=1.8); tag(a, "(a)")
    a = ax[1]
    a.semilogy(t[n], ct["W_kss"][n], color=BLUE, label="Kalman, exact worst case")
    a.semilogy(t[n], ct["W_hgo"][n], color=ORANGE, ls="--", label="observer, exact worst case")
    a.plot([250.0], [float(ct["cert_kss"])], marker="<", ms=5, mfc=BLUE, mec="white", mew=0.6, ls="none", clip_on=False)
    a.plot([250.0], [float(ct["cert_hgo"])], marker="<", ms=5, mfc=ORANGE, mec="white", mew=0.6, ls="none", clip_on=False,
           label="certified upper bounds after the deadline")
    a.semilogy(t[n], ct["max_kalman_steady_state"][n], color=BLUE, lw=0.7, alpha=0.65, label="Kalman, largest observed")
    a.semilogy(t[n], ct["max_hgo_worst_case_tuned"][n], color=ORANGE, lw=0.7, ls="--", alpha=0.65, label="observer, largest observed")
    a.axvline(C["setting"]["deadline"] * 1e3, color=MUTED, lw=0.6, ls=(0, (4, 2)))
    a.set_ylim(8e-3, 5); a.set_xlabel("time [ms]"); a.set_ylabel("speed error [rad/s]"); a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), handlelength=1.8); tag(a, "(b)")
    a = ax[2]
    ex = C["exact"]
    names = [("kalman_steady_state", "Kalman, constant gain", BLUE, "o"), ("hgo_mse_tuned", "observer, MSE-tuned", ORANGE, "s"),
             ("hgo_worst_case_tuned", "observer, worst-case", ORANGE, "D")]
    m0 = min(ex[k]["mse_steady"] for k, *_ in names); w0 = min(ex[k]["certified_post_deadline"] for k, *_ in names)
    for i, (k, lab, col, mk) in enumerate(names):
        a.plot([ex[k]["mse_steady"] / m0], [1.0 - 0.18 * (i - 1)], marker=mk, ms=5.5, mfc=col, mec="white", mew=0.9, ls="none", label=lab)
        a.plot([ex[k]["certified_post_deadline"] / w0], [0.0 - 0.18 * (i - 1)], marker=mk, ms=5.5, mfc=col, mec="white", mew=0.9, ls="none")
        a.text(ex[k]["mse_steady"] / m0 + 0.012, 1.0 - 0.18 * (i - 1), f"{ex[k]['mse_steady'] / m0:.2f}", fontsize=7.5, va="center", color=INK)
        a.text(ex[k]["certified_post_deadline"] / w0 + 0.012, 0.0 - 0.18 * (i - 1), f"{ex[k]['certified_post_deadline'] / w0:.2f}", fontsize=7.5, va="center", color=INK)
    a.set_yticks([1.0, 0.0]); a.set_yticklabels(["mean-square\nerror,\nexact", "guaranteed\nlimit,\ncertified"]); a.set_ylim(-0.5, 1.5); a.set_xlim(0.97, 1.45)
    a.axvline(1.0, color=MUTED, lw=0.6); a.set_xlabel("relative to the best"); a.legend(loc="upper right", bbox_to_anchor=(1.04, -0.24), handletextpad=0.3); a.grid(axis="y", visible=False); tag(a, "(c)")
    fig.tight_layout(w_pad=1.2); fig.savefig(FIG / "fig_common.pdf"); plt.close(fig)


def _joint_sketch(a):
    a.set_aspect("equal"); a.axis("off"); a.set_xlim(-1.5, 1.9); a.set_ylim(-1.75, 0.55)
    th = math.radians(38)
    a.add_patch(Rectangle((-0.5, 0.0), 1.0, 0.14, fc="#e9e8e3", ec=MUTED, lw=0.6, hatch="////"))
    a.plot([0, 0], [0, -1.2], color=MUTED, lw=0.6, ls=(0, (3, 2)))
    x, y = 1.1 * math.sin(th), -1.1 * math.cos(th)
    a.plot([0, x], [0, y], color=INK, lw=3.0, solid_capstyle="round")
    a.add_patch(Circle((0, 0), 0.085, fc="white", ec=INK, lw=1.0, zorder=5)); a.add_patch(Circle((x, y), 0.12, fc=MUTED, ec=INK, lw=0.8, zorder=5))
    a.add_patch(Arc((0, 0), 1.0, 1.0, angle=-90, theta1=0, theta2=38, color=INK, lw=0.8))
    a.text(0.12, -0.72, "$x_1$", fontsize=8)
    a.add_patch(FancyArrowPatch((x, y - 0.12), (x, y - 0.5), arrowstyle="-|>", mutation_scale=7, color=MUTED, lw=0.8)); a.text(x + 0.08, y - 0.42, "gravity", fontsize=7.5, color=MUTED)
    a.add_patch(Arc((0, 0), 0.46, 0.46, theta1=20, theta2=160, color=ORANGE, lw=1.2))
    a.add_patch(FancyArrowPatch((-0.2, 0.1), (-0.235, 0.03), arrowstyle="-|>", mutation_scale=6, color=ORANGE, lw=1.0))
    a.text(0.0, 0.33, "torque $u$ (applied, known)", fontsize=7.5, color=INK, ha="center")
    a.text(-1.5, -1.38, "measured: encoder $y_k = x_1(t_k) + v_k$", fontsize=7.5, color=INK, va="top")
    a.text(-1.5, -1.58, "hidden: velocity $x_2$ (estimated)", fontsize=7.5, color=INK, va="top")


def _moving_max(x, w):
    x = np.asarray(x); n = len(x) // w
    return x[: n * w].reshape(n, w).max(axis=1)


def fig_closed_loop():
    J = json.loads((RES / "joint.json").read_text()); jt = np.load(RES / "joint_traces.npz")
    fig = plt.figure(figsize=(W2, 4.7))
    gs = fig.add_gridspec(2, 6)
    a = fig.add_subplot(gs[0, 0:2]); _joint_sketch(a)
    a.text(-0.02, 1.03, "(a)", transform=a.transAxes, fontsize=9.5, fontweight="bold", ha="right", va="bottom")
    t = jt["t"]; ton = 1.1
    series = [("observer", "released observer in the loop", ORANGE, "-"), ("filtered_difference", "filtered difference in the loop", AQUA, "--"),
              ("state_feedback", "true state in the loop (reference)", INK, ":")]
    a = fig.add_subplot(gs[0, 2:])
    a.axvspan(ton, t[-1], color="#f1f0ea", lw=0, zorder=0); a.text(ton + 0.03, 1.07, "disturbance on", fontsize=7.5, color=MUTED)
    a.plot(t, jt["cl_ref"], color=VIOLET, lw=0.8, label="reference angle")
    for k, lab, col, ls in series:
        a.plot(t, jt[f"cl_{k}_x"][:, 0], color=col, ls=ls, lw=1.0, label=lab)
    a.set_ylim(-0.08, 1.18); a.set_ylabel("joint angle [rad]"); a.set_xlabel("time [s]"); a.set_xlim(-0.1, 4.9); a.set_xticks([0, 0.5, 1, 1.5, 2, 2.5]); a.legend(loc="center right", ncol=1); tag(a, "(b)")
    a = fig.add_subplot(gs[1, 0:2])
    names = ["observer", "filtered_difference", "state_feedback"]; lab = ["released\nobserver", "filtered\ndifference", "true\nstate"]; cols = [ORANGE, AQUA, MUTED]
    vals = [J["closed_loop"][k]["rms_tracking_error"] * 1e3 for k in names]
    a.bar(range(3), vals, width=0.5, color=cols)
    for i, v in enumerate(vals):
        a.text(i, v + 0.7, f"{v:.1f}", ha="center", fontsize=7.5, color=INK)
    a.set_xticks(range(3)); a.set_xticklabels(lab, fontsize=7.5); a.set_ylabel("RMS tracking error [mrad]"); a.set_ylim(0, max(vals) * 1.22); a.grid(axis="x", visible=False); tag(a, "(c)")
    a = fig.add_subplot(gs[1, 2:4])
    w = 25
    eo = np.abs(jt["cl_observer_x"][:, 1] - jt["cl_observer_xh"][:, 1]); ef = np.abs(jt["cl_filtered_difference_x"][:, 1] - jt["cl_filtered_difference_xh"][:, 1])
    tw = t[: (len(t) // w) * w].reshape(-1, w).mean(axis=1)
    a.axvspan(ton, t[-1], color="#f1f0ea", lw=0, zorder=0)
    a.semilogy(tw, _moving_max(ef, w), color=AQUA, ls="--", label="filtered difference")
    a.semilogy(tw, _moving_max(eo, w), color=ORANGE, label="released observer")
    a.axhline(J["certified_post_deadline"], color=ORANGE, lw=0.8, ls=(0, (1, 1.5))); a.text(0.2, J["certified_post_deadline"] * 1.18, "certified bound (observer)", fontsize=7.5, color=INK)
    a.set_ylim(6e-3, 3); a.set_xlabel("time [s]"); a.set_ylabel("velocity error, 25 ms maxima [rad/s]"); a.legend(loc="upper right"); tag(a, "(d)")
    a = fig.add_subplot(gs[1, 4:])
    n = (t >= 2.2) & (t <= 2.35)
    a.plot(t[n], jt["cl_filtered_difference_u"][n], color=AQUA, lw=0.9, ls="--", label="filtered difference")
    a.plot(t[n], jt["cl_observer_u"][n], color=ORANGE, lw=0.9, label="released observer")
    a.plot(t[n], jt["cl_state_feedback_u"][n], color=INK, lw=0.9, ls=":", label="true state")
    lo, hi = a.get_ylim(); a.set_ylim(lo, hi + 0.85 * (hi - lo))
    a.set_xlabel("time [s]"); a.set_ylabel("applied torque [N m]"); a.legend(loc="upper right"); tag(a, "(e)")
    fig.tight_layout(h_pad=1.5, w_pad=0.9); fig.savefig(FIG / "fig_closed_loop.pdf"); plt.close(fig)


def fig_evaluation():
    S = json.loads((RES / "test_summary.json").read_text())
    st, nl = S["structured"], S["nl"]
    fig, ax = plt.subplots(2, 2, figsize=(W2, 5.6), gridspec_kw={"width_ratios": [2.3, 1.0], "height_ratios": [7, 7]})

    def bars(a, labels, cov, inc, cnt, title, t):
        y = np.arange(len(labels))[::-1]
        a.barh(y + 0.19, [c or 0 for c in cov], height=0.36, color=BLUE, label="useful coverage (share of the requests that admit a design)")
        a.barh(y - 0.19, inc, height=0.36, color=ORANGE, label="incorrect releases (share of all requests)")
        for yi, c_, i_, k in zip(y, cov, inc, cnt):
            a.text((c_ or 0) + 0.012, yi + 0.19, "--" if c_ is None else f"{c_:.2f}", va="center", fontsize=7.5, color=INK)
            a.text(i_ + 0.012, yi - 0.19, k + " incorrect", va="center", fontsize=7.5, color=INK)
        a.set_yticks(y); a.set_yticklabels(labels, fontsize=7.5); a.set_xlim(0, 1.32); a.set_xlabel("proportion"); a.grid(axis="y", visible=False); tag(a, t)
        return y

    def side(a, y, vals, xlabel, title, t, fmt="{:.2f}"):
        a.barh(y, vals, height=0.5, color=VIOLET)
        for yi, v in zip(y, vals):
            a.text(v + 0.02 * max(vals), yi, fmt.format(v), va="center", fontsize=7.5, color=INK)
        a.set_yticks(y); a.set_yticklabels([]); a.set_xlim(0, max(vals) * 1.3 + 0.05); a.set_xlabel(xlabel); a.grid(axis="y", visible=False); tag(a, t)

    pol = [("coordinator", "full framework"), ("coordinator_limited_user", "full framework, user knows ranges only"), ("selector", "fixed-priority selector"), ("kalman_only", "Kalman specialist alone"), ("khalil_only", "Khalil specialist alone"),
           ("ask_all", "ask for every missing field"), ("default_fill", "never ask (defaults, midpoints)")]
    pol = [(k, l) for k, l in pol if k in st]
    y = bars(ax[0, 0], [l for _, l in pol], [st[k]["useful_coverage"] for k, _ in pol], [st[k]["incorrect_per_request"] for k, _ in pol],
             [f"{st[k]['incorrect_releases']}/{st[k]['n']}" for k, _ in pol], f"Structured specifications (n = {st['coordinator']['n']})", "(a)")
    side(ax[0, 1], y, [st[k]["questions_per_request"] for k, _ in pol], "questions per request", "Clarification effort", "(b)")
    tr = sorted(k.split("|")[0] for k in nl if k.startswith("final_") and k.endswith("|blind|coordinator"))
    rows = [(f"{t}|blind|coordinator", f"trained reader, seed {t[-1]}") for t in tr] + [
        ("select_neural_frozen_s0|blind|coordinator", "frozen vectors, linear heads"), ("rules|blind|coordinator", "keyword rules"),
        ("null|blind|coordinator", "reads nothing"), (f"{tr[0]}|blind|single_model", "single model decides (seed 0)")]
    rows = [(k, l) for k, l in rows if k in nl]
    y = bars(ax[1, 0], [l for _, l in rows], [nl[k]["useful_coverage"] for k, _ in rows], [nl[k]["incorrect_per_request"] for k, _ in rows],
             [f"{nl[k]['incorrect_releases']}/{nl[k]['n']}" for k, _ in rows], f"Worded requests, readback confirmed unread (n = {nl[rows[0][0]]['n']})", "(c)")
    att = [k.replace("|blind|", "|attentive|") for k, _ in rows]
    side(ax[1, 1], y, [nl[k]["edits_per_request"] for k in att], "corrections per request", "Attentive review", "(d)")
    h, l = ax[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, fontsize=7.8, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.965), h_pad=1.2, w_pad=0.8); fig.savefig(FIG / "fig_evaluation.pdf"); plt.close(fig)


def fig_ambiguity():
    """Sidebar figure S1: mean-square error of fixed gains over the admissible completions of the example in
    "The Same Data, Two Different Answers" (closed form, no stored results)."""
    r = np.linspace(1.0, 1.5, 201)
    best = (2 - r) * r / 2                                   # conditional-mean gain for each completion
    k4 = (1 - 0.25) ** 2 * (2 - r) + 0.25 ** 2 * r           # K = 1/4
    k2 = (1 - 0.5) ** 2 * (2 - r) + 0.5 ** 2 * r             # K = 1/2
    fig, a = plt.subplots(figsize=(3.3, 2.6))
    a.axhline(0.65, color=MUTED, lw=0.8, ls=(0, (1, 2)), label="requested level 0.65")
    a.plot(r, best, color=INK, lw=1.4, label="smallest attainable")
    a.plot(r, k4, color=BLUE, lw=1.4, label="fixed gain $K = 1/4$")
    a.plot(r, k2, color=ORANGE, lw=1.4, ls="--", label="fixed gain $K = 1/2$")
    a.text(1.06, 0.566, "$K = 1/4$", fontsize=7, color=BLUE, va="top")
    a.text(1.42, 0.512, "$K = 1/2$", fontsize=7, color=ORANGE, ha="right", va="bottom")
    a.text(1.015, 0.488, "smallest attainable", fontsize=7, color=INK, va="top")
    a.set_xlim(1.0, 1.5); a.set_ylim(0.33, 0.70)
    a.set_xlabel("noise variance $r$ of the completion"); a.set_ylabel("mean-square error")
    a.legend(loc="upper center", bbox_to_anchor=(0.45, -0.24), fontsize=6.6, handlelength=1.8, ncol=2, columnspacing=1.0)
    fig.tight_layout(); fig.savefig(FIG / "fig_ambiguity.pdf"); plt.close(fig)

if __name__ == "__main__":
    done = []
    for name, fn, need in (("fidelity", fig_fidelity, ["joint.json", "boom_traces.npz"]), ("common", fig_common, ["common_domain.json"]),
                           ("closed_loop", fig_closed_loop, ["joint.json", "joint_traces.npz"]), ("evaluation", fig_evaluation, ["test_summary.json"]), ("ambiguity", fig_ambiguity, [])):
        if all((RES / n).exists() for n in need):
            fn(); done.append(name)
    print("figures written:", done)
