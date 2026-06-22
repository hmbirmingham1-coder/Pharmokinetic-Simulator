
"""
visualization.py — Publication-Quality PK Plots
================================================

Each function produces a self-contained, annotated figure suitable for
a README hero image or a Jupyter notebook.

Plot types:
  1. single_dose_plot   — C1 + C2 vs time, annotated with Cmax/Tmax/AUC
  2. multi_dose_plot    — accumulation to steady state, trough lines
  3. route_comparison   — IV bolus vs IV infusion vs oral side by side
  4. parameter_sensitivity — vary one parameter, overlay curves (spider plot)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch

from pk_model import solve_2cmt, terminal_half_life, distribution_half_life, clearance
from dosing import iv_bolus, iv_infusion, oral_dose, multi_dose, regular_dosing
from analysis import auc_trapz, auc_extrap_inf, cmax_tmax


# ---------------------------------------------------------------------------
# Style defaults
# ---------------------------------------------------------------------------

BLUE   = "#2563eb"
GREEN  = "#16a34a"
ORANGE = "#ea580c"
RED    = "#dc2626"
PURPLE = "#7c3aed"
GRAY   = "#6b7280"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
})


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _param_box(ax, text, loc="upper right"):
    """Add a styled parameter info box to an axes."""
    x = 0.98 if "right" in loc else 0.02
    ha = "right" if "right" in loc else "left"
    ax.text(x, 0.96, text, transform=ax.transAxes,
            fontsize=8, va="top", ha=ha,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow",
                      edgecolor="#d1d5db", alpha=0.9))


# ---------------------------------------------------------------------------
# 1. Single Dose — flagship plot
# ---------------------------------------------------------------------------

def single_dose_plot(k10=0.12, k12=0.3, k21=0.08, V1=10.0,
                     dose_mg=1000, route="iv_bolus",
                     ka=1.2, F=0.85, infusion_hr=1.0,
                     t_end=48, save_path="single_dose.png"):
    """
    Four-panel plot:
      Top-left  : C1 (central) with Cmax/Tmax/AUC annotations
      Top-right : C2 (peripheral) showing tissue equilibration delay
      Bottom-left: log-scale C1 — shows biexponential decay clearly
      Bottom-right: parameter summary table
    """
    t_eval = np.linspace(0, t_end, 2000)

    if route == "iv_bolus":
        input_fn = iv_bolus(dose_mg)
        route_label = f"IV Bolus {dose_mg} mg"
    elif route == "iv_infusion":
        input_fn = iv_infusion(dose_mg, infusion_hr)
        route_label = f"IV Infusion {dose_mg} mg over {infusion_hr} hr"
    else:
        input_fn = oral_dose(dose_mg, F, ka)
        route_label = f"Oral {dose_mg} mg (F={F}, ka={ka}/hr)"

    t, C1, C2 = solve_2cmt((0, t_end), t_eval, input_fn, k10, k12, k21, V1)

    cmax1, tmax1 = cmax_tmax(t, C1)
    cmax2, tmax2 = cmax_tmax(t, C2)
    auc1 = auc_trapz(t, C1)
    cl = clearance(V1, k10)
    t12_dist = distribution_half_life(k10, k12, k21)
    t12_term = terminal_half_life(k10, k12, k21)

    fig = plt.figure(figsize=(14, 9))
    fig.suptitle(f"Two-Compartment PK Model — {route_label}",
                 fontsize=14, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    # ── Panel 1: C1 linear scale ──────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t, C1, color=BLUE, linewidth=2, label="C1 (central/plasma)")
    ax1.fill_between(t, C1, alpha=0.1, color=BLUE, label=f"AUC = {auc1:.1f} mg·hr/L")

    ax1.axvline(tmax1, color=ORANGE, linestyle="--", alpha=0.6)
    ax1.axhline(cmax1, color=ORANGE, linestyle="--", alpha=0.6)
    ax1.plot(tmax1, cmax1, "o", color=ORANGE, markersize=8, zorder=5)
    ax1.annotate(f"Cmax={cmax1:.2f}\nTmax={tmax1:.1f} hr",
                 xy=(tmax1, cmax1), xytext=(tmax1 + t_end*0.04, cmax1 * 0.92),
                 fontsize=8, color="darkorange")

    ax1.set_xlabel("Time (hr)")
    ax1.set_ylabel("Concentration (mg/L)")
    ax1.set_title("Central Compartment (C1) — Plasma", fontweight="bold")
    ax1.legend(fontsize=8, loc="upper right")
    ax1.set_xlim(0, t_end)
    ax1.set_ylim(bottom=0)

    # ── Panel 2: C2 linear scale ──────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(t, C2, color=GREEN, linewidth=2, label="C2 (peripheral/tissue)")
    ax2.plot(tmax2, cmax2, "o", color=GREEN, markersize=8, zorder=5)
    ax2.annotate(f"Cmax={cmax2:.2f}\nTmax={tmax2:.1f} hr\n(tissue delay)",
                 xy=(tmax2, cmax2), xytext=(tmax2 + t_end*0.04, cmax2 * 0.90),
                 fontsize=8, color="darkgreen")
    ax2.plot(t, C1, color=BLUE, linewidth=1, alpha=0.35, linestyle="--",
             label="C1 (reference)")

    ax2.set_xlabel("Time (hr)")
    ax2.set_ylabel("Concentration (mg/L)")
    ax2.set_title("Peripheral Compartment (C2) — Tissue", fontweight="bold")
    ax2.legend(fontsize=8)
    ax2.set_xlim(0, t_end)
    ax2.set_ylim(bottom=0)

    # ── Panel 3: Semi-log C1 — biexponential decay ────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    mask = C1 > 1e-4
    ax3.semilogy(t[mask], C1[mask], color=BLUE, linewidth=2)
    ax3.set_xlabel("Time (hr)")
    ax3.set_ylabel("log Concentration (mg/L)")
    ax3.set_title("Semi-log C1 — Biexponential Decay", fontweight="bold")
    ax3.annotate(f"α phase (t½α ≈ {t12_dist:.2f} hr)\ndistribution",
                 xy=(t_end * 0.05, C1[mask][int(len(C1[mask]) * 0.05)]),
                 fontsize=8, color=GRAY)
    ax3.annotate(f"β phase (t½β ≈ {t12_term:.2f} hr)\nelimination",
                 xy=(t_end * 0.6, C1[mask][int(len(C1[mask]) * 0.75)]),
                 fontsize=8, color=GRAY)

    # ── Panel 4: Parameter table ───────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("off")

    table_data = [
        ["Parameter", "Value", "Units", "Meaning"],
        ["k10", f"{k10}", "1/hr", "Elimination from central"],
        ["k12", f"{k12}", "1/hr", "Central → Peripheral"],
        ["k21", f"{k21}", "1/hr", "Peripheral → Central"],
        ["V1",  f"{V1}", "L",    "Central volume"],
        ["CL",  f"{cl:.2f}", "L/hr", "Systemic clearance"],
        ["t½α", f"{t12_dist:.2f}", "hr", "Distribution half-life"],
        ["t½β", f"{t12_term:.2f}", "hr", "Terminal half-life"],
        ["Cmax (C1)", f"{cmax1:.3f}", "mg/L", "Peak plasma conc."],
        ["AUC (C1)", f"{auc1:.1f}", "mg·hr/L", "Exposure"],
    ]

    tbl = ax4.table(
        cellText=table_data[1:],
        colLabels=table_data[0],
        loc="center",
        cellLoc="left",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.1, 1.35)

    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor("#dbeafe")
            cell.set_text_props(fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#f8fafc")
        cell.set_edgecolor("#e2e8f0")

    ax4.set_title("Model Parameters & Derived Metrics", fontweight="bold", pad=12)

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[Saved] {save_path}")
    return fig


# ---------------------------------------------------------------------------
# 2. Multi-Dose Accumulation
# ---------------------------------------------------------------------------

def multi_dose_plot(k10=0.12, k12=0.3, k21=0.08, V1=10.0,
                    dose_mg=1000, n_doses=8, interval_hr=12,
                    route="iv_bolus", ka=1.2, F=0.85,
                    save_path="multi_dose.png"):
    """
    Show drug accumulation across multiple doses.

    Key teaching point: after ~5 terminal half-lives, the trough concentrations
    plateau — this is steady state. The accumulation index predicts how much
    higher steady-state is compared to a single dose.
    """
    t12_term = terminal_half_life(k10, k12, k21)
    t_end = n_doses * interval_hr + interval_hr
    t_eval = np.linspace(0, t_end, 5000)
    dose_times = regular_dosing(n_doses, interval_hr)

    if route == "iv_bolus":
        input_fn = multi_dose(iv_bolus, dose_times, dose_mg=dose_mg)
        route_label = f"IV Bolus {dose_mg} mg q{interval_hr}h × {n_doses}"
    else:
        input_fn = multi_dose(oral_dose, dose_times,
                              dose_mg=dose_mg, F=F, ka=ka)
        route_label = f"Oral {dose_mg} mg q{interval_hr}h × {n_doses}"

    t, C1, C2 = solve_2cmt((0, t_end), t_eval, input_fn, k10, k12, k21, V1)

    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    fig.suptitle(f"Multiple Dosing — {route_label}\n"
                 f"(t½β = {t12_term:.1f} hr → SS at ~{5*t12_term:.0f} hr)",
                 fontsize=13, fontweight="bold")

    for ax, C, color, label in zip(
        axes,
        [C1, C2],
        [BLUE, GREEN],
        ["Central (C1) — Plasma", "Peripheral (C2) — Tissue"]
    ):
        ax.plot(t, C, color=color, linewidth=1.8, label=label)
        for i, td in enumerate(dose_times):
            ax.axvline(td, color=GRAY, linestyle=":", alpha=0.5,
                       linewidth=0.8, label="Dose" if i == 0 else "")

        # Shade the last interval as "steady state region"
        ss_start = dose_times[-2] if len(dose_times) >= 2 else 0
        ax.axvspan(ss_start, t_end, alpha=0.07, color=color, label="Near steady state")

        ax.set_ylabel("Concentration (mg/L)", fontsize=10)
        ax.legend(fontsize=9, loc="upper left")
        ax.set_ylim(bottom=0)

    axes[1].set_xlabel("Time (hr)", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[Saved] {save_path}")
    return fig


# ---------------------------------------------------------------------------
# 3. Route Comparison
# ---------------------------------------------------------------------------

def route_comparison_plot(k10=0.12, k12=0.3, k21=0.08, V1=10.0,
                          dose_mg=1000, ka=1.2, F=0.85, infusion_hr=1.0,
                          t_end=36, save_path="route_comparison.png"):
    """
    Overlay all three dosing routes on one plot.

    Teaching points:
      IV bolus   → instant Cmax, highest peak, fastest drop
      IV infusion → Cmax delayed to end of infusion, lower peak, same AUC
      Oral       → Cmax most delayed (absorption + distribution), F < 1 → lower AUC
    """
    t_eval = np.linspace(0, t_end, 2000)

    routes = {
        "IV Bolus": (iv_bolus(dose_mg), BLUE),
        f"IV Infusion ({infusion_hr} hr)": (iv_infusion(dose_mg, infusion_hr), RED),
        f"Oral (F={F}, ka={ka})": (oral_dose(dose_mg, F, ka), GREEN),
    }

    fig, (ax_c1, ax_c2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Dosing Route Comparison — Same Dose, Different Kinetics",
                 fontsize=13, fontweight="bold")

    for label, (input_fn, color) in routes.items():
        t, C1, C2 = solve_2cmt((0, t_end), t_eval, input_fn, k10, k12, k21, V1)
        cmax1, tmax1 = cmax_tmax(t, C1)
        auc1 = auc_trapz(t, C1)
        ax_c1.plot(t, C1, color=color, linewidth=2,
                   label=f"{label}\nCmax={cmax1:.2f}, AUC={auc1:.0f}")
        ax_c2.plot(t, C2, color=color, linewidth=2, label=label, linestyle="--")

    ax_c1.set_title("Central Compartment (C1)", fontweight="bold")
    ax_c1.set_xlabel("Time (hr)")
    ax_c1.set_ylabel("Concentration (mg/L)")
    ax_c1.legend(fontsize=8)
    ax_c1.set_xlim(0, t_end)
    ax_c1.set_ylim(bottom=0)

    ax_c2.set_title("Peripheral Compartment (C2)", fontweight="bold")
    ax_c2.set_xlabel("Time (hr)")
    ax_c2.set_ylabel("Concentration (mg/L)")
    ax_c2.legend(fontsize=9)
    ax_c2.set_xlim(0, t_end)
    ax_c2.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[Saved] {save_path}")
    return fig


# ---------------------------------------------------------------------------
# 4. Parameter Sensitivity (Spider Plot)
# ---------------------------------------------------------------------------

def sensitivity_plot(param="k10", values=None,
                     k10=0.12, k12=0.3, k21=0.08, V1=10.0,
                     dose_mg=1000, t_end=36,
                     save_path="sensitivity.png"):
    """
    Vary one parameter across a range, overlay the resulting C1 curves.

    Reveals how each parameter shapes the concentration-time profile —
    essential for understanding drug behavior and building intuition.
    """
    if values is None:
        defaults = {"k10": [0.05, 0.10, 0.15, 0.25, 0.40],
                    "k12": [0.1, 0.2, 0.3, 0.5, 0.8],
                    "k21": [0.04, 0.08, 0.12, 0.20],
                    "V1":  [5, 10, 20, 40]}
        values = defaults.get(param, [0.1, 0.2, 0.3])

    t_eval = np.linspace(0, t_end, 1000)
    input_fn = iv_bolus(dose_mg)
    colors = plt.cm.viridis(np.linspace(0.15, 0.9, len(values)))

    fig, ax = plt.subplots(figsize=(10, 6))

    for val, color in zip(values, colors):
        params = dict(k10=k10, k12=k12, k21=k21, V1=V1)
        params[param] = val
        t, C1, _ = solve_2cmt((0, t_end), t_eval, input_fn, **params)
        ax.plot(t, C1, color=color, linewidth=2, label=f"{param} = {val}")

    units = {"k10": "1/hr", "k12": "1/hr", "k21": "1/hr", "V1": "L"}
    meanings = {
        "k10": "↑ k10 → faster elimination (steeper late slope)",
        "k12": "↑ k12 → faster distribution to tissue (lower early C1)",
        "k21": "↑ k21 → faster tissue return (higher late C1)",
        "V1":  "↑ V1  → larger volume → lower initial C1",
    }

    ax.set_xlabel("Time (hr)", fontsize=12)
    ax.set_ylabel("C1 — Plasma Concentration (mg/L)", fontsize=12)
    ax.set_title(f"Parameter Sensitivity: {param} ({units.get(param, '')})\n"
                 f"{meanings.get(param, '')}",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, title=f"{param} values")
    ax.set_xlim(0, t_end)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[Saved] {save_path}")
    return fig


if __name__ == "__main__":
    print("=== PharmaSim — Running all plots ===\n")
    print("1/4  Single IV bolus dose...")
    single_dose_plot()
    print("2/4  Multiple dose accumulation...")
    multi_dose_plot()
    print("3/4  Route comparison (IV bolus vs infusion vs oral)...")
    route_comparison_plot()
    print("4/4  Parameter sensitivity (k10)...")
    sensitivity_plot(param="k10")
    print("\nDone. Check the pharmasim/ folder for saved PNGs.")
