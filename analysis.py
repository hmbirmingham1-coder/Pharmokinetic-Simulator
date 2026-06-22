"""
analysis.py — Pharmacokinetic Analysis Functions
=================================================

Given a solved concentration-time profile (t, C arrays from the ODE solver),
compute the standard PK metrics used in drug development and clinical dosing.

All metrics are computed NUMERICALLY from the solver output, not analytically —
this is the correct approach when you have a general ODE (no closed-form solution).

Metrics implemented:
  AUC    — area under the concentration-time curve (trapezoidal)
  AUC∞   — AUC extrapolated to infinity
  Cmax   — peak concentration
  Tmax   — time of peak concentration
  t½     — terminal half-life (from log-linear regression of elimination phase)
  Ctrough — trough concentration (just before next dose)
  Css    — steady-state average concentration (multiple dosing)
"""

import numpy as np
from scipy.stats import linregress


# ---------------------------------------------------------------------------
# AUC — Trapezoidal method
# ---------------------------------------------------------------------------

def auc_trapz(t, C):
    """
    Compute AUC using the linear trapezoidal rule.

    AUC = Σ (t[i+1] - t[i]) * (C[i] + C[i+1]) / 2

    Units: (mg/L) * hr = mg·hr/L

    Why trapezoidal?
      Numerically stable, model-independent, used in FDA NDA submissions.
      The log-trapezoidal method is better during declining phases but
      trapezoidal is standard for a first implementation.
    """
    return np.trapz(C, t)


def auc_extrap_inf(t, C, terminal_fraction=0.2):
    """
    AUC from t=0 to infinity.

    AUC∞ = AUC_obs + C_last / lambda_z

    lambda_z is estimated by log-linear regression on the terminal
    (elimination) phase — the last `terminal_fraction` of the time course.
    """
    auc_obs = auc_trapz(t, C)

    # Fit terminal elimination phase
    n = len(t)
    start = int(n * (1 - terminal_fraction))
    t_term = t[start:]
    C_term = C[start:]

    # Remove zeros to avoid log(0)
    mask = C_term > 0
    if mask.sum() < 3:
        return auc_obs  # not enough points to extrapolate

    slope, _, _, _, _ = linregress(t_term[mask], np.log(C_term[mask]))
    lambda_z = -slope  # lambda_z > 0

    if lambda_z <= 0:
        return auc_obs

    C_last = C_term[mask][-1]
    auc_tail = C_last / lambda_z
    return auc_obs + auc_tail


# ---------------------------------------------------------------------------
# Cmax / Tmax
# ---------------------------------------------------------------------------

def cmax_tmax(t, C):
    """
    Return (Cmax, Tmax) — peak concentration and time to peak.

    For two-compartment models, C1 peaks earlier (distribution phase) and
    C2 peaks later (tissue equilibration). Report separately.
    """
    idx = np.argmax(C)
    return C[idx], t[idx]


# ---------------------------------------------------------------------------
# Terminal half-life
# ---------------------------------------------------------------------------

def terminal_half_life(t, C, terminal_fraction=0.25):
    """
    Estimate t½ from the terminal log-linear slope of the concentration curve.

    1. Take the last `terminal_fraction` of the time course (elimination phase)
    2. Fit log(C) vs t with linear regression → slope = -lambda_z
    3. t½ = ln(2) / lambda_z

    This is EXACTLY what pharmacokineticists do from clinical data.
    """
    n = len(t)
    start = int(n * (1 - terminal_fraction))
    t_term = t[start:]
    C_term = C[start:]

    mask = C_term > 1e-6
    if mask.sum() < 3:
        return np.nan

    slope, intercept, r, _, _ = linregress(t_term[mask], np.log(C_term[mask]))
    lambda_z = -slope

    if lambda_z <= 0:
        return np.nan

    return np.log(2) / lambda_z, r**2  # return (t½, R²) — R² checks fit quality


# ---------------------------------------------------------------------------
# Trough concentration
# ---------------------------------------------------------------------------

def ctrough(t, C, dose_interval_hr):
    """
    Find trough concentration — C just before the next dose.

    For steady-state analysis, find the minimum C in the last dosing interval.
    """
    t_last_dose = t[-1] - dose_interval_hr
    mask = t >= t_last_dose
    if mask.sum() == 0:
        return np.nan
    return np.min(C[mask])


# ---------------------------------------------------------------------------
# Steady-state metrics (multiple dosing)
# ---------------------------------------------------------------------------

def steady_state_avg(t, C, dose_interval_hr, n_intervals=1):
    """
    Average concentration over the last `n_intervals` dosing intervals.

    Css_avg = AUC_interval / tau

    For a drug at steady state (after ~5 half-lives), Css_avg is the
    concentration that determines therapeutic effect for many drug classes.
    """
    t_start = t[-1] - n_intervals * dose_interval_hr
    mask = t >= t_start
    if mask.sum() < 2:
        return np.nan
    auc_interval = auc_trapz(t[mask], C[mask])
    return auc_interval / (n_intervals * dose_interval_hr)


def accumulation_index(t_half, dose_interval_hr):
    """
    Accumulation index R = 1 / (1 - exp(-lambda_z * tau))

    Tells you how much drug accumulates relative to a single dose.
    R=1 → no accumulation (tau >> t½)
    R>1 → accumulation (tau < 5*t½)

    This is purely analytical — useful for dose regimen design.
    """
    lambda_z = np.log(2) / t_half
    return 1.0 / (1.0 - np.exp(-lambda_z * dose_interval_hr))


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

def pk_summary(t, C1, C2=None, dose_interval_hr=None, label="Central (C1)"):
    """
    Print a formatted PK summary for a concentration-time profile.
    """
    cmax, tmax = cmax_tmax(t, C1)
    auc = auc_trapz(t, C1)
    auc_inf = auc_extrap_inf(t, C1)
    t_half_result = terminal_half_life(t, C1)

    print(f"\n{'='*50}")
    print(f"  PK Summary — {label}")
    print(f"{'='*50}")
    print(f"  Cmax          : {cmax:.4f} mg/L")
    print(f"  Tmax          : {tmax:.2f} hr")
    print(f"  AUC (obs)     : {auc:.2f} mg·hr/L")
    print(f"  AUC∞ (extrap) : {auc_inf:.2f} mg·hr/L")

    if t_half_result is not np.nan and isinstance(t_half_result, tuple):
        t12, r2 = t_half_result
        print(f"  t½ (terminal) : {t12:.2f} hr  (R²={r2:.4f})")
    else:
        print(f"  t½ (terminal) : N/A")

    if dose_interval_hr:
        css = steady_state_avg(t, C1, dose_interval_hr)
        trough = ctrough(t, C1, dose_interval_hr)
        print(f"  Css (avg)     : {css:.4f} mg/L")
        print(f"  Ctrough       : {trough:.4f} mg/L")

    if C2 is not None:
        cmax2, tmax2 = cmax_tmax(t, C2)
        print(f"\n  Peripheral (C2):")
        print(f"  Cmax          : {cmax2:.4f} mg/L")
        print(f"  Tmax          : {tmax2:.2f} hr  (delayed vs C1 — tissue equilibration)")

    print(f"{'='*50}\n")
