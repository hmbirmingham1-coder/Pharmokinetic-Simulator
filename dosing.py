"""
dosing.py — Drug Input Functions
=================================

Each function returns a callable  input_fn(t) → mg/hr
that is passed into the ODE solver.

Dosing routes implemented:
  1. IV bolus       — instantaneous injection modeled as very-short infusion
  2. IV infusion    — constant rate over a defined window
  3. Oral           — first-order absorption (ka), pre-solved analytically
                      and injected into the ODE as a rate term
  4. Multi-dose     — any of the above repeated on a fixed interval

CLINICAL CONTEXT:
  IV bolus    → anesthetics, emergency drugs, direct blood delivery
  IV infusion → vancomycin, chemotherapy — controlled steady-state
  Oral        → most outpatient drugs — stomach absorption adds delay (Tmax)
"""

import numpy as np


# ---------------------------------------------------------------------------
# 1. IV Bolus
# ---------------------------------------------------------------------------

def iv_bolus(dose_mg, bolus_duration_hr=0.0167):
    """
    Model an IV bolus as a very short constant infusion (default 1 minute = 0.0167 hr).

    Why not a true delta function? ODE solvers need continuous inputs —
    a 1-min infusion is clinically equivalent and numerically stable.

    Returns
    -------
    input_fn : t → mg/hr (rate into central compartment)
    """
    rate = dose_mg / bolus_duration_hr

    def input_fn(t):
        return rate if 0 <= t <= bolus_duration_hr else 0.0

    return input_fn


# ---------------------------------------------------------------------------
# 2. IV Infusion
# ---------------------------------------------------------------------------

def iv_infusion(dose_mg, infusion_duration_hr):
    """
    Constant-rate IV infusion over a defined window.

    rate = dose / duration  (mg/hr)

    Clinical example: vancomycin 1000 mg over 1 hour → rate = 1000 mg/hr
    """
    rate = dose_mg / infusion_duration_hr

    def input_fn(t):
        return rate if 0 <= t <= infusion_duration_hr else 0.0

    return input_fn


# ---------------------------------------------------------------------------
# 3. Oral (first-order absorption)
# ---------------------------------------------------------------------------

def oral_dose(dose_mg, F, ka):
    """
    Oral dosing: drug absorbed from GI tract with first-order rate constant ka.

    The amount remaining in the GI tract at time t:
      A_gut(t) = F * dose * exp(-ka * t)

    The rate of absorption into plasma (= rate into central compartment):
      input(t) = ka * A_gut(t) = F * dose * ka * exp(-ka * t)

    Parameters
    ----------
    dose_mg : administered dose (mg)
    F       : bioavailability (0–1) — fraction that reaches systemic circulation
    ka      : absorption rate constant (1/hr)

    Note: F < 1 due to first-pass hepatic metabolism and incomplete GI absorption.
    """
    absorbed = F * dose_mg

    def input_fn(t):
        if t < 0:
            return 0.0
        return ka * absorbed * np.exp(-ka * t)

    return input_fn


# ---------------------------------------------------------------------------
# 4. Multi-dose (superposition)
# ---------------------------------------------------------------------------

def multi_dose(single_dose_fn_builder, dose_times_hr, **kwargs):
    """
    Combine multiple doses by summing their individual input functions.

    Works because the system is LINEAR — input functions superpose directly.

    Parameters
    ----------
    single_dose_fn_builder : one of {iv_bolus, iv_infusion, oral_dose}
    dose_times_hr          : list of times (hr) when each dose is administered
    **kwargs               : arguments forwarded to single_dose_fn_builder

    Example
    -------
    # 500 mg oral q8h × 6 doses
    fn = multi_dose(oral_dose, [0, 8, 16, 24, 32, 40],
                    dose_mg=500, F=0.85, ka=1.2)
    """
    fns = []
    for t_dose in dose_times_hr:
        # Build a time-shifted version of the single-dose input function
        base_fn = single_dose_fn_builder(**kwargs)

        def shifted(t, _fn=base_fn, _t0=t_dose):
            tau = t - _t0
            return _fn(tau) if tau >= 0 else 0.0

        fns.append(shifted)

    def input_fn(t):
        return sum(f(t) for f in fns)

    return input_fn


# ---------------------------------------------------------------------------
# Convenience: build dose time list
# ---------------------------------------------------------------------------

def regular_dosing(n_doses, interval_hr, start_hr=0.0):
    """
    Return a list of dose times for n_doses given every interval_hr.

    Example: regular_dosing(6, 8) → [0, 8, 16, 24, 32, 40]
    """
    return [start_hr + i * interval_hr for i in range(n_doses)]
