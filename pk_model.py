"""
pk_model.py — Two-Compartment Pharmacokinetic ODE System
=========================================================

BIOLOGY:
  Central compartment (1) = blood plasma + highly perfused organs (liver, kidneys, lungs)
  Peripheral compartment (2) = muscle, fat, connective tissue (slower equilibration)

ODE SYSTEM:
  dC1/dt = -(k10 + k12)*C1 + k21*C2 + input(t)/V1
  dC2/dt =  k12*C1 - k21*C2

PARAMETERS:
  k10  [1/hr] — elimination from central (= CL / V1)
  k12  [1/hr] — transfer: central → peripheral
  k21  [1/hr] — transfer: peripheral → central
  V1   [L]    — volume of central compartment
  V2   [L]    — volume of peripheral compartment (= V1 * k12/k21 at steady state)

UNITS THROUGHOUT:
  Concentration : mg/L  (= µg/mL)
  Time          : hr
  Dose          : mg
  Rate          : mg/hr
"""

import numpy as np
from scipy.integrate import solve_ivp


# ---------------------------------------------------------------------------
# Core ODE right-hand side
# ---------------------------------------------------------------------------

def _ode_system(t, y, k10, k12, k21, V1, input_fn):
    """
    Two-compartment ODE — called by solve_ivp at each time step.

    y[0] = C1 (central concentration, mg/L)
    y[1] = C2 (peripheral concentration, mg/L)

    input_fn(t) returns the drug input RATE into central compartment (mg/hr).
    """
    C1, C2 = y
    dC1_dt = -(k10 + k12) * C1 + k21 * C2 + input_fn(t) / V1
    dC2_dt =  k12 * C1 - k21 * C2
    return [dC1_dt, dC2_dt]


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

def solve_2cmt(t_span, t_eval, input_fn, k10, k12, k21, V1,
               C1_0=0.0, C2_0=0.0):
    """
    Integrate the two-compartment ODE system.

    Parameters
    ----------
    t_span   : (t_start, t_end) in hours
    t_eval   : 1-D array of time points at which to record solution
    input_fn : callable — input_fn(t) → drug input rate (mg/hr) at time t
    k10, k12, k21 : rate constants (1/hr)
    V1       : central volume (L)
    C1_0, C2_0 : initial concentrations (mg/L), default 0

    Returns
    -------
    t  : time array (hr)
    C1 : central concentration (mg/L)
    C2 : peripheral concentration (mg/L)
    """
    sol = solve_ivp(
        fun=lambda t, y: _ode_system(t, y, k10, k12, k21, V1, input_fn),
        t_span=t_span,
        y0=[C1_0, C2_0],
        t_eval=t_eval,
        method="RK45",       # explicit Runge-Kutta — good for non-stiff PK
        rtol=1e-8,           # tight tolerances for accuracy
        atol=1e-10,
        dense_output=False,
    )
    return sol.t, sol.y[0], sol.y[1]


# ---------------------------------------------------------------------------
# Derived parameters (useful to display alongside plots)
# ---------------------------------------------------------------------------

def volume_peripheral(V1, k12, k21):
    """V2 = V1 * k12 / k21  [L] — steady-state peripheral volume."""
    return V1 * k12 / k21


def volume_distribution_ss(V1, k12, k21):
    """Vss = V1 + V2  [L] — total volume at steady state."""
    return V1 + volume_peripheral(V1, k12, k21)


def clearance(V1, k10):
    """CL = V1 * k10  [L/hr] — systemic clearance."""
    return V1 * k10


def alpha_beta_exponents(k10, k12, k21):
    """
    Macro rate constants for the biexponential decay:
      C1(t) = A*exp(-alpha*t) + B*exp(-beta*t)

    alpha = fast distribution phase
    beta  = slow elimination phase  (terminal half-life = ln2/beta)
    """
    s = k10 + k12 + k21
    alpha = (s + np.sqrt(s**2 - 4 * k10 * k21)) / 2
    beta  = (s - np.sqrt(s**2 - 4 * k10 * k21)) / 2
    return alpha, beta


def terminal_half_life(k10, k12, k21):
    """t½β = ln(2) / beta  [hr]"""
    _, beta = alpha_beta_exponents(k10, k12, k21)
    return np.log(2) / beta


def distribution_half_life(k10, k12, k21):
    """t½α = ln(2) / alpha  [hr]"""
    alpha, _ = alpha_beta_exponents(k10, k12, k21)
    return np.log(2) / alpha
