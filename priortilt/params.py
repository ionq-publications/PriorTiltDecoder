# This file is part of PriorTiltDecoder.

# Copyright (c) 2026 IonQ, Inc., all rights reserved

# Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International License (CC BY-NC-SA 4.0).

# You may obtain a copy of the License at:
# https://creativecommons.org/licenses/by-nc-sa/4.0/

"""Optimized PriorTilt parameters of the three codes benchmarked in the paper, and the lookup rules that pick them.

PriorTilt's logical error rate depends strongly on its offset parameters Theta = (theta, pi, c_1, c_2), and the
optimum depends on the code AND on the physical error rate. Parameter names: ``threshold_factor`` = theta (the
error-prone condition S_j < theta * Lambda_j), ``prob_positive`` = pi (probability of the nonnegative offset),
``pos_coef`` = c_1 and ``neg_coef`` = c_2 (coefficients of the nonnegative and nonpositive offsets). The code
orders a tuple as (prob_positive, pos_coef, neg_coef, threshold_factor) = (pi, c_1, c_2, theta); the paper's
tables order it as (theta, pi, c_1, c_2).

This module records the optimized values of the paper's three-phase mode, the mode used for all results of the
paper: a piecewise-constant parameter schedule, i.e. the phase boundaries (b_1, b_2) (``ends``) and one parameter
tuple per phase (``THREE_PHASE_SCHEDULES``, the paper's table of optimized parameters; ``XYZ_THREE_PHASE_SCHEDULES``
for the XYZ decoding of BB144). The paper's uniform mode is the special case of a schedule with a single phase
(``{"ends": [], "phases": [[...]]}``); no uniform-mode parameters are provided because the paper reports no
uniform-mode results.

All sets were optimized on the standard XZ-decoding X- and Z-memory circuits provided in ``StimCircuit/`` (the
same parameters serve both bases), with ``initial_iters`` 60, ``round_len`` 6 and ``max_rounds`` 3000. A
different circuit, noise model or number of syndrome extraction cycles may need re-optimization.

Lookup (``tuned_config``): the physical error rate is matched to the closest optimized rate in the ratio sense; a
warning is emitted when ``p`` lies outside the optimized range.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple, Union

__all__ = [
    "CODES",
    "THREE_PHASE_SCHEDULES",
    "XYZ_THREE_PHASE_SCHEDULES",
    "xyz_schedule",
    "DEFAULT_MAX_ROUNDS",
    "DEFAULT_INITIAL_ITERS",
    "DEFAULT_ROUND_LEN",
    "PriorTiltConfig",
    "UntunedCodeError",
    "closest_rate",
    "optimized_rates",
    "normalize_code",
    "tuned_config",
    "three_phase_schedule",
]

# The paper's setting: I_0 = 60 ordinary BP iterations, modified-BP rounds of r = 6 iterations, at most R = 3000 rounds.
DEFAULT_MAX_ROUNDS = 3000
DEFAULT_INITIAL_ITERS = 60
DEFAULT_ROUND_LEN = 6

# Codes with optimized parameters: name -> (number of physical qubits, syndrome extraction cycles of the memory
# experiment, description).
CODES: Dict[str, Tuple[int, int, str]] = {
    "BB144": (144, 12, "bivariate bicycle code [[144,12,12]] (gross code), uniform circuit-level noise model of Bravyi et al."),
    "BB70": (70, 9, "bivariate bicycle code [[70,6,9]], moving-qubit circuit-level noise model of Tripier et al."),
    "GB102": (102, 9, "generalized bicycle code [[102,22,9]], moving-qubit circuit-level noise model of Tripier et al."),
}

_LENGTH_TO_CODE: Dict[int, str] = {n: name for name, (n, _, _) in CODES.items()}

# Three-phase mode: code -> physical error rate -> {"ends": [b1, b2], "phases": [Theta_1, Theta_2, Theta_3]}.
# Each tuple is (prob_positive, pos_coef, neg_coef, threshold_factor) = (pi, c_1, c_2, theta). Round tau uses
# Theta_1 for tau <= b1, Theta_2 for b1 < tau <= b2 and Theta_3 for tau > b2. "optimized_at" is the rate the
# schedule was optimized at (two GB102 rates share one set). These are exactly the parameters of the paper's table
# of optimized parameters (see also simulation_results/paper_data, column decoder_params).
THREE_PHASE_SCHEDULES: Dict[str, Dict[float, Dict[str, object]]] = {
    "BB144": {
        0.001: {
            "ends": [100, 800],
            "phases": [[0.2744, 0.5, 0.476, 1.4], [0.1949, 0.5, 0.7683, 1.19], [0.0977, 0.4626, 2.45, 2.5671]],
            "optimized_at": 0.001,
        },
        0.002: {
            "ends": [100, 800],
            "phases": [[0.196, 0.5, 0.56, 1.4], [0.1392, 0.5, 0.5488, 1.4], [0.0977, 0.3304, 2.45, 2.1755]],
            "optimized_at": 0.002,
        },
        0.003: {
            "ends": [120, 1000],
            "phases": [[0.196, 0.5, 0.392, 1.4], [0.118, 0.5, 0.56, 1.4], [0.1396, 0.2, 1.75, 2.1755]],
            "optimized_at": 0.003,
        },
        0.004: {
            "ends": [120, 1000],
            "phases": [[0.1372, 0.59, 0.392, 1.4], [0.118, 0.425, 0.56, 1.4], [0.1153, 0.2, 1.75, 1.8492]],
            "optimized_at": 0.004,
        },
        0.005: {
            "ends": [70, 1000],
            "phases": [[0.196, 0.5, 0.392, 1.4], [0.0826, 0.413, 0.476, 1.652], [0.1153, 0.17, 1.4875, 1.8492]],
            "optimized_at": 0.005,
        },
    },
    "BB70": {
        0.0005: {
            "ends": [80, 860],
            "phases": [[0.5476, 0.59, 0.476, 2.0], [0.5488, 0.2499, 1.029, 2.0], [0.2808, 0.3499, 3.4692, 2.36]],
            "optimized_at": 0.0005,
        },
        0.0008: {
            "ends": [80, 860],
            "phases": [[0.6442, 0.59, 0.476, 2.0], [0.5488, 0.2499, 1.029, 2.0], [0.2006, 0.2499, 2.478, 2.36]],
            "optimized_at": 0.0008,
        },
        0.001: {
            "ends": [80, 860],
            "phases": [[0.5459, 0.59, 0.56, 2.0], [0.5488, 0.2499, 1.029, 2.0], [0.2006, 0.2499, 2.478, 2.36]],
            "optimized_at": 0.001,
        },
        0.002: {
            "ends": [100, 1070],
            "phases": [[0.4626, 0.5, 0.392, 2.0], [0.392, 0.2499, 0.735, 2.0], [0.17, 0.1785, 1.77, 2.0]],
            "optimized_at": 0.002,
        },
    },
    "GB102": {
        0.0005: {
            "ends": [100, 1560],
            "phases": [[0.6997, 0.59, 0.392, 1.0115], [0.588, 0.595, 0.476, 1.7], [0.3009, 0.2832, 1.0413, 1.4]],
            "optimized_at": 0.0008,
        },
        0.0008: {
            "ends": [100, 1560],
            "phases": [[0.6997, 0.59, 0.392, 1.0115], [0.588, 0.595, 0.476, 1.7], [0.3009, 0.2832, 1.0413, 1.4]],
            "optimized_at": 0.0008,
        },
        0.001: {
            "ends": [100, 1560],
            "phases": [[0.4998, 0.59, 0.392, 1.19], [0.588, 0.425, 0.476, 1.7], [0.3009, 0.2832, 1.0413, 1.4]],
            "optimized_at": 0.001,
        },
        0.002: {
            "ends": [200, 1560],
            "phases": [[0.4213, 0.59, 0.2744, 0.98], [0.42, 0.425, 0.476, 2.36], [0.255, 0.2023, 1.4875, 1.4]],
            "optimized_at": 0.002,
        },
    },
}

#: XYZ decoding of BB144 at p = 0.003: both detector types are used jointly (the detectors after the first cycle
#: compare all stabilizer measurements), so the decoder works on the full, non-decomposed detector error model
#: (about 68,000 error mechanisms instead of 8,784) and can exploit the correlations between the X- and Z-error
#: components. Its own three-phase schedule with phase boundaries (50, 420) and R = 6000 rounds, as given in the
#: paper. Use PriorTilt(dem, schedule=xyz_schedule(...), max_rounds=6000). Not part of the tuned_config lookup,
#: which serves standard XZ decoding.
XYZ_THREE_PHASE_SCHEDULES: Dict[str, Dict[float, Dict[str, object]]] = {
    "BB144": {
        0.003: {
            "ends": [50, 420],
            "phases": [[0.07, 0.5, 0.8, 2.0], [0.1372, 0.5, 0.784, 0.686], [0.049, 0.5, 1.568, 1.372]],
            "max_rounds": 6000,
            "optimized_at": 0.003,
        },
    },
}

class UntunedCodeError(ValueError):
    """Raised when PriorTilt is asked for optimized parameters of a code it has not been optimized for."""


def _tuning_request(code: object) -> str:
    known = ", ".join(CODES)
    return (
        f"PriorTilt has no optimized parameters for code {code!r}. Its logical error rate depends strongly on the "
        f"offset parameters (prob_positive, pos_coef, neg_coef, threshold_factor) of each phase, so optimize a schedule "
        f"for your code and noise model before using it, then pass it explicitly (schedule=...). Codes with recorded "
        f"parameters: {known}."
    )


def normalize_code(code: Union[str, int]) -> str:
    """Canonical name ("BB144", "BB70", "GB102") for a code given by name (any case) or by number of qubits."""
    if isinstance(code, str):
        name = code.strip().upper()
        if name in CODES:
            return name
        raise UntunedCodeError(_tuning_request(code))
    if isinstance(code, int) and not isinstance(code, bool):
        if code in _LENGTH_TO_CODE:
            return _LENGTH_TO_CODE[code]
        raise UntunedCodeError(_tuning_request(code))
    raise TypeError(f"code must be a code name or a number of qubits, not {type(code).__name__}")


def closest_rate(p: float, rates: Iterable[float]) -> float:
    """The optimized rate closest to `p` in the ratio sense: argmin over r of |log(p / r)| (ties -> the lower rate)."""
    if not (p > 0):
        raise ValueError(f"the physical error rate must be positive, got {p}")
    rates = sorted(rates)
    if not rates:
        raise ValueError("no optimized rates")
    return min(rates, key=lambda r: (abs(math.log(p / r)), r))


def optimized_rates(code: Union[str, int]) -> List[float]:
    """Physical error rates with optimized three-phase parameters for `code`."""
    return sorted(THREE_PHASE_SCHEDULES.get(normalize_code(code), {}))


def _match_rate(name: str, p: float, rates: List[float]) -> float:
    r = closest_rate(p, rates)
    lo, hi = min(rates), max(rates)
    if p < lo * (1 - 1e-9) or p > hi * (1 + 1e-9):
        warnings.warn(
            f"PriorTilt parameters for {name} were optimized for physical error rates in [{lo:g}, {hi:g}]; "
            f"p={p:g} is outside that range, using the set optimized at {r:g}. The optimum drifts with the physical "
            f"error rate, so consider optimizing at p={p:g}.",
            stacklevel=4,
        )
    return r


def three_phase_schedule(code: Union[str, int], p: float) -> Tuple[Dict[str, object], float]:
    """(schedule dict with keys ends / phases, optimized rate used) for `code` at physical error rate `p`."""
    name = normalize_code(code)
    table = THREE_PHASE_SCHEDULES[name]
    r = _match_rate(name, p, list(table))
    sch = table[r]
    return {"ends": list(sch["ends"]), "phases": [list(t) for t in sch["phases"]]}, r


@dataclass
class PriorTiltConfig:
    """Resolved decoder configuration (what PriorTilt actually runs with): a three-phase (or, more generally, m-phase)
    parameter schedule and the iteration budget."""

    schedule: Dict[str, object]  # {"ends": [b1, b2, ...], "phases": [[prob_positive, pos_coef, neg_coef, threshold_factor], ...]}
    max_rounds: int = DEFAULT_MAX_ROUNDS
    initial_iters: int = DEFAULT_INITIAL_ITERS
    round_len: int = DEFAULT_ROUND_LEN
    code: Optional[str] = None
    p_requested: Optional[float] = None
    p_used: Optional[float] = None

    def validate(self) -> None:
        if not self.schedule or "ends" not in self.schedule or "phases" not in self.schedule:
            raise ValueError('a schedule is a dict {"ends": [b1, b2, ...], "phases": [[...], ...]}')
        ends = list(self.schedule["ends"])
        phases = self.schedule["phases"]
        if len(phases) != len(ends) + 1:
            raise ValueError("schedule needs len(ends) + 1 phases")
        if any(e <= 0 for e in ends) or any(b <= a for a, b in zip(ends, ends[1:])):
            raise ValueError("schedule ends (phase boundaries) must be positive and strictly increasing")
        if any(len(t) != 4 for t in phases):
            raise ValueError("each schedule phase is (prob_positive, pos_coef, neg_coef, threshold_factor) = (pi, c_1, c_2, theta)")
        for k in ("max_rounds", "initial_iters", "round_len"):
            if int(getattr(self, k)) < 1:
                raise ValueError(f"{k} must be >= 1")

    @property
    def num_phases(self) -> int:
        return len(self.schedule["phases"])

    def schedule_arrays(self) -> Tuple[List[int], List[float], List[float], List[float], List[float]]:
        """(ends, prob_positive, pos_coef, neg_coef, threshold_factor) lists for set_round_schedule."""
        phases = self.schedule["phases"]
        return (
            [int(e) for e in self.schedule["ends"]],
            [float(t[0]) for t in phases],
            [float(t[1]) for t in phases],
            [float(t[2]) for t in phases],
            [float(t[3]) for t in phases],
        )

    def as_dict(self) -> dict:
        """JSON-serializable record of the configuration (stored in the decoder_params column of result files)."""
        return {
            "initial_iters": self.initial_iters,
            "round_len": self.round_len,
            "max_rounds": self.max_rounds,
            "num_results": 1,
            "schedule": {"ends": list(self.schedule["ends"]), "phases": [list(t) for t in self.schedule["phases"]]},
        }

    def describe(self) -> str:
        where = f" for {self.code} at p={self.p_used:g}" if self.code else ""
        if self.p_requested is not None and self.p_used is not None and abs(self.p_requested - self.p_used) > 1e-12:
            where += f" (requested p={self.p_requested:g})"
        mode = {1: "uniform", 3: "three-phase"}.get(self.num_phases, f"{self.num_phases}-phase")
        return (
            f"PriorTilt {mode} mode{where}: phase boundaries {self.schedule['ends']}, parameters per phase "
            f"(prob_positive, pos_coef, neg_coef, threshold_factor) = {self.schedule['phases']}; max_rounds={self.max_rounds}"
        )


def tuned_config(
    code: Union[str, int],
    p: float,
    *,
    max_rounds: int = DEFAULT_MAX_ROUNDS,
    initial_iters: int = DEFAULT_INITIAL_ITERS,
    round_len: int = DEFAULT_ROUND_LEN,
) -> PriorTiltConfig:
    """The paper's optimized three-phase configuration for `code` at physical error rate `p` (closest optimized rate)."""
    name = normalize_code(code)
    sch, r = three_phase_schedule(name, p)
    cfg = PriorTiltConfig(
        schedule=sch, max_rounds=max_rounds, initial_iters=initial_iters, round_len=round_len, code=name, p_requested=p, p_used=r
    )
    cfg.validate()
    return cfg


def xyz_schedule(code: Union[str, int] = "BB144", p: float = 0.003) -> Dict[str, object]:
    """The three-phase schedule of the paper's XYZ decoding of BB144 at p = 0.003, as a dict {"ends", "phases"} for
    PriorTilt(schedule=...), together with the maximum number of rounds under "max_rounds" (R = 6000). Only
    (BB144, 0.003) exists."""
    name = normalize_code(code)
    table = XYZ_THREE_PHASE_SCHEDULES.get(name, {})
    if not table:
        raise UntunedCodeError(f"no XYZ schedule recorded for {name}; recorded: {sorted(XYZ_THREE_PHASE_SCHEDULES)}")
    rates = sorted(table)
    r = closest_rate(p, rates)
    if abs(math.log(p / r)) > 1e-9:
        raise UntunedCodeError(f"no XYZ schedule recorded for {name} at p={p:g}; recorded rates: {rates}")
    e = table[r]
    return {"ends": list(e["ends"]), "phases": [list(ph) for ph in e["phases"]], "max_rounds": int(e["max_rounds"])}
