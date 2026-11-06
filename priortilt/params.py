# This file is part of PriorTiltDecoder.

# Copyright (c) 2026 IonQ, Inc., all rights reserved

# Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International License (CC BY-NC-SA 4.0).

# You may obtain a copy of the License at:
# https://creativecommons.org/licenses/by-nc-sa/4.0/

"""Optimised PriorTilt parameters of the three codes benchmarked in the paper, and the lookup rules that pick them.

PriorTilt's logical error rate depends strongly on its tilt parameters, and the optimum depends on the code AND on
the physical error rate. This module records the fitted values for the paper's codes in both operating modes:

* **three-phase mode** (the paper's curves): a round schedule, i.e. two block boundaries and one
  (prob_positive, pos_coef, neg_coef, threshold_factor) tuple per block (``THREE_PHASE_SCHEDULES``);
* **uniform mode**: one tuple used in every tilt round (``UNIFORM_PARAMS``). Kept for reference; the three-phase
  mode is the better of the two at every benchmarked point and costs the same decoding time.

All sets were fitted on the standard X and Z memory experiments provided in ``StimCircuit/`` (the same parameters
serve both bases), with ``initial_iters`` 60, ``round_len`` 6 and ``max_rounds`` 3000. A different circuit, noise
model or number of syndrome rounds may need re-tuning.

Lookup (``tuned_config``): the physical error rate is matched to the closest fitted rate in the ratio sense; a
warning is emitted when ``p`` lies outside the fitted range.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple, Union

__all__ = [
    "CODES",
    "UNIFORM_PARAMS",
    "THREE_PHASE_SCHEDULES",
    "XYZ_THREE_PHASE_SCHEDULES",
    "xyz_schedule",
    "DEFAULT_MAX_ROUNDS",
    "DEFAULT_INITIAL_ITERS",
    "DEFAULT_ROUND_LEN",
    "PARAM_NAMES",
    "PriorTiltConfig",
    "UntunedCodeError",
    "closest_rate",
    "fitted_rates",
    "normalize_code",
    "tuned_config",
    "uniform_params",
    "three_phase_schedule",
]

# The paper's campaign: 60 plain BP iterations, tilt rounds of 6 iterations, at most 3000 tilt rounds.
DEFAULT_MAX_ROUNDS = 3000
DEFAULT_INITIAL_ITERS = 60
DEFAULT_ROUND_LEN = 6

# Codes with optimised parameters: name -> (number of physical qubits, syndrome rounds of the memory experiment, description).
CODES: Dict[str, Tuple[int, int, str]] = {
    "BB144": (144, 12, "bivariate bicycle [[144,12,12]], uniform circuit-level depolarising noise"),
    "BB70": (70, 9, "bivariate bicycle [[70,6,9]], trapped-ion circuit-level noise (FlyingIonicModel)"),
    "GB102": (102, 9, "generalised bicycle [[102,22,9]], trapped-ion circuit-level noise (FlyingIonicModel)"),
}

_LENGTH_TO_CODE: Dict[int, str] = {n: name for name, (n, _, _) in CODES.items()}

# Uniform mode: code -> physical error rate -> (prob_positive, pos_coef, neg_coef, threshold_factor).
UNIFORM_PARAMS: Dict[str, Dict[float, Dict[str, float]]] = {
    "BB144": {
        0.001: {"prob_positive": 0.2, "pos_coef": 0.5, "neg_coef": 1.0, "threshold_factor": 2.0},
        0.002: {"prob_positive": 0.1, "pos_coef": 0.5, "neg_coef": 0.8, "threshold_factor": 2.0},
        0.003: {"prob_positive": 0.1, "pos_coef": 0.5, "neg_coef": 0.8, "threshold_factor": 2.0},
        0.004: {"prob_positive": 0.1, "pos_coef": 0.5, "neg_coef": 0.8, "threshold_factor": 2.0},
        0.005: {"prob_positive": 0.1, "pos_coef": 0.5, "neg_coef": 0.8, "threshold_factor": 2.0},
    },
    "BB70": {
        0.0005: {"prob_positive": 0.4, "pos_coef": 0.5, "neg_coef": 0.8, "threshold_factor": 2.0},
        0.0008: {"prob_positive": 0.2, "pos_coef": 0.5, "neg_coef": 0.8, "threshold_factor": 2.0},
        0.001: {"prob_positive": 0.2, "pos_coef": 0.5, "neg_coef": 0.8, "threshold_factor": 2.0},
        0.002: {"prob_positive": 0.2, "pos_coef": 0.5, "neg_coef": 0.8, "threshold_factor": 2.0},
    },
    "GB102": {
        0.0005: {"prob_positive": 0.6, "pos_coef": 0.5, "neg_coef": 0.8, "threshold_factor": 2.0},
        0.0008: {"prob_positive": 0.4, "pos_coef": 0.5, "neg_coef": 0.8, "threshold_factor": 2.0},
        0.001: {"prob_positive": 0.4, "pos_coef": 0.5, "neg_coef": 0.8, "threshold_factor": 2.0},
        0.002: {"prob_positive": 0.3, "pos_coef": 0.5, "neg_coef": 0.8, "threshold_factor": 2.0},
    },
}

# Three-phase mode: code -> physical error rate -> {"ends": [b1, b2], "phases": [tuple, tuple, tuple]}.
# Each tuple is (prob_positive, pos_coef, neg_coef, threshold_factor). The tilt computed after `completed` finished
# tilt rounds uses phase 0 while completed < b1, phase 1 while b1 <= completed < b2, phase 2 after. "fitted_at" is
# the rate the schedule was fitted at (two GB102 rates share one fit). These are exactly the schedules of the
# paper's curves (see simulation_results/paper_data, column decoder_params).
THREE_PHASE_SCHEDULES: Dict[str, Dict[float, Dict[str, object]]] = {
    "BB144": {
        0.001: {
            "ends": [100, 800],
            "phases": [[0.2744, 0.5, 0.476, 1.4], [0.1949, 0.5, 0.7683, 1.19], [0.0977, 0.4626, 2.45, 2.5671]],
            "fitted_at": 0.001,
        },
        0.002: {
            "ends": [100, 800],
            "phases": [[0.196, 0.5, 0.56, 1.4], [0.1392, 0.5, 0.5488, 1.4], [0.0977, 0.3304, 2.45, 2.1755]],
            "fitted_at": 0.002,
        },
        0.003: {
            "ends": [120, 1000],
            "phases": [[0.196, 0.5, 0.392, 1.4], [0.118, 0.5, 0.56, 1.4], [0.1396, 0.2, 1.75, 2.1755]],
            "fitted_at": 0.003,
        },
        0.004: {
            "ends": [120, 1000],
            "phases": [[0.1372, 0.59, 0.392, 1.4], [0.118, 0.425, 0.56, 1.4], [0.1153, 0.2, 1.75, 1.8492]],
            "fitted_at": 0.004,
        },
        0.005: {
            "ends": [70, 1000],
            "phases": [[0.196, 0.5, 0.392, 1.4], [0.0826, 0.413, 0.476, 1.652], [0.1153, 0.17, 1.4875, 1.8492]],
            "fitted_at": 0.005,
        },
    },
    "BB70": {
        0.0005: {
            "ends": [80, 860],
            "phases": [[0.5476, 0.59, 0.476, 2.0], [0.5488, 0.2499, 1.029, 2.0], [0.2808, 0.3499, 3.4692, 2.36]],
            "fitted_at": 0.0005,
        },
        0.0008: {
            "ends": [80, 860],
            "phases": [[0.6442, 0.59, 0.476, 2.0], [0.5488, 0.2499, 1.029, 2.0], [0.2006, 0.2499, 2.478, 2.36]],
            "fitted_at": 0.0008,
        },
        0.001: {
            "ends": [80, 860],
            "phases": [[0.5459, 0.59, 0.56, 2.0], [0.5488, 0.2499, 1.029, 2.0], [0.2006, 0.2499, 2.478, 2.36]],
            "fitted_at": 0.001,
        },
        0.002: {
            "ends": [100, 1070],
            "phases": [[0.4626, 0.5, 0.392, 2.0], [0.392, 0.2499, 0.735, 2.0], [0.17, 0.1785, 1.77, 2.0]],
            "fitted_at": 0.002,
        },
    },
    "GB102": {
        0.0005: {
            "ends": [100, 1560],
            "phases": [[0.6997, 0.59, 0.392, 1.0115], [0.588, 0.595, 0.476, 1.7], [0.3009, 0.2832, 1.0413, 1.4]],
            "fitted_at": 0.0008,
        },
        0.0008: {
            "ends": [100, 1560],
            "phases": [[0.6997, 0.59, 0.392, 1.0115], [0.588, 0.595, 0.476, 1.7], [0.3009, 0.2832, 1.0413, 1.4]],
            "fitted_at": 0.0008,
        },
        0.001: {
            "ends": [100, 1560],
            "phases": [[0.4998, 0.59, 0.392, 1.19], [0.588, 0.425, 0.476, 1.7], [0.3009, 0.2832, 1.0413, 1.4]],
            "fitted_at": 0.001,
        },
        0.002: {
            "ends": [200, 1560],
            "phases": [[0.4213, 0.59, 0.2744, 0.98], [0.42, 0.425, 0.476, 2.36], [0.255, 0.2023, 1.4875, 1.4]],
            "fitted_at": 0.002,
        },
    },
}

#: XYZ decoding (the paper's correlated-decoding experiment on BB144 at p = 0.003): the memory circuit whose detectors
#: after the first round compare ALL stabiliser measurements, so the decoder works on the full, non-decomposed
#: detector error model (about 68,000 error mechanisms instead of 8,784). Its own three-phase schedule, fitted on
#: xyz_x pools and validated on both bases, with a 6000-round budget. Use PriorTilt(dem, schedule=xyz_schedule(...),
#: max_rounds=6000). Not part of the tuned_config lookup, which serves the standard memory experiments.
XYZ_THREE_PHASE_SCHEDULES: Dict[str, Dict[float, Dict[str, object]]] = {
    "BB144": {
        0.003: {
            "ends": [50, 420],
            "phases": [[0.07, 0.5, 0.8, 2.0], [0.1372, 0.5, 0.784, 0.686], [0.049, 0.5, 1.568, 1.372]],
            "max_rounds": 6000,
            "fitted_at": 0.003,
        },
    },
}

PARAM_NAMES = ("prob_positive", "pos_coef", "neg_coef", "threshold_factor")


class UntunedCodeError(ValueError):
    """Raised when PriorTilt is asked for tuned parameters of a code it has not been tuned for."""


def _tuning_request(code: object) -> str:
    known = ", ".join(CODES)
    return (
        f"PriorTilt has no optimised parameters for code {code!r}. Its logical error rate depends strongly on the "
        f"tilt parameters, so tune prob_positive, pos_coef, neg_coef and threshold_factor (or fit a three-phase "
        f"schedule) for your code and noise model before using it, then pass them explicitly (prob_positive=..., "
        f"pos_coef=..., neg_coef=..., threshold_factor=..., or schedule=...). Codes with recorded parameters: {known}."
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
    """The fitted rate closest to `p` in the ratio sense: argmin over r of |log(p / r)| (ties -> the lower rate)."""
    if not (p > 0):
        raise ValueError(f"the physical error rate must be positive, got {p}")
    rates = sorted(rates)
    if not rates:
        raise ValueError("no fitted rates")
    return min(rates, key=lambda r: (abs(math.log(p / r)), r))


def fitted_rates(code: Union[str, int], mode: str = "three_phase") -> List[float]:
    """Physical error rates with recorded parameters for `code` in `mode` ("three_phase" or "uniform")."""
    name = normalize_code(code)
    table = UNIFORM_PARAMS if mode == "uniform" else THREE_PHASE_SCHEDULES
    return sorted(table.get(name, {}))


def _match_rate(name: str, p: float, rates: List[float], mode: str) -> float:
    r = closest_rate(p, rates)
    lo, hi = min(rates), max(rates)
    if p < lo * (1 - 1e-9) or p > hi * (1 + 1e-9):
        warnings.warn(
            f"PriorTilt {mode} parameters for {name} were fitted for physical error rates in [{lo:g}, {hi:g}]; "
            f"p={p:g} is outside that range, using the set fitted at {r:g}. The optimum drifts with the noise rate, "
            f"so consider tuning at p={p:g}.",
            stacklevel=4,
        )
    return r


def uniform_params(code: Union[str, int], p: float) -> Tuple[Dict[str, float], float]:
    """(uniform parameter dict, fitted rate used) for `code` at physical error rate `p`."""
    name = normalize_code(code)
    table = UNIFORM_PARAMS[name]
    r = _match_rate(name, p, list(table), "uniform")
    return dict(table[r]), r


def three_phase_schedule(code: Union[str, int], p: float) -> Tuple[Dict[str, object], float]:
    """(schedule dict with keys ends / phases, fitted rate used) for `code` at `p`."""
    name = normalize_code(code)
    table = THREE_PHASE_SCHEDULES[name]
    r = _match_rate(name, p, list(table), "three-phase")
    sch = table[r]
    return {"ends": list(sch["ends"]), "phases": [list(t) for t in sch["phases"]]}, r


@dataclass
class PriorTiltConfig:
    """Resolved decoder configuration (what PriorTilt actually runs with)."""

    mode: str  # "uniform" or "three_phase"
    prob_positive: float
    pos_coef: float
    neg_coef: float
    threshold_factor: float
    schedule: Optional[Dict[str, object]] = None  # {"ends": [...], "phases": [[p_pos, pos, neg, thr], ...]}
    max_rounds: int = DEFAULT_MAX_ROUNDS
    initial_iters: int = DEFAULT_INITIAL_ITERS
    round_len: int = DEFAULT_ROUND_LEN
    code: Optional[str] = None
    p_requested: Optional[float] = None
    p_used: Optional[float] = None
    overrides: Dict[str, float] = field(default_factory=dict)

    def validate(self) -> None:
        if self.mode not in ("uniform", "three_phase"):
            raise ValueError(f"mode must be 'uniform' or 'three_phase', not {self.mode!r}")
        if self.mode == "three_phase":
            if not self.schedule:
                raise ValueError("three_phase mode needs a schedule")
            ends = list(self.schedule["ends"])
            phases = self.schedule["phases"]
            if len(phases) != len(ends) + 1:
                raise ValueError("schedule needs len(ends) + 1 phases")
            if any(e <= 0 for e in ends) or any(b <= a for a, b in zip(ends, ends[1:])):
                raise ValueError("schedule ends must be positive and strictly increasing")
            if any(len(t) != 4 for t in phases):
                raise ValueError("each schedule phase is (prob_positive, pos_coef, neg_coef, threshold_factor)")
        for k in ("max_rounds", "initial_iters", "round_len"):
            if int(getattr(self, k)) < 1:
                raise ValueError(f"{k} must be >= 1")

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
        """JSON-serialisable record of the configuration (stored in the decoder_params column of result files)."""
        d = {"initial_iters": self.initial_iters, "round_len": self.round_len, "max_rounds": self.max_rounds, "num_results": 1}
        if self.mode == "three_phase":
            d["schedule"] = {"ends": list(self.schedule["ends"]), "phases": [list(t) for t in self.schedule["phases"]]}
        else:
            d.update(
                prob_positive=self.prob_positive,
                pos_coef=self.pos_coef,
                neg_coef=self.neg_coef,
                threshold_factor=self.threshold_factor,
            )
        return d

    def describe(self) -> str:
        where = f" for {self.code} at p={self.p_used:g}" if self.code else ""
        if self.p_requested is not None and self.p_used is not None and abs(self.p_requested - self.p_used) > 1e-12:
            where += f" (requested p={self.p_requested:g})"
        if self.mode == "uniform":
            body = (
                f"prob_positive={self.prob_positive:g}, pos_coef={self.pos_coef:g}, neg_coef={self.neg_coef:g}, "
                f"threshold_factor={self.threshold_factor:g}"
            )
        else:
            body = f"schedule ends={self.schedule['ends']}, phases={self.schedule['phases']}"
        return f"PriorTilt {self.mode} mode{where}: {body}; max_rounds={self.max_rounds}"


def tuned_config(
    code: Union[str, int],
    p: float,
    mode: str = "auto",
    *,
    max_rounds: int = DEFAULT_MAX_ROUNDS,
    initial_iters: int = DEFAULT_INITIAL_ITERS,
    round_len: int = DEFAULT_ROUND_LEN,
    **overrides: float,
) -> PriorTiltConfig:
    """The optimised configuration for `code` at physical error rate `p` in `mode` ("three_phase", "uniform", or
    "auto" = three_phase, the paper's decoder).

    Keyword overrides (prob_positive, pos_coef, neg_coef, threshold_factor) replace the looked-up uniform values;
    they are rejected in three-phase mode, where the schedule sets the parameters."""
    name = normalize_code(code)
    bad = set(overrides) - set(PARAM_NAMES)
    if bad:
        raise TypeError(f"unknown parameter(s) {sorted(bad)}; tunable parameters are {PARAM_NAMES}")
    if mode not in ("auto", "uniform", "three_phase"):
        raise ValueError(f"mode must be 'auto', 'uniform' or 'three_phase', not {mode!r}")
    if mode in ("auto", "three_phase"):
        if overrides:
            raise TypeError("explicit tilt parameters do not apply in three_phase mode (the schedule sets them); use mode='uniform'")
        sch, r = three_phase_schedule(name, p)
        first = sch["phases"][0]
        cfg = PriorTiltConfig(
            mode="three_phase",
            prob_positive=float(first[0]),
            pos_coef=float(first[1]),
            neg_coef=float(first[2]),
            threshold_factor=float(first[3]),
            schedule=sch,
            max_rounds=max_rounds,
            initial_iters=initial_iters,
            round_len=round_len,
            code=name,
            p_requested=p,
            p_used=r,
        )
    else:
        prm, r = uniform_params(name, p)
        prm.update({k: float(v) for k, v in overrides.items()})
        cfg = PriorTiltConfig(
            mode="uniform",
            prob_positive=prm["prob_positive"],
            pos_coef=prm["pos_coef"],
            neg_coef=prm["neg_coef"],
            threshold_factor=prm["threshold_factor"],
            schedule=None,
            max_rounds=max_rounds,
            initial_iters=initial_iters,
            round_len=round_len,
            code=name,
            p_requested=p,
            p_used=r,
            overrides=dict(overrides),
        )
    cfg.validate()
    return cfg


def xyz_schedule(code: Union[str, int] = "BB144", p: float = 0.003) -> Dict[str, object]:
    """The three-phase schedule of the paper's XYZ decoding experiment, as a dict {"ends", "phases"} for
    PriorTilt(schedule=...), together with the round budget under "max_rounds" (6000). Only (BB144, 0.003) exists."""
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
