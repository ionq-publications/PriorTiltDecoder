# This file is part of PriorTiltDecoder.

# Copyright (c) 2026 IonQ, Inc., all rights reserved

# Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International License (CC BY-NC-SA 4.0).

# You may obtain a copy of the License at:
# https://creativecommons.org/licenses/by-nc-sa/4.0/

"""PriorTilt: min-sum belief propagation with transient prior tilts, for circuit-level decoding of qLDPC codes.

Single-solution decoder with a Rust core (``decoder/``, Python module ``priortilt_rs``). Two operating modes:

* **three-phase** (the paper's decoder): a schedule of two round boundaries and one parameter tuple per block;
* **uniform**: one parameter tuple for every tilt round.

Typical use, with the optimised parameters of one of the paper's codes::

    import stim
    from priortilt import PriorTilt

    circuit = stim.Circuit.from_file("StimCircuit/BB[[144,12,12]],memory_Z,error_rate=0.003,syndrome_rounds=12.stim")
    dem = circuit.detector_error_model()
    dec = PriorTilt(dem, code="BB144", p=0.003)              # three-phase mode, 3000 rounds (paper setting)
    dets, obs = circuit.compile_detector_sampler().sample(1000, separate_observables=True)
    predictions = dec.decode_batch(dets)                      # bool array, shots x observables
    print((predictions != obs).any(axis=1).sum(), "logical failures")

The physical error rate ``p`` is the two-qubit gate error rate of the circuit; it is matched to the closest fitted
rate. For a code without recorded parameters pass the parameters explicitly (after tuning them)::

    dec = PriorTilt(dem, prob_positive=0.1, pos_coef=0.5, neg_coef=0.8, threshold_factor=2.0)
    dec = PriorTilt(dem, schedule={"ends": [100, 800], "phases": [[...], [...], [...]]})
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Dict, Optional, Union

import numpy as np
import scipy.sparse
import stim

from priortilt import params as _params
from priortilt.dem import DemMatrices, dem_to_matrices
from priortilt.params import (
    CODES,
    DEFAULT_INITIAL_ITERS,
    DEFAULT_MAX_ROUNDS,
    DEFAULT_ROUND_LEN,
    PARAM_NAMES,
    THREE_PHASE_SCHEDULES,
    UNIFORM_PARAMS,
    PriorTiltConfig,
    UntunedCodeError,
    tuned_config,
)

__all__ = [
    "PriorTilt",
    "PriorTiltConfig",
    "UntunedCodeError",
    "CODES",
    "UNIFORM_PARAMS",
    "THREE_PHASE_SCHEDULES",
    "tuned_config",
    "dem_to_matrices",
    "DemMatrices",
]

try:
    from priortilt_rs import PriorTiltDecoder as _RustF64
    from priortilt_rs import PriorTiltDecoderF32 as _RustF32
except ImportError as _e:  # pragma: no cover - depends on the build
    _RustF64 = _RustF32 = None
    _IMPORT_ERROR = _e


def _core_class(precision: str):
    if _RustF64 is None:
        raise ImportError(
            "the Rust extension `priortilt_rs` is not installed. Build it with `pip install ./decoder` from the "
            "repository root (needs a Rust toolchain, https://rustup.rs)."
        ) from _IMPORT_ERROR
    cls = {"f64": _RustF64, "f32": _RustF32}.get(precision)
    if cls is None:
        raise ValueError("precision must be 'f64' or 'f32'")
    return cls


def _create_core(precision: str, pcm, priors, cfg: PriorTiltConfig, seed: int):
    core = _core_class(precision)(
        scipy.sparse.csc_matrix(pcm),
        np.asarray(priors, dtype=np.float64),
        initial_iters=int(cfg.initial_iters),
        round_len=int(cfg.round_len),
        max_rounds=int(cfg.max_rounds),
        threshold_factor=float(cfg.threshold_factor),
        neg_coef=float(cfg.neg_coef),
        pos_coef=float(cfg.pos_coef),
        prob_positive=float(cfg.prob_positive),
        seed=int(seed),
    )
    if cfg.mode == "three_phase":
        ends, ppos, pos, neg, thr = cfg.schedule_arrays()
        core.set_round_schedule(
            np.asarray(ends, dtype=np.int64),
            np.asarray(ppos, dtype=np.float64),
            np.asarray(pos, dtype=np.float64),
            np.asarray(neg, dtype=np.float64),
            np.asarray(thr, dtype=np.float64),
        )
    return core


class PriorTilt:
    """PriorTilt decoder on a stim DetectorErrorModel.

    Parameters
    ----------
    model : stim.DetectorErrorModel
        The detector error model of the circuit to decode (``circuit.detector_error_model()``; the decoder handles
        undecomposed hyperedges, so ``decompose_errors`` is not needed).
    code : str or int, optional
        Code whose optimised parameters to use ("BB144", "BB70", "GB102", any case, or the number of physical
        qubits). Requires ``p``. Other codes raise ``UntunedCodeError``.
    p : float, optional
        Physical (two-qubit gate) error rate of the circuit; the closest fitted rate is used (a warning is emitted
        outside the fitted range).
    mode : {"auto", "three_phase", "uniform"}
        Operating mode. "auto" (default) is three-phase (the paper's decoder) with ``code`` or ``schedule`` and
        uniform with explicit scalar parameters.
    prob_positive, pos_coef, neg_coef, threshold_factor : float, optional
        Explicit uniform-mode tilt parameters. Without ``code`` all four are required; with ``code`` they override
        the looked-up values (uniform mode only).
    schedule : dict, optional
        Explicit three-phase schedule ``{"ends": [b1, b2, ...], "phases": [[prob_positive, pos_coef, neg_coef,
        threshold_factor], ...]}`` with ``len(phases) == len(ends) + 1``. Mutually exclusive with ``code``.
    max_rounds : int
        Tilt-round budget (default 3000, the paper's setting; each round is ``round_len`` BP iterations). A larger
        budget lowers the logical error rate at low noise and barely changes the mean decoding time.
    initial_iters, round_len : int
        Plain BP iterations before the tilt rounds (60) and iterations per tilt round (6). The optimised
        parameters assume these values.
    seed : int
        Seed of the kick-direction generator; negative (default) = nondeterministic.
    precision : {"f64", "f32"}
        Message precision ("f32" is faster and statistically equivalent; the paper used "f64").
    """

    def __init__(
        self,
        model: stim.DetectorErrorModel,
        *,
        code: Optional[Union[str, int]] = None,
        p: Optional[float] = None,
        mode: str = "auto",
        prob_positive: Optional[float] = None,
        pos_coef: Optional[float] = None,
        neg_coef: Optional[float] = None,
        threshold_factor: Optional[float] = None,
        schedule: Optional[Dict[str, object]] = None,
        max_rounds: int = DEFAULT_MAX_ROUNDS,
        initial_iters: int = DEFAULT_INITIAL_ITERS,
        round_len: int = DEFAULT_ROUND_LEN,
        seed: int = -1,
        precision: str = "f64",
    ):
        explicit = {
            k: v
            for k, v in dict(
                prob_positive=prob_positive, pos_coef=pos_coef, neg_coef=neg_coef, threshold_factor=threshold_factor
            ).items()
            if v is not None
        }
        self.config = self._resolve_config(
            code, p, mode, explicit, schedule, max_rounds=max_rounds, initial_iters=initial_iters, round_len=round_len
        )
        self._matrices = dem_to_matrices(model)
        self.num_detectors = model.num_detectors
        self.num_observables = model.num_observables
        self.num_errors = int(self._matrices.check_matrix.shape[1])
        self._seed = int(seed)
        self._precision = precision
        self._core = _create_core(precision, self._matrices.check_matrix, self._matrices.priors, self.config, self._seed)

    # ------------------------------------------------------------------ configuration

    @staticmethod
    def _resolve_config(code, p, mode, explicit, schedule, *, max_rounds, initial_iters, round_len) -> PriorTiltConfig:
        if schedule is not None:
            if code is not None or explicit:
                raise TypeError("schedule= is an explicit configuration; do not combine it with code= or scalar parameters")
            if mode not in ("auto", "three_phase"):
                raise ValueError("an explicit schedule implies mode='three_phase'")
            first = list(schedule["phases"][0])
            cfg = PriorTiltConfig(
                mode="three_phase",
                prob_positive=float(first[0]),
                pos_coef=float(first[1]),
                neg_coef=float(first[2]),
                threshold_factor=float(first[3]),
                schedule={
                    "ends": [int(e) for e in schedule["ends"]],
                    "phases": [list(map(float, t)) for t in schedule["phases"]],
                },
                max_rounds=max_rounds,
                initial_iters=initial_iters,
                round_len=round_len,
            )
            cfg.validate()
            return cfg
        if code is not None:
            if p is None:
                raise TypeError("p (physical error rate) is required together with code")
            return tuned_config(code, p, mode, max_rounds=max_rounds, initial_iters=initial_iters, round_len=round_len, **explicit)
        missing = [k for k in PARAM_NAMES if k not in explicit]
        if missing:
            raise UntunedCodeError(
                "PriorTilt needs either code= and p= (optimised parameters of one of the paper's codes) or all four "
                f"tilt parameters {PARAM_NAMES} (missing {missing}) or a schedule=. Its logical error rate depends "
                "strongly on these parameters: tune them for your code and noise model before use. Codes with "
                f"recorded parameters: {', '.join(CODES)}."
            )
        if mode not in ("auto", "uniform"):
            raise ValueError("explicit scalar parameters imply mode='uniform'")
        cfg = PriorTiltConfig(
            mode="uniform",
            prob_positive=float(explicit["prob_positive"]),
            pos_coef=float(explicit["pos_coef"]),
            neg_coef=float(explicit["neg_coef"]),
            threshold_factor=float(explicit["threshold_factor"]),
            max_rounds=max_rounds,
            initial_iters=initial_iters,
            round_len=round_len,
        )
        cfg.validate()
        return cfg

    @property
    def mode(self) -> str:
        return self.config.mode

    @property
    def precision(self) -> str:
        return self._precision

    @property
    def seed(self) -> int:
        return self._seed

    @seed.setter
    def seed(self, value: int) -> None:
        self._seed = int(value)
        self._core.seed = self._seed

    def __repr__(self) -> str:
        return f"PriorTilt({self.config.describe()}; precision={self._precision})"

    # ------------------------------------------------------------------ pickling (sinter workers, deepcopy)

    def __getstate__(self) -> dict:
        return {
            "check_matrix": self._matrices.check_matrix,
            "observables_matrix": self._matrices.observables_matrix,
            "priors": self._matrices.priors,
            "num_detectors": self.num_detectors,
            "num_observables": self.num_observables,
            "num_errors": self.num_errors,
            "config": self.config,
            "seed": self._seed,
            "precision": self._precision,
        }

    def __setstate__(self, state: dict) -> None:
        self._matrices = SimpleNamespace(
            check_matrix=state["check_matrix"], observables_matrix=state["observables_matrix"], priors=state["priors"]
        )
        self.num_detectors = state["num_detectors"]
        self.num_observables = state["num_observables"]
        self.num_errors = state["num_errors"]
        self.config = state["config"]
        self._seed = state["seed"]
        self._precision = state["precision"]
        self._core = _create_core(self._precision, self._matrices.check_matrix, self._matrices.priors, self.config, self._seed)

    # ------------------------------------------------------------------ decoding

    @property
    def converged(self) -> bool:
        """Whether the last single-shot decode reached a syndrome-consistent solution."""
        return bool(self._core.converge)

    @property
    def iterations(self) -> int:
        """BP iterations of the last single-shot decode."""
        return int(self._core.iterations)

    @property
    def tilt_rounds(self) -> int:
        """Tilt rounds completed in the last single-shot decode."""
        return int(self._core.tilt_rounds)

    def tilt_statistics(self) -> dict:
        """Diagnostic counters accumulated over all decodes since the last reset_tilt_statistics():
        `nonzero_offsets` = number of (bit, tilt round) pairs that received a nonzero prior offset,
        `bit_rounds` = number of (bit, tilt round) pairs examined (= num_errors per tilt round),
        `tilt_rounds` = number of tilt rounds. nonzero_offsets / bit_rounds is the fraction of bits with a
        nonzero offset per tilt round (the quantity tabulated in the paper)."""
        u, b, t = self._core.tilt_statistics()
        return {"nonzero_offsets": int(u), "bit_rounds": int(b), "tilt_rounds": int(t)}

    def reset_tilt_statistics(self) -> None:
        self._core.reset_tilt_statistics()

    def decode_errors(self, syndrome: np.ndarray) -> np.ndarray:
        """Error-mechanism estimate (uint8, length num_errors) for one shot of detector data."""
        syn = np.ascontiguousarray(syndrome, dtype=np.uint8)
        if syn.ndim != 1 or syn.shape[0] != self.num_detectors:
            raise ValueError(f"syndrome must have shape ({self.num_detectors},), got {syn.shape}")
        return np.asarray(self._core.decode(syn), dtype=np.uint8)

    def decode(self, syndrome: np.ndarray) -> np.ndarray:
        """Decode one shot of detector data; returns the predicted observable flips (uint8, length num_observables)."""
        corr = self.decode_errors(syndrome)
        return (self._matrices.observables_matrix @ corr) % 2

    def decode_batch(
        self, shots: np.ndarray, *, bit_packed_shots: bool = False, bit_packed_predictions: bool = False
    ) -> np.ndarray:
        """Decode a batch of shots (``shots x num_detectors``); returns a bool array ``shots x num_observables``
        (bit packed along axis 1 when ``bit_packed_predictions``)."""
        if bit_packed_shots:
            shots = np.unpackbits(shots, axis=1, bitorder="little")[:, : self.num_detectors]
        shots = np.ascontiguousarray(shots, dtype=np.uint8)
        if shots.ndim != 2 or shots.shape[1] != self.num_detectors:
            raise ValueError(f"shots must have shape (n, {self.num_detectors}), got {shots.shape}")
        corr = self._core.decode_batch(shots)
        predictions = ((self._matrices.observables_matrix @ corr.T) % 2).T.astype(bool)
        if bit_packed_predictions:
            predictions = np.packbits(predictions, axis=1, bitorder="little")
        return predictions


params = _params
