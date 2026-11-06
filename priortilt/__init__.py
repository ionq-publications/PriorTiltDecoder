# This file is part of PriorTiltDecoder.

# Copyright (c) 2026 IonQ, Inc., all rights reserved

# Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International License (CC BY-NC-SA 4.0).

# You may obtain a copy of the License at:
# https://creativecommons.org/licenses/by-nc-sa/4.0/

"""PriorTilt: min-sum belief propagation with sparse perturbations of the prior log-likelihood ratios, for
circuit-level decoding of quantum LDPC codes.

The decoder runs up to 60 ordinary min-sum BP iterations and then continues, without resetting its messages,
with modified BP in rounds of six iterations, for at most ``max_rounds`` rounds. At every round boundary the
error nodes that are *error-prone* (posterior LLR sum over the round below ``threshold_factor`` times the prior
LLR) receive a randomized prior offset for the next round; all other nodes keep their original prior. The first
syndrome-consistent hard decision is returned. Rust core in ``decoder/`` (Python module ``priortilt_rs``).

All results of the paper use the three-phase mode: the offset parameters change at two phase boundaries, one
(prob_positive, pos_coef, neg_coef, threshold_factor) tuple per phase (a ``schedule``). The paper's uniform mode
is the special case of a schedule with a single phase and no boundaries.

Parameter names versus the paper's notation: ``threshold_factor`` = theta, ``prob_positive`` = pi,
``pos_coef`` = c_1, ``neg_coef`` = c_2; the code orders a tuple as (pi, c_1, c_2, theta), the paper as
(theta, pi, c_1, c_2). The phase boundaries (b_1, b_2) are the ``ends`` of a schedule.

Typical use, with the optimized parameters of one of the paper's codes::

    import stim
    from priortilt import PriorTilt

    circuit = stim.Circuit.from_file("StimCircuit/BB[[144,12,12]],memory_Z,error_rate=0.003,syndrome_rounds=12.stim")
    dem = circuit.detector_error_model()
    dec = PriorTilt(dem, code="BB144", p=0.003)              # three-phase mode, 3000 rounds (paper setting)
    dets, obs = circuit.compile_detector_sampler().sample(1000, separate_observables=True)
    predictions = dec.decode_batch(dets)                      # bool array, shots x observables
    print((predictions != obs).any(axis=1).sum(), "logical failures")

The physical error rate ``p`` is matched to the closest rate with optimized parameters. For a code without
recorded parameters pass a schedule explicitly (after optimizing it)::

    dec = PriorTilt(dem, schedule={"ends": [100, 800], "phases": [[...], [...], [...]]})   # three-phase mode
    dec = PriorTilt(dem, schedule={"ends": [], "phases": [[0.1, 0.5, 0.8, 2.0]]})         # uniform mode
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
    THREE_PHASE_SCHEDULES,
    PriorTiltConfig,
    UntunedCodeError,
    tuned_config,
)

__all__ = [
    "PriorTilt",
    "PriorTiltConfig",
    "UntunedCodeError",
    "CODES",
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
    ends, ppos, pos, neg, thr = cfg.schedule_arrays()
    core = _core_class(precision)(
        scipy.sparse.csc_matrix(pcm),
        np.asarray(priors, dtype=np.float64),
        initial_iters=int(cfg.initial_iters),
        round_len=int(cfg.round_len),
        max_rounds=int(cfg.max_rounds),
        seed=int(seed),
    )
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
        Code whose optimized parameters to use ("BB144", "BB70", "GB102", any case, or the number of physical
        qubits). Requires ``p``. Other codes raise ``UntunedCodeError``.
    p : float, optional
        Physical error rate of the circuit; the closest rate with optimized parameters is used (a warning is
        emitted outside the optimized range).
    schedule : dict, optional
        Explicit parameter schedule ``{"ends": [b1, b2, ...], "phases": [[prob_positive, pos_coef, neg_coef,
        threshold_factor], ...]}`` with ``len(phases) == len(ends) + 1``: phase k applies to rounds
        ``ends[k-1] < tau <= ends[k]`` (two boundaries and three tuples = the paper's three-phase mode; no boundary
        and one tuple = the uniform mode). Mutually exclusive with ``code``.
    max_rounds : int
        Maximum number R of modified-BP rounds (default 3000, the paper's setting for standard XZ decoding; the
        total budget is ``initial_iters + round_len * max_rounds`` BP iterations). A larger budget lowers the
        logical error rate at low physical error rates and barely changes the mean decoding time.
    initial_iters, round_len : int
        Ordinary BP iterations before the modified-BP rounds (I_0 = 60) and iterations per round (r = 6). The
        optimized parameters assume these values.
    seed : int
        Seed of the random sign choice of the prior offsets; negative (default) = nondeterministic.
    precision : {"f64", "f32"}
        Message precision ("f32" is faster and statistically equivalent; the paper used "f64").
    """

    def __init__(
        self,
        model: stim.DetectorErrorModel,
        *,
        code: Optional[Union[str, int]] = None,
        p: Optional[float] = None,
        schedule: Optional[Dict[str, object]] = None,
        max_rounds: int = DEFAULT_MAX_ROUNDS,
        initial_iters: int = DEFAULT_INITIAL_ITERS,
        round_len: int = DEFAULT_ROUND_LEN,
        seed: int = -1,
        precision: str = "f64",
    ):
        self.config = self._resolve_config(code, p, schedule, max_rounds=max_rounds, initial_iters=initial_iters, round_len=round_len)
        self._matrices = dem_to_matrices(model)
        self.num_detectors = model.num_detectors
        self.num_observables = model.num_observables
        self.num_errors = int(self._matrices.check_matrix.shape[1])
        self._seed = int(seed)
        self._precision = precision
        self._core = _create_core(precision, self._matrices.check_matrix, self._matrices.priors, self.config, self._seed)

    # ------------------------------------------------------------------ configuration

    @staticmethod
    def _resolve_config(code, p, schedule, *, max_rounds, initial_iters, round_len) -> PriorTiltConfig:
        if schedule is not None:
            if code is not None:
                raise TypeError("schedule= is an explicit configuration; do not combine it with code=")
            cfg = PriorTiltConfig(
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
        if code is None:
            raise UntunedCodeError(
                "PriorTilt needs either code= and p= (the paper's optimized parameters of one of its codes) or an explicit "
                "schedule=. Its logical error rate depends strongly on the offset parameters: optimize them for your code "
                f"and noise model before use. Codes with recorded parameters: {', '.join(CODES)}."
            )
        if p is None:
            raise TypeError("p (physical error rate) is required together with code")
        return tuned_config(code, p, max_rounds=max_rounds, initial_iters=initial_iters, round_len=round_len)

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
    def rounds(self) -> int:
        """Modified-BP rounds completed in the last single-shot decode."""
        return int(self._core.rounds)

    def offset_statistics(self) -> dict:
        """Diagnostic counters accumulated over all decodes since the last reset_offset_statistics():
        `error_prone_nodes` = number of (error node, round) pairs that satisfied the error-prone condition
        S_j < theta * Lambda_j (only these nodes receive a nonzero prior offset),
        `node_rounds` = number of (error node, round) pairs examined (= num_errors per modified-BP round),
        `rounds` = number of modified-BP rounds. error_prone_nodes / node_rounds is the mean fraction of
        error-prone nodes per round, the quantity tabulated in the paper; it bounds the fraction of nonzero prior
        offsets from above."""
        u, b, t = self._core.offset_statistics()
        return {"error_prone_nodes": int(u), "node_rounds": int(b), "rounds": int(t)}

    def reset_offset_statistics(self) -> None:
        self._core.reset_offset_statistics()

    def decode_errors(self, syndrome: np.ndarray) -> np.ndarray:
        """Estimated error vector (uint8, length num_errors, one entry per error mechanism) for one shot of
        detector data."""
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
