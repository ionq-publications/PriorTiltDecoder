# This file is part of PriorTiltDecoder.

# Copyright (c) 2026 IonQ, Inc., all rights reserved

# Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International License (CC BY-NC-SA 4.0).

# You may obtain a copy of the License at:
# https://creativecommons.org/licenses/by-nc-sa/4.0/

"""Conversion of a stim DetectorErrorModel into the matrices the decoder works on: the parity-check matrix H
(detectors x error mechanisms), the logical operator matrix A (logical observables x error mechanisms) and the
error probabilities p_j of the mechanisms, in the paper's notation.

Every ``error`` instruction of the (flattened) model is one error mechanism. Mechanisms with the same set of
detectors are merged into one column: their probabilities combine as independent events
(p = p1 (1 - p2) + p2 (1 - p1)) and the observables of the last merged instruction are kept. This is the same
convention as the ``stimbposd`` package, so the decoder sees identical matrices and priors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, FrozenSet, List

import numpy as np
import stim
from scipy.sparse import csc_matrix


@dataclass
class DemMatrices:
    check_matrix: csc_matrix  # H: detectors x error mechanisms (uint8)
    observables_matrix: csc_matrix  # A: logical observables x error mechanisms (uint8)
    priors: np.ndarray  # p_j: probability of each error mechanism (float64)


def _xor_of_groups(groups: List[List[int]]) -> FrozenSet[int]:
    out: set = set()
    for g in groups:
        out ^= set(g)
    return frozenset(out)


def _to_csc(columns: Dict[int, FrozenSet[int]], shape) -> csc_matrix:
    rows, cols = [], []
    for c, rs in columns.items():
        for r in rs:
            rows.append(r)
            cols.append(c)
    data = np.ones(len(rows), dtype=np.uint8)
    return csc_matrix((data, (np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64))), shape=shape)


def dem_to_matrices(dem: stim.DetectorErrorModel) -> DemMatrices:
    """Check matrix, observables matrix and priors of a detector error model (hyperedges allowed)."""
    column_of: Dict[FrozenSet[int], int] = {}
    obs_of: Dict[int, FrozenSet[int]] = {}
    prior_of: Dict[int, float] = {}
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        prob = inst.args_copy()[0]
        dets: List[List[int]] = [[]]
        obs: List[List[int]] = [[]]
        for t in inst.targets_copy():
            if t.is_relative_detector_id():
                dets[-1].append(t.val)
            elif t.is_logical_observable_id():
                obs[-1].append(t.val)
            elif t.is_separator():
                dets.append([])
                obs.append([])
        key = _xor_of_groups(dets)
        if key not in column_of:
            column_of[key] = len(column_of)
            prior_of[column_of[key]] = 0.0
        c = column_of[key]
        obs_of[c] = _xor_of_groups(obs)
        prior_of[c] = prior_of[c] * (1 - prob) + prob * (1 - prior_of[c])
    n = len(column_of)
    check = _to_csc({c: k for k, c in column_of.items()}, (dem.num_detectors, n))
    observables = _to_csc(obs_of, (dem.num_observables, n))
    priors = np.array([prior_of[c] for c in range(n)], dtype=np.float64)
    return DemMatrices(check_matrix=check, observables_matrix=observables, priors=priors)
