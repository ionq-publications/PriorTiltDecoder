# This file is part of PriorTiltDecoder.

# Copyright (c) 2026 IonQ, Inc., all rights reserved

# Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International License (CC BY-NC-SA 4.0).

# You may obtain a copy of the License at:
# https://creativecommons.org/licenses/by-nc-sa/4.0/

"""sinter wrapper for the PriorTilt decoder.

The optimised parameters depend on the code and on the physical error rate, and sinter hands the decoder only the
detector error model, so one wrapper is configured per (code, p)::

    import sinter
    from priortilt.sinter_priortilt import SinterDecoder_PriorTilt

    decoders = {f"PriorTilt_BB144_p{p:g}": SinterDecoder_PriorTilt(code="BB144", p=p) for p in rates}
    tasks = [sinter.Task(circuit=circ, decoder=f"PriorTilt_BB144_p{p:g}", json_metadata={"p": p}) for p, circ in ...]
    stats = sinter.collect(tasks=tasks, custom_decoders=decoders, num_workers=8, max_errors=100, max_shots=10**7)

Every worker builds its own decoder in ``compile_decoder_for_dem`` with a fresh nondeterministic seed unless a
``seed`` >= 0 is given.
"""

from __future__ import annotations

import pathlib
from typing import Dict, Optional, Union

import numpy as np
import stim
from sinter import CompiledDecoder, Decoder

from priortilt import PriorTilt


class SinterCompiledDecoder_PriorTilt(CompiledDecoder):
    def __init__(self, decoder: PriorTilt):
        self.decoder = decoder

    def decode_shots_bit_packed(self, *, bit_packed_detection_event_data: np.ndarray) -> np.ndarray:
        return self.decoder.decode_batch(
            shots=bit_packed_detection_event_data, bit_packed_shots=True, bit_packed_predictions=True
        )


class SinterDecoder_PriorTilt(Decoder):
    def __init__(
        self,
        *,
        code: Optional[Union[str, int]] = None,
        p: Optional[float] = None,
        mode: str = "auto",
        prob_positive: Optional[float] = None,
        pos_coef: Optional[float] = None,
        neg_coef: Optional[float] = None,
        threshold_factor: Optional[float] = None,
        schedule: Optional[Dict[str, object]] = None,
        max_rounds: int = 3000,
        initial_iters: int = 60,
        round_len: int = 6,
        seed: int = -1,
        precision: str = "f64",
    ):
        """Same parameters as ``PriorTilt`` (without the model). The configuration is resolved here, so an unknown
        code or a missing parameter fails at construction rather than inside the sinter workers."""
        self.kwargs = dict(
            code=code,
            p=p,
            mode=mode,
            prob_positive=prob_positive,
            pos_coef=pos_coef,
            neg_coef=neg_coef,
            threshold_factor=threshold_factor,
            schedule=schedule,
            max_rounds=max_rounds,
            initial_iters=initial_iters,
            round_len=round_len,
            seed=seed,
            precision=precision,
        )
        explicit = {
            k: v
            for k, v in dict(
                prob_positive=prob_positive, pos_coef=pos_coef, neg_coef=neg_coef, threshold_factor=threshold_factor
            ).items()
            if v is not None
        }
        self.config = PriorTilt._resolve_config(
            code, p, mode, explicit, schedule, max_rounds=max_rounds, initial_iters=initial_iters, round_len=round_len
        )

    def _make(self, dem: stim.DetectorErrorModel) -> PriorTilt:
        return PriorTilt(dem, **self.kwargs)

    def compile_decoder_for_dem(self, *, dem: stim.DetectorErrorModel) -> CompiledDecoder:
        return SinterCompiledDecoder_PriorTilt(self._make(dem))

    def decode_via_files(
        self,
        *,
        num_shots: int,
        num_dets: int,
        num_obs: int,
        dem_path: pathlib.Path,
        dets_b8_in_path: pathlib.Path,
        obs_predictions_b8_out_path: pathlib.Path,
        tmp_dir: pathlib.Path,
    ) -> None:
        dem = stim.DetectorErrorModel.from_file(dem_path)
        decoder = self._make(dem)
        shots = stim.read_shot_data_file(path=dets_b8_in_path, format="b8", num_detectors=dem.num_detectors, bit_packed=False)
        predictions = decoder.decode_batch(shots)
        stim.write_shot_data_file(data=predictions, path=obs_predictions_b8_out_path, format="b8", num_observables=dem.num_observables)
