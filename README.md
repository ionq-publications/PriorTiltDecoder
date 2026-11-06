# PriorTilt Decoder for Quantum LDPC Codes

This repository contains the data and the source code required to reproduce the simulation results presented in the paper
**PriorTilt: Decoding quantum LDPC codes with sparse perturbations of belief propagation priors**
by Min Ye and Nicolas Delfosse (IonQ, 2026; arXiv number to be added).

Two levels of reproduction are provided:

1. **From the stored data.** `simulation_results/paper_data/` holds the raw simulation results (shots, failures, wall time,
   decoder parameters) behind every point of the paper's main figure, for all three decoders: PriorTilt (three-phase mode),
   Relay-BP and beam search. The notebook plots the figure from these files.
2. **From scratch, for PriorTilt.** The Rust implementation of the PriorTilt decoder (three-phase mode) with its optimised
   parameters for every (code, noise rate) of the paper, and the 28 noisy Stim circuits that were decoded, so that the
   PriorTilt points can be re-simulated. Relay-BP and beam search are not included; their implementations are available
   from [trmue/relay](https://github.com/trmue/relay) and
   [ionq-publications/BeamSearchDecoder](https://github.com/ionq-publications/BeamSearchDecoder).

## Reproduction Instructions

1. **Install dependencies** (Python 3.10 or newer, and a Rust toolchain from [rustup.rs](https://rustup.rs) to build the decoder):
   ```bash
   pip install -r requirements.txt
   ```

2. **Build the decoder extension.** From the repository root:
   ```bash
   pip install ./decoder
   ```
   This compiles the Rust crate in `decoder/` with maturin and installs the Python module `priortilt_rs` (about 15 s).

3. **Run the notebook.** Open `PriorTilt.ipynb` and execute the cells following the instructions inside. It plots the paper's
   figure from the stored data, re-simulates one point with PriorTilt in about a minute, and documents the full
   (very long) simulation.

## File Overview

* `StimCircuit/`: the 28 noisy Stim circuits of the paper (one file per code, noise rate and memory basis, plus the two XYZ
  circuits) and a
  `manifest.json` with their sizes and checksums. File names follow
  `<code>,memory_<X|Z>,error_rate=<p>,syndrome_rounds=<r>.stim`.
  The two `xyz_memory_*` files are the XYZ-decoding circuits of BB144 at `p_CNOT` = 0.003 (detectors on all stabilisers after
  the first round, so the decoder sees the full detector error model).
* `simulation_results/paper_data/`: the simulation results behind the paper's figure.
  * `<code>_<decoder>.csv`: the raw rows in sinter's CSV format (`shots, errors, discards, seconds, decoder, json_metadata,
    decoder_params`), one file per code and decoder; `decoder_params` records the exact decoder configuration of each row.
  * `paper_figure_data.csv`: the same data as one tidy table with the logical error rate per cycle and its 1-sigma error.
  * `offset_fraction.csv`: the paper's measurement of the fraction of bits that receive a nonzero prior offset per tilt
    round (three-phase mode, per code, noise rate and basis, with shot counts and per-shot statistics).
  * `BB144_XYZ_p0.003.csv`: the XYZ-decoding results of BB144 at `p_CNOT` = 0.003 (PriorTilt three-phase with 6000 rounds,
    Relay-BP with 5 solutions), sinter format.
* `simulation_results/paper_figure.pdf`: the figure as produced by the notebook from the stored data.
* `simulation_results/`: where your own `full_simulation` runs are written.
* `decoder/`: the Rust implementation of PriorTilt (`decoder/src/lib.rs`) with its Python bindings (pyo3 / maturin).
* `priortilt/`: Python package: the `PriorTilt` decoder class (`priortilt/__init__.py`), the optimised parameters and
  three-phase schedules of the paper's codes (`priortilt/params.py`), the detector-error-model conversion
  (`priortilt/dem.py`) and a sinter wrapper (`priortilt/sinter_priortilt.py`).
* `simulation_functions.py`: the functions used in `PriorTilt.ipynb` (plotting, `one_point_simulation`, `full_simulation`,
  `decoding_time`, `offset_fraction` / `offset_fraction_table`, `print_xyz_table` / `xyz_one_point_simulation`).
* `PriorTilt.ipynb`: notebook to reproduce the paper's results.

## The benchmark

| code | memory experiment | noise model | noise rates `p_CNOT` |
|---|---|---|---|
| BB [[144,12,12]] | 12 syndrome extraction rounds | uniform circuit-level depolarising noise | 0.001, 0.002, 0.003, 0.004, 0.005 |
| BB [[70,6,9]] | 9 rounds | trapped-ion circuit-level noise model (`FlyingIonicModel`) | 0.0005, 0.0008, 0.001, 0.002 |
| GB [[102,22,9]] | 9 rounds | trapped-ion circuit-level noise model (`FlyingIonicModel`) | 0.0005, 0.0008, 0.001, 0.002 |

The figure of merit is the logical error rate per syndrome extraction cycle, `(failures / shots) / syndrome_rounds`, for the X
and the Z memory experiment (the paper's curves show their sum). The decoders are run as in the paper:

* **PriorTilt**, three-phase mode: one solution, 60 plain min-sum BP iterations, then tilt rounds of 6 iterations with at most
  3000 rounds, and the three-phase schedule fitted for each (code, noise rate). The schedules are listed in
  `priortilt/params.py` (`THREE_PHASE_SCHEDULES`) and recorded in the `decoder_params` column of the data files.
* **Relay-BP**: 301 legs, one solution, gamma parameters tuned per code (BB144: gamma0 0.125, interval [-0.24, 0.66];
  BB70: 0.125, [-0.24, 0.5]; GB102: 0.03, [-0.3, 0.45]).
* **Beam search**: `beam32_340iters` (beam width 32, 40 initial iterations, 30 per round, 10 rounds).

The paper also reports how sparse the perturbation is: the fraction of bits that receive a nonzero prior offset per tilt
round, 0.15% to 0.4% on these codes (10 to 30 bits out of 3,850 to 8,784). The decoder exposes the counters behind this
number (`PriorTilt.tilt_statistics()`), `offset_fraction_table()` in the notebook re-measures the paper's table, and the
paper's rows are stored in `simulation_results/paper_data/offset_fraction.csv`.

**XYZ decoding.** For BB144 at `p_CNOT` = 0.003 the paper also decodes the XYZ circuits, whose detectors after the first round
compare all stabiliser measurements so that the decoder works on the full detector error model (about 68,000 error
mechanisms). PriorTilt uses a dedicated three-phase schedule (`priortilt.params.xyz_schedule()`), one solution and 6000
rounds, and reaches an x + z logical error rate per cycle of 1.04e-5 (5.32e-6 for X, 5.07e-6 for Z memory), against 1.09e-5
for Relay-BP with five solutions (601 legs). The circuits, the schedule and the
stored rows are in the repository; re-simulating this point costs hundreds of CPU-hours per basis (0.6 s per shot on a laptop
core), so the
notebook only prints the stored table and leaves the simulation cell commented out.

## Using the decoder on your own circuits

```python
import stim
from priortilt import PriorTilt

circuit = stim.Circuit.from_file("StimCircuit/BB[[144,12,12]],memory_Z,error_rate=0.003,syndrome_rounds=12.stim")
decoder = PriorTilt(circuit.detector_error_model(), code="BB144", p=0.003)   # three-phase mode, paper parameters
dets, obs = circuit.compile_detector_sampler().sample(10_000, separate_observables=True)
predictions = decoder.decode_batch(dets)                                       # bool array, shots x observables
print((predictions != obs).any(axis=1).sum(), "logical failures")
```

PriorTilt's logical error rate depends strongly on its parameters, and the optimum depends on the code and on the noise rate.
The parameters shipped here are for the three codes and the circuits above; for another code, circuit or noise model the
parameters have to be tuned first and passed explicitly (`PriorTilt(dem, schedule=...)` or
`PriorTilt(dem, prob_positive=..., pos_coef=..., neg_coef=..., threshold_factor=...)`). Asking for an unknown code raises an
error on purpose.

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{ye2026priortilt,
      title={PriorTilt: Decoding quantum LDPC codes with sparse perturbations of belief propagation priors},
      author={Ye, Min and Delfosse, Nicolas},
      year={2026},
      journal={arXiv preprint}
}
```

## License
Copyright (c) 2026 IonQ, Inc., all rights reserved

This work is licensed under a [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-nc-sa/4.0/).

[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
