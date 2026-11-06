# PriorTilt Decoder for Quantum LDPC Codes

This repository contains the data and the source code required to reproduce the simulation results presented in the paper
**PriorTilt: Decoding quantum LDPC codes with sparse perturbations of belief propagation priors**
by Min Ye and Nicolas Delfosse (IonQ, 2026; arXiv number to be added).

PriorTilt is a variant of min-sum belief propagation (BP) that perturbs a small fraction of the prior log-likelihood ratios
(LLRs) while leaving the message-passing rules unchanged. After up to 60 ordinary BP iterations the decoder continues, without
resetting its messages, with modified BP in rounds of six iterations. At every round boundary the *error-prone* nodes, those
whose posterior LLR sum over the round falls below a threshold, receive a randomized prior offset determined by their recent
posterior and outgoing-message statistics; all other nodes keep their original prior. In the three-phase mode used for all
results of the paper, the offset parameters change at two phase boundaries.

Two levels of reproduction are provided:

1. **From the stored data.** `simulation_results/paper_data/` holds the raw simulation results (shots, logical errors, wall
   time, decoder parameters) behind every point of the paper's figure, for all three decoders: PriorTilt (three-phase mode),
   Relay-BP and the beam search decoder. The notebook plots the figure from these files.
2. **From scratch, for PriorTilt.** The Rust implementation of the PriorTilt decoder with the optimized parameters of every
   (code, physical error rate) of the paper, and the 28 noisy Stim circuits that were decoded, so that the PriorTilt points
   can be re-simulated. Relay-BP and the beam search decoder are not included; their implementations are available from
   [trmue/relay](https://github.com/trmue/relay) and
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

* `StimCircuit/`: the 28 noisy Stim circuits of the paper (one file per code, physical error rate and memory basis, plus the
  two XYZ-decoding circuits) and a `manifest.json` with their sizes and checksums. File names follow
  `<code>,memory_<X|Z>,error_rate=<p>,syndrome_rounds=<cycles>.stim`, where `error_rate` is the physical error rate p and
  `syndrome_rounds` the number of syndrome extraction cycles. The two `xyz_memory_*` files are the XYZ-decoding circuits of
  BB144 at p = 0.003: their detectors after the first cycle compare all stabilizer measurements, so the decoder uses both
  detector types jointly on the full detector error model.
* `simulation_results/paper_data/`: the simulation results behind the paper's figure and tables.
  * `<code>_<decoder>.csv`: the raw rows in sinter's CSV format (`shots, errors, discards, seconds, decoder, json_metadata,
    decoder_params`), one file per code and decoder. `decoder_params` records the exact decoder configuration of each row;
    `json_metadata` records the code, the physical error rate (`p_CNOT`), the memory basis (`x_or_z`) and the number of
    syndrome extraction cycles (`syndrome_rounds`).
  * `paper_figure_data.csv`: the same data as one tidy table with the logical error rate per cycle and its 1-sigma error.
  * `offset_fraction.csv`: the paper's measurement of the mean percentage of error-prone nodes per modified-BP round
    (three-phase mode, per code, physical error rate and memory basis, with shot counts and per-shot statistics).
  * `BB144_XYZ_p0.003.csv`: the XYZ-decoding results of BB144 at p = 0.003 (PriorTilt in the three-phase mode with at most
    6000 rounds, and Relay-BP-5), sinter format.
* `simulation_results/paper_figure.pdf`: the figure as produced by the notebook from the stored data.
* `simulation_results/`: where your own `full_simulation` runs are written.
* `decoder/`: the Rust implementation of PriorTilt (`decoder/src/lib.rs`) with its Python bindings (pyo3 / maturin).
* `priortilt/`: Python package: the `PriorTilt` decoder class (`priortilt/__init__.py`), the optimized parameters of the
  paper's codes (`priortilt/params.py`), the detector-error-model conversion (`priortilt/dem.py`) and a sinter wrapper
  (`priortilt/sinter_priortilt.py`).
* `simulation_functions.py`: the functions used in `PriorTilt.ipynb` (plotting, `one_point_simulation`, `full_simulation`,
  `decoding_time`, `offset_fraction` / `offset_fraction_table`, `print_xyz_table` / `xyz_one_point_simulation`).
* `PriorTilt.ipynb`: notebook to reproduce the paper's results.

## The benchmark

| code | memory experiment | circuit-level noise model | physical error rates p |
|---|---|---|---|
| BB144 [[144,12,12]] (bivariate bicycle, gross code) | 12 syndrome extraction cycles | uniform noise model of Bravyi et al. | 0.001, 0.002, 0.003, 0.004, 0.005 |
| BB70 [[70,6,9]] (bivariate bicycle) | 9 cycles | moving-qubit noise model of Tripier et al. (qubit loss and leakage ignored) | 0.0005, 0.0008, 0.001, 0.002 |
| GB102 [[102,22,9]] (generalized bicycle) | 9 cycles | moving-qubit noise model of Tripier et al. (qubit loss and leakage ignored) | 0.0005, 0.0008, 0.001, 0.002 |

Each memory experiment uses d syndrome extraction cycles, d being the code distance. The figure of merit is the logical error
rate (LER) per syndrome extraction cycle, `(logical errors / shots) / cycles`, of the X-memory and the Z-memory experiment; the
LER reported in the paper is the sum of the two. In the code the physical error rate is the argument `p_CNOT` and the number
of cycles is `syndrome_rounds`. The decoders are configured as in the paper:

* **PriorTilt**, three-phase mode: 60 ordinary BP iterations, then modified-BP rounds of 6 iterations with at most R = 3000
  rounds (a total budget of 18,060 BP iterations), returning the first syndrome-consistent solution. The parameter sets
  (θ, π, c₁, c₂) of the three phases and the phase boundaries (b₁, b₂) are optimized for each code and physical error rate; they
  are listed in `priortilt/params.py` (`THREE_PHASE_SCHEDULES`) and recorded in the `decoder_params` column of the data files.
  In the code the parameters are named `threshold_factor` = θ (error-prone condition), `prob_positive` = π (probability of the
  nonnegative offset), `pos_coef` = c₁ and `neg_coef` = c₂ (coefficients of the nonnegative and nonpositive offsets), and a
  phase tuple is ordered (`prob_positive`, `pos_coef`, `neg_coef`, `threshold_factor`); the paper's tables order it (θ, π, c₁, c₂).
* **Relay-BP**: the Relay-BP-1 configuration, 301 legs (an initial leg of 80 iterations and 300 legs of 60 iterations, a total
  budget of 18,080 iterations), returning the first syndrome-consistent solution, with the memory-strength parameters of the
  paper's table: γ_init = 0.125 and [γ_lb, γ_ub] = [-0.24, 0.66] for BB144, 0.125 and [-0.24, 0.50] for BB70, 0.03 and
  [-0.30, 0.45] for GB102.
* **Beam search decoder**: the `beam32_340iters` configuration (beam width 32, 40 initial iterations, 30 per round, 10 rounds).

**Fraction of nonzero prior offsets.** The paper quantifies how sparse the prior perturbations are: the mean percentage of
error nodes that satisfy the error-prone condition per modified-BP round, which bounds the fraction of nonzero prior offsets
from above. It is below 0.5% for all three codes at p = 0.001 and 0.002 (10 to 30 of the 3,850 to 8,784 error nodes). The
decoder exposes the counters behind this number (`PriorTilt.offset_statistics()`), `offset_fraction_table()` in the notebook
re-measures the paper's table, and the paper's rows are stored in `simulation_results/paper_data/offset_fraction.csv`.

**XYZ decoding.** All other results use standard XZ decoding, which decodes the X- and Z-error components separately. For
BB144 at p = 0.003 the paper also performs XYZ decoding, which uses both detector types jointly on the full detector error
model (about 68,000 error mechanisms) and can exploit the correlations between the two error components. PriorTilt uses a
dedicated three-phase schedule (`priortilt.params.xyz_schedule()`, phase boundaries (50, 420)) with at most R = 6000 rounds
and reaches an LER per cycle of 1.04e-5 (5.32e-6 for the X-memory and 5.07e-6 for the Z-memory experiment); Relay-BP-5
(601 legs, five syndrome-consistent solutions) reaches 5.03e-6 and 5.93e-6. The circuits, the schedule and the stored rows
are in the repository; re-simulating this point costs hundreds of CPU-hours per basis (0.6 s per shot on a laptop core), so
the notebook only prints the stored table and leaves the simulation cell commented out.

## Using the decoder on your own circuits

```python
import stim
from priortilt import PriorTilt

circuit = stim.Circuit.from_file("StimCircuit/BB[[144,12,12]],memory_Z,error_rate=0.003,syndrome_rounds=12.stim")
decoder = PriorTilt(circuit.detector_error_model(), code="BB144", p=0.003)   # three-phase mode, the paper's optimized parameters
dets, obs = circuit.compile_detector_sampler().sample(10_000, separate_observables=True)
predictions = decoder.decode_batch(dets)                                       # bool array, shots x observables
print((predictions != obs).any(axis=1).sum(), "logical errors")
```

PriorTilt's logical error rate depends strongly on its parameters, and the optimum depends on the code and on the physical
error rate. The parameters shipped here are for the three codes and the circuits above; for another code, circuit or noise
model a schedule has to be optimized first and passed explicitly (`PriorTilt(dem, schedule={"ends": [b1, b2], "phases":
[...]})`; the paper's uniform mode is a schedule with a single phase and no boundaries). Asking for an unknown code raises an
error on purpose. No uniform-mode parameters are provided, since the paper reports no uniform-mode results.

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
