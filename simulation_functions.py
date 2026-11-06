# This file is part of PriorTiltDecoder.

# Copyright (c) 2026 IonQ, Inc., all rights reserved

# Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International License (CC BY-NC-SA 4.0).

# You may obtain a copy of the License at:
# https://creativecommons.org/licenses/by-nc-sa/4.0/

"""Helper functions used by PriorTilt.ipynb: load the paper's circuits and data, plot the paper's figure, and
re-simulate the PriorTilt points with sinter.

Paper setting reproduced here: PriorTilt in the three-phase mode (one solution, 60 plain BP iterations, tilt
rounds of 6 iterations, at most 3000 tilt rounds, the schedule fitted at each (code, noise rate)), standard X and
Z memory experiments, logical error rate per syndrome extraction cycle = (failures / shots) / syndrome_rounds.
"""

from __future__ import annotations

import json
import multiprocessing
import time
from pathlib import Path
from typing import Dict, Iterable, Optional

import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
import sinter
import stim

from priortilt import PriorTilt
from priortilt.params import THREE_PHASE_SCHEDULES, tuned_config, xyz_schedule
from priortilt.sinter_priortilt import SinterDecoder_PriorTilt

ROOT = Path(__file__).resolve().parent
CIRCUIT_DIR = ROOT / "StimCircuit"
PAPER_DATA_DIR = ROOT / "simulation_results" / "paper_data"
RESULTS_DIR = ROOT / "simulation_results"
TEMP_DIR = RESULTS_DIR / "temp_files"

# The three codes of the paper's figure: family, [[n, k, d]], syndrome rounds of the memory experiment, noise rates.
CODES: Dict[str, dict] = {
    "BB144": dict(family="BB", n=144, k=12, d=12, syndrome_rounds=12, rates=[0.001, 0.002, 0.003, 0.004, 0.005],
                  title="BB [[144,12,12]], 12 cycles"),
    "BB70": dict(family="BB", n=70, k=6, d=9, syndrome_rounds=9, rates=[0.0005, 0.0008, 0.001, 0.002],
                 title="BB [[70,6,9]], 9 cycles"),
    "GB102": dict(family="GB", n=102, k=22, d=9, syndrome_rounds=9, rates=[0.0005, 0.0008, 0.001, 0.002],
                  title="GB [[102,22,9]], 9 cycles"),
}
DECODER_LABELS = {"PriorTilt_3phase": "PriorTilt (3-phase)", "RelayBP": "Relay-BP", "BeamSearch32": "beam search (beam32_340iters)"}
DECODER_STYLES = {
    "PriorTilt_3phase": dict(ls="-", color="C0", marker="o"),
    "RelayBP": dict(ls="--", color="C1", marker="s"),
    "BeamSearch32": dict(ls="-.", color="C2", marker="^"),
}


def _code_name(code) -> str:
    """Accepts "BB144", "bb144", 144 ..."""
    if isinstance(code, int):
        for name, c in CODES.items():
            if c["n"] == code:
                return name
        raise ValueError(f"no code with {code} qubits; choose from {list(CODES)}")
    name = str(code).strip().upper()
    if name not in CODES:
        raise ValueError(f"unknown code {code!r}; choose from {list(CODES)}")
    return name


def _closest_listed_rate(code: str, p: float) -> float:
    rates = CODES[code]["rates"]
    for r in rates:
        if abs(r - p) <= 1e-9 * max(1.0, r):
            return r
    raise ValueError(f"no circuit for {code} at p_CNOT={p:g}; available rates: {rates}")


# ----------------------------------------------------------------------------------------------- circuits


def circuit_file(code, p_CNOT: float, x_or_z: str) -> Path:
    """Path of the stim circuit of the memory experiment (code, p_CNOT, basis "x" or "z")."""
    name = _code_name(code)
    c = CODES[name]
    p = _closest_listed_rate(name, p_CNOT)
    if x_or_z not in ("x", "z"):
        raise ValueError("x_or_z must be 'x' or 'z'")
    return CIRCUIT_DIR / f"{c['family']}[[{c['n']},{c['k']},{c['d']}]],memory_{x_or_z.upper()},error_rate={p:g},syndrome_rounds={c['syndrome_rounds']}.stim"


def load_circuit(code, p_CNOT: float, x_or_z: str) -> stim.Circuit:
    """The noisy stim circuit of the memory experiment (code, p_CNOT, basis)."""
    return stim.Circuit.from_file(str(circuit_file(code, p_CNOT, x_or_z)))


def list_circuits() -> pd.DataFrame:
    """Table of the provided circuits (from StimCircuit/manifest.json)."""
    return pd.DataFrame(json.load(open(CIRCUIT_DIR / "manifest.json")))


# ----------------------------------------------------------------------------------------------- decoder


def priortilt_decoder(code, p_CNOT: float, max_rounds: int = 3000, precision: str = "f64", seed: int = -1) -> SinterDecoder_PriorTilt:
    """sinter decoder: PriorTilt three-phase mode with the schedule fitted for (code, p_CNOT); 3000 rounds = paper."""
    return SinterDecoder_PriorTilt(code=_code_name(code), p=p_CNOT, mode="three_phase", max_rounds=max_rounds, precision=precision, seed=seed)


def decoder_name(code, p_CNOT: float) -> str:
    return f"PriorTilt_3phase_{_code_name(code)}_p{p_CNOT:g}"


def decoder_dictionary(code) -> Dict[str, SinterDecoder_PriorTilt]:
    """One configured sinter decoder per noise rate of `code` (the parameters depend on the rate)."""
    name = _code_name(code)
    return {decoder_name(name, p): priortilt_decoder(name, p) for p in CODES[name]["rates"]}


def _metadata(code: str, p: float, x_or_z: str) -> dict:
    return {"code": code, "p_CNOT": p, "x_or_z": x_or_z, "syndrome_rounds": CODES[code]["syndrome_rounds"]}


def _params_json(code: str, p: float, max_rounds: int = 3000) -> str:
    return json.dumps(tuned_config(code, p, "three_phase", max_rounds=max_rounds).as_dict(), separators=(",", ":"))


# ----------------------------------------------------------------------------------------------- paper data


def load_paper_data(code=None, decoder: Optional[str] = None) -> pd.DataFrame:
    """The data behind the paper's figure (simulation_results/paper_data/paper_figure_data.csv): one row per
    (code, decoder, basis, p) with shots, errors, seconds, ler_per_cycle and its 1-sigma error. Optionally filtered."""
    df = pd.read_csv(PAPER_DATA_DIR / "paper_figure_data.csv")
    if code is not None:
        df = df[df.code == _code_name(code)]
    if decoder is not None:
        df = df[df.decoder == decoder]
    return df.reset_index(drop=True)


def load_sinter_csv(path, decoder_label: Optional[str] = None) -> pd.DataFrame:
    """Tidy table from a sinter-format CSV written by this repository (paper_data/*.csv or full_simulation output)."""
    raw = pd.read_csv(path)
    raw.columns = raw.columns.str.strip()
    md = raw["json_metadata"].apply(json.loads)
    df = pd.DataFrame(
        dict(
            code=md.apply(lambda m: m["code"]),
            decoder=raw["decoder"].str.strip() if decoder_label is None else decoder_label,
            basis=md.apply(lambda m: m["x_or_z"]),
            p=md.apply(lambda m: m["p_CNOT"]),
            syndrome_rounds=md.apply(lambda m: m["syndrome_rounds"]),
            shots=raw["shots"].astype(int),
            errors=raw["errors"].astype(int),
            seconds=raw["seconds"].astype(float),
        )
    )
    # several rows for one point (resumed runs) are pooled
    df = df.groupby(["code", "decoder", "basis", "p", "syndrome_rounds"], as_index=False).agg(
        shots=("shots", "sum"), errors=("errors", "sum"), seconds=("seconds", "sum")
    )
    df["ler_per_cycle"] = df.errors / df.shots / df.syndrome_rounds
    df["ler_err_1sigma"] = np.sqrt(np.maximum(df.errors, 1)) / df.shots / df.syndrome_rounds
    return df.sort_values(["code", "decoder", "basis", "p"]).reset_index(drop=True)


# ----------------------------------------------------------------------------------------------- plotting


def _sum_xz(df: pd.DataFrame) -> pd.DataFrame:
    """x + z logical error rate per cycle per (decoder, p); only points with both bases."""
    g = df.groupby(["decoder", "p"], as_index=False).agg(
        ler=("ler_per_cycle", "sum"), var=("ler_err_1sigma", lambda v: float((v**2).sum())), n=("basis", "count"), errors=("errors", "sum")
    )
    g = g[g.n == 2].copy()
    g["err"] = np.sqrt(g["var"])
    return g


def _style_axis(ax, code: str, rates: Iterable[float], first: bool):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("physical error rate $p$")
    ps = sorted(set(rates))
    ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(ps))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    if first:
        ax.set_ylabel("logical error rate per syndrome extraction cycle")
    ax.set_title(CODES[code]["title"])
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)


def plot_logical_error_rate(code, sum_xz: bool = True, ax=None, own_results: bool = True, show_table: bool = True):
    """Logical error rate per cycle vs p_CNOT for one code: PriorTilt (3-phase), Relay-BP and beam search from the
    paper's data. sum_xz=True draws one curve per decoder (x + z); False draws x (circles) and z (squares) separately.
    own_results=True also draws the points of your own full_simulation(code) run (hollow black markers) when the file
    simulation_results/<code>_PriorTilt_3phase.csv exists. Returns the DataFrame of the rows drawn."""
    name = _code_name(code)
    df = load_paper_data(name)
    if ax is None:
        _, ax = plt.subplots(figsize=(5.2, 4.4))
    for dec in ("PriorTilt_3phase", "RelayBP", "BeamSearch32"):
        sub = df[df.decoder == dec]
        if sub.empty:
            continue
        st = dict(DECODER_STYLES[dec])
        if sum_xz:
            g = _sum_xz(sub)
            ax.errorbar(g.p, g.ler, yerr=g.err, ms=5, capsize=2, label=DECODER_LABELS[dec], **st)
        else:
            for xz, mk in (("x", "o"), ("z", "s")):
                s = sub[sub.basis == xz]
                st["marker"] = mk
                ax.errorbar(s.p, s.ler_per_cycle, yerr=s.ler_err_1sigma, ms=5, capsize=2, label=f"{DECODER_LABELS[dec]}, {xz} memory", **st)
    own_file = RESULTS_DIR / f"{name}_PriorTilt_3phase.csv"
    if own_results and own_file.exists():
        mine = load_sinter_csv(own_file, decoder_label="PriorTilt_3phase")
        mine = mine[mine.errors > 0]
        if sum_xz:
            g = _sum_xz(mine)
            ax.errorbar(g.p, g.ler, yerr=g.err, fmt="D", mfc="none", color="k", ms=7, capsize=2, label="PriorTilt (3-phase), your run")
        else:
            for xz, mk in (("x", "o"), ("z", "s")):
                s = mine[mine.basis == xz]
                ax.errorbar(s.p, s.ler_per_cycle, yerr=s.ler_err_1sigma, fmt=mk, mfc="none", color="k", ms=8, capsize=2, label=f"your run, {xz} memory")
    _style_axis(ax, name, df.p, first=True)
    if show_table:
        print_table(name)
    return df


def plot_paper_figure(sum_xz: bool = True, own_results: bool = False, savefig: Optional[str] = None, show_table: bool = False):
    """The paper's figure: one panel per code (BB144, BB70, GB102), three decoders. savefig="file.pdf" writes it."""
    codes = list(CODES)
    fig, axes = plt.subplots(1, len(codes), figsize=(5.2 * len(codes), 4.4))
    for i, (ax, code) in enumerate(zip(axes, codes)):
        plot_logical_error_rate(code, sum_xz=sum_xz, ax=ax, own_results=own_results, show_table=show_table)
        if i > 0:
            ax.set_ylabel("")
    fig.tight_layout()
    if savefig:
        fig.savefig(savefig, dpi=150, bbox_inches="tight")
    return fig


def print_table(code):
    """Per-basis and x+z logical error rates of the three decoders with the PriorTilt / Relay-BP and PriorTilt / beam ratios."""
    name = _code_name(code)
    df = load_paper_data(name)
    print(f"\n{CODES[name]['title']}: LER per cycle (PriorTilt 3-phase / Relay-BP / beam32), ratios, errors")
    print(f"{'basis':>5s} {'p':>7s} {'PriorTilt':>11s} {'Relay-BP':>11s} {'beam32':>11s} {'PT/RBP':>7s} {'PT/beam':>8s} {'errors PT/RBP/beam':>20s}")
    piv = df.pivot_table(index=["basis", "p"], columns="decoder", values=["ler_per_cycle", "errors"])
    for (xz, p), row in piv.iterrows():
        pt, rb, bm = (row[("ler_per_cycle", d)] for d in ("PriorTilt_3phase", "RelayBP", "BeamSearch32"))
        e = "/".join(f"{int(row[('errors', d)])}" for d in ("PriorTilt_3phase", "RelayBP", "BeamSearch32"))
        print(f"{xz:>5s} {p:7.4f} {pt:11.3e} {rb:11.3e} {bm:11.3e} {pt / rb:7.2f} {pt / bm:8.2f} {e:>20s}")
    tot = {d: _sum_xz(df[df.decoder == d]).set_index("p") for d in ("PriorTilt_3phase", "RelayBP", "BeamSearch32")}
    for p in sorted(df.p.unique()):
        if all(p in t.index for t in tot.values()):
            pt, rb, bm = (tot[d].loc[p, "ler"] for d in ("PriorTilt_3phase", "RelayBP", "BeamSearch32"))
            e = "/".join(f"{int(tot[d].loc[p, 'errors'])}" for d in ("PriorTilt_3phase", "RelayBP", "BeamSearch32"))
            print(f"{'x+z':>5s} {p:7.4f} {pt:11.3e} {rb:11.3e} {bm:11.3e} {pt / rb:7.2f} {pt / bm:8.2f} {e:>20s}")


# ----------------------------------------------------------------------------------------------- simulation


def _num_workers(num_workers: Optional[int]) -> int:
    return num_workers if num_workers else max(1, multiprocessing.cpu_count() - 1)


def one_point_simulation(
    code="BB144",
    p_CNOT: float = 0.005,
    basis: str = "z",
    maximum_shots: int = 1_000_000,
    maximum_errors: int = 100,
    max_rounds: int = 3000,
    num_workers: Optional[int] = None,
) -> pd.DataFrame:
    """Simulate the memory experiment(s) of `code` at `p_CNOT` with PriorTilt (three-phase mode, the schedule fitted
    for this point, `max_rounds` tilt rounds; 3000 = paper) until `maximum_errors` logical failures or `maximum_shots`
    shots per basis, and compare with the paper's values. `basis` is "z" (Z memory), "x" (X memory) or "xz" (both;
    the x + z sum is what the paper's figure shows). sinter prints its progress (shots and failures collected so far)
    while the simulation runs.

    The defaults (BB144, Z memory, p_CNOT = 0.005) take about a minute on a laptop with 11 worker processes (about
    4,000 shots for 100 failures); basis="xz" doubles that. p_CNOT = 0.004 takes about 3x longer (34,000 shots per
    basis) and p_CNOT = 0.003 about 6x longer (320,000 shots per basis). The paper's points at p_CNOT = 0.001 needed
    7-9 x 10^8 shots per basis: see full_simulation for those.
    Returns a DataFrame with one row per basis (shots, errors, seconds, LER per cycle)."""
    name = _code_name(code)
    p = _closest_listed_rate(name, p_CNOT)
    bases = {"z": ("z",), "x": ("x",), "xz": ("x", "z"), "zx": ("x", "z")}.get(str(basis).lower())
    if bases is None:
        raise ValueError(f"basis must be 'x', 'z' or 'xz', got {basis!r}")
    rounds = CODES[name]["syndrome_rounds"]
    dec_name = decoder_name(name, p)
    decoders = {dec_name: priortilt_decoder(name, p, max_rounds=max_rounds)}
    print(f"PriorTilt three-phase mode on {CODES[name]['title']} at p_CNOT = {p:g} ({max_rounds} rounds)")
    print(f"  schedule: {tuned_config(name, p, 'three_phase').describe()}")
    rows = []
    paper = load_paper_data(name, "PriorTilt_3phase")
    for xz in bases:
        t0 = time.time()
        stats = sinter.collect(
            num_workers=_num_workers(num_workers),
            max_shots=maximum_shots,
            max_errors=maximum_errors,
            tasks=[sinter.Task(circuit=load_circuit(name, p, xz), json_metadata=_metadata(name, p, xz))],
            decoders=[dec_name],
            custom_decoders=decoders,
            print_progress=True,
        )
        s = stats[0]
        ler = s.errors / s.shots / rounds
        ref = paper[(paper.basis == xz) & np.isclose(paper.p, p)]
        ref_ler = float(ref.ler_per_cycle.iloc[0]) if len(ref) else float("nan")
        rows.append(dict(code=name, basis=xz, p=p, syndrome_rounds=rounds, shots=s.shots, errors=s.errors, seconds=s.seconds,
                         ler_per_cycle=ler, ler_err_1sigma=np.sqrt(max(s.errors, 1)) / s.shots / rounds, paper_ler_per_cycle=ref_ler))
        print(f"  memory_{xz.upper()}: {s.errors} failures in {s.shots} shots -> LER per cycle {ler:.3e} "
              f"(+-{100 / np.sqrt(max(s.errors, 1)):.0f}%); paper: {ref_ler:.3e}; wall time {time.time() - t0:.0f} s")
    df = pd.DataFrame(rows)
    if len(bases) == 2:
        print(f"  x + z: {df.ler_per_cycle.sum():.3e} (paper: {df.paper_ler_per_cycle.sum():.3e})")
    return df


def generate_tasks(code, rates: Optional[Iterable[float]] = None):
    """sinter tasks of the paper's PriorTilt points for `code`: both bases at every noise rate (or the given rates)."""
    name = _code_name(code)
    for p in (CODES[name]["rates"] if rates is None else rates):
        p = _closest_listed_rate(name, p)
        for xz in ("x", "z"):
            yield sinter.Task(circuit=load_circuit(name, p, xz), decoder=decoder_name(name, p), json_metadata=_metadata(name, p, xz))


def full_simulation(
    code,
    maximum_shots: int = 1_000_000_000,
    maximum_errors: int = 100,
    rates: Optional[Iterable[float]] = None,
    num_workers: Optional[int] = None,
    max_rounds: int = 3000,
) -> Path:
    """Re-run the paper's PriorTilt curve for `code`: every noise rate (or `rates`), both bases, until `maximum_errors`
    failures or `maximum_shots` shots per point. The result is written in sinter CSV format (with the decoder
    parameters of each row) to simulation_results/<code>_PriorTilt_3phase.csv and picked up by
    plot_logical_error_rate(code, own_results=True).

    WARNING: this is a very long computation. The paper's points at the lowest noise rates used up to 10^9 shots per
    basis, i.e. hundreds of CPU-days per code; at the highest rate of each code a point takes minutes. Choose `rates`
    and `maximum_shots` accordingly. The run is resumable: existing rows of the output file are reused and only the
    missing shots are collected."""
    name = _code_name(code)
    final_file = RESULTS_DIR / f"{name}_PriorTilt_3phase.csv"
    temp_file = TEMP_DIR / f"{name}_PriorTilt_3phase.csv"
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    decoders = {decoder_name(name, p): priortilt_decoder(name, p, max_rounds=max_rounds) for p in CODES[name]["rates"]}
    existing = [str(f) for f in (final_file, temp_file) if f.exists()]
    sinter.collect(
        num_workers=_num_workers(num_workers),
        max_shots=maximum_shots,
        max_errors=maximum_errors,
        tasks=list(generate_tasks(name, rates)),
        custom_decoders=decoders,
        existing_data_filepaths=existing,
        save_resume_filepath=str(temp_file),
        print_progress=True,
    )
    # merge the resume file into the final file, adding the decoder parameters of each row
    frames = [pd.read_csv(f) for f in (final_file, temp_file) if f.exists()]
    df = pd.concat(frames, ignore_index=True)
    df.columns = df.columns.str.strip()
    if "decoder_params" not in df.columns:
        df["decoder_params"] = ""
    df["decoder"] = df["decoder"].str.strip()
    md = df["json_metadata"].apply(json.loads)
    df["decoder_params"] = [
        _params_json(name, m["p_CNOT"], max_rounds) if not isinstance(v, str) or not v.strip() else v
        for m, v in zip(md, df["decoder_params"])
    ]
    df = df.drop_duplicates(subset=["strong_id"] if "strong_id" in df.columns else None)
    df.to_csv(final_file, index=False)
    if temp_file.exists():
        temp_file.unlink()
    print("->", final_file)
    return final_file


def decoding_time(code="BB144", p_CNOT: float = 0.001, num_shots: int = 10_000, max_rounds: int = 3000) -> dict:
    """Single-thread decoding time of PriorTilt (three-phase mode) on the Z memory experiment: mean and 99th
    percentile of the per-shot wall time (one decode call per shot) over `num_shots` sampled shots, the two
    statistics reported in the paper. The 99th percentile rests on num_shots / 100 shots."""
    name = _code_name(code)
    p = _closest_listed_rate(name, p_CNOT)
    circuit = load_circuit(name, p, "z")
    dec = PriorTilt(circuit.detector_error_model(), code=name, p=p, max_rounds=max_rounds)
    dets, obs = circuit.compile_detector_sampler().sample(num_shots, separate_observables=True)
    dets = dets.astype(np.uint8)
    t = np.empty(num_shots)
    failures = 0
    for i in range(num_shots):
        t0 = time.perf_counter()
        pred = dec.decode(dets[i])
        t[i] = time.perf_counter() - t0
        failures += int(np.any(pred.astype(bool) != obs[i]))
    out = dict(code=name, p=p, shots=num_shots, mean_ms=1e3 * t.mean(), p99_ms=1e3 * np.quantile(t, 0.99), failures=failures)
    print(f"PriorTilt three-phase on {CODES[name]['title']}, memory_Z, p_CNOT = {p:g}, {num_shots} shots: "
          f"mean {out['mean_ms']:.3f} ms, 99th percentile {out['p99_ms']:.2f} ms, {failures} logical failures")
    return out


# ---------------------------------------------------------------------------------------------------------------------
# Fraction of bits with a nonzero prior offset per tilt round (the table of the paper's "only a few bits are kicked"
# paragraph). Measured with the decoder's diagnostic counters (PriorTilt.tilt_statistics).
# ---------------------------------------------------------------------------------------------------------------------

#: Shots per basis used in the paper for the offset-fraction table.
OFFSET_FRACTION_SHOTS = {0.001: 100_000, 0.002: 30_000}


def load_paper_offset_fraction(code=None) -> pd.DataFrame:
    """The paper's measurement of the fraction of bits with a nonzero prior offset per tilt round (three-phase mode),
    one row per (code, p_CNOT, basis): shots, shots that reached the tilt phase, tilt rounds, nonzero offsets,
    offset_fraction = nonzero_offsets / (tilt_rounds * num_error_mechanisms), per-shot 90th percentile and maximum."""
    df = pd.read_csv(PAPER_DATA_DIR / "offset_fraction.csv")
    if code is not None:
        df = df[df.code == _code_name(code)]
    return df.reset_index(drop=True)


def _pooled_offset_fraction(df: pd.DataFrame) -> float:
    """Numerator and denominator summed over the rows (the paper's table pools the X and Z memory experiments)."""
    return float(df.nonzero_offsets.sum() / (df.tilt_rounds * df.num_error_mechanisms).sum())


def _offset_fraction_worker(args):
    code, p, xz, num_shots, sampler_seed, decoder_seed, max_rounds = args
    circuit = load_circuit(code, p, xz)
    dec = PriorTilt(circuit.detector_error_model(), code=code, p=p, max_rounds=max_rounds, seed=decoder_seed)
    dets, obs = circuit.compile_detector_sampler(seed=sampler_seed).sample(num_shots, separate_observables=True)
    dets = dets.astype(np.uint8)
    nonzero = bit_rounds = tilt_rounds = 0
    per_shot, failures = [], 0
    for i in range(num_shots):
        dec.reset_tilt_statistics()
        pred = dec.decode(dets[i])
        st = dec.tilt_statistics()
        if st["bit_rounds"]:  # the shot reached the tilt phase
            per_shot.append(st["nonzero_offsets"] / st["bit_rounds"])
            nonzero += st["nonzero_offsets"]
            bit_rounds += st["bit_rounds"]
            tilt_rounds += st["tilt_rounds"]
        failures += int(np.any(pred.astype(bool) != obs[i]))
    return dict(nonzero=nonzero, bit_rounds=bit_rounds, tilt_rounds=tilt_rounds, per_shot=per_shot, failures=failures,
                num_error_mechanisms=dec.num_errors)


def offset_fraction(
    code,
    p_CNOT: float,
    basis: str = "xz",
    num_shots: Optional[int] = None,
    num_workers: Optional[int] = None,
    max_rounds: int = 3000,
    seed: int = 7,
) -> pd.DataFrame:
    """Measure the fraction of bits (error mechanisms) that receive a nonzero prior offset per tilt round, for PriorTilt
    in the three-phase mode at (`code`, `p_CNOT`), and compare with the paper's value.

    Definition (as in the paper): decode `num_shots` sampled shots per basis (default: the paper's 100,000 at
    p_CNOT = 0.001 and 30,000 at 0.002). Only shots that are still unconverged after the plain BP phase enter the tilt
    rounds; at the end of every tilt round the decoder counts the bits whose LLR sum is below threshold_factor x prior,
    i.e. the bits that get a nonzero offset. offset_fraction = (number of (bit, tilt round) pairs with a nonzero offset)
    / (number of tilt rounds x number of error mechanisms), summed over all contributing shots, so every tilt round has
    the same weight. The per-shot columns apply the same ratio to each contributing shot on its own.

    Returns one row per basis with the measured and the paper's values; the printed "x + z pooled" value sums numerator
    and denominator over both bases, which is the number in the paper's table. The shots and the decoder seeds differ
    from the paper's run, so agreement is statistical (within about 10% for BB144 and BB70; GB102 at p_CNOT = 0.001
    rests on fewer than 100 contributing shots per basis and fluctuates by about 20%). Runs on `num_workers` processes."""
    name = _code_name(code)
    p = _closest_listed_rate(name, p_CNOT)
    bases = {"z": ("z",), "x": ("x",), "xz": ("x", "z"), "zx": ("x", "z")}.get(str(basis).lower())
    if bases is None:
        raise ValueError(f"basis must be 'x', 'z' or 'xz', got {basis!r}")
    shots = int(num_shots if num_shots is not None else OFFSET_FRACTION_SHOTS.get(p, 30_000))
    workers = _num_workers(num_workers)
    paper = load_paper_offset_fraction(name)
    print(f"PriorTilt three-phase mode on {CODES[name]['title']} at p_CNOT = {p:g}: {shots} shots per basis, {workers} workers")
    rows = []
    for xz in bases:
        t0 = time.time()
        chunks = np.diff(np.linspace(0, shots, workers + 1).astype(int))
        tasks = [(name, p, xz, int(n), seed * 1_000_003 + 1000 * k + (xz == "z"), seed * 7_919 + 100 * k + (xz == "z"), max_rounds)
                 for k, n in enumerate(chunks) if n > 0]
        with multiprocessing.get_context("spawn").Pool(len(tasks)) as pool:
            parts = pool.map(_offset_fraction_worker, tasks)
        nonzero = sum(r["nonzero"] for r in parts)
        bit_rounds = sum(r["bit_rounds"] for r in parts)
        tilt_rounds = sum(r["tilt_rounds"] for r in parts)
        per_shot = np.array(sum((r["per_shot"] for r in parts), []))
        n_err = parts[0]["num_error_mechanisms"]
        ref = paper[(paper.basis == xz) & np.isclose(paper.p, p)]
        rows.append(dict(
            code=name, basis=xz, p=p, syndrome_rounds=CODES[name]["syndrome_rounds"], shots=shots,
            num_error_mechanisms=n_err, shots_reaching_tilt_phase=len(per_shot), tilt_rounds=tilt_rounds, nonzero_offsets=nonzero,
            offset_fraction=nonzero / bit_rounds if bit_rounds else float("nan"),
            per_shot_p90=float(np.quantile(per_shot, 0.9)) if len(per_shot) else float("nan"),
            per_shot_max=float(per_shot.max()) if len(per_shot) else float("nan"),
            failures=sum(r["failures"] for r in parts),
            paper_offset_fraction=float(ref.offset_fraction.iloc[0]) if len(ref) else float("nan"),
        ))
        r = rows[-1]
        print(f"  memory_{xz.upper()}: {r['shots_reaching_tilt_phase']} shots reached the tilt phase, {tilt_rounds} tilt rounds, "
              f"nonzero offsets on {100 * r['offset_fraction']:.3f}% of the {n_err} bits per tilt round "
              f"(paper: {100 * r['paper_offset_fraction']:.3f}%); per-shot 90th pct {100 * r['per_shot_p90']:.2f}%, "
              f"max {100 * r['per_shot_max']:.2f}%; wall time {time.time() - t0:.0f} s")
    df = pd.DataFrame(rows)
    if len(bases) == 2:
        ref = paper[np.isclose(paper.p, p)]
        print(f"  x + z pooled: {100 * _pooled_offset_fraction(df):.2f}% of bits per tilt round"
              f" (paper: {100 * _pooled_offset_fraction(ref):.2f}%)")
    return df


def offset_fraction_table(
    codes: Iterable[str] = ("BB70",),
    rates: Iterable[float] = (0.001, 0.002),
    num_shots: Optional[int] = None,
    num_workers: Optional[int] = None,
) -> pd.DataFrame:
    """Reproduce the paper's table of the fraction of bits with a nonzero prior offset per tilt round (three-phase
    mode, X and Z pooled) for the given codes and noise rates; see offset_fraction for the definition and the sample
    sizes. The default, BB70 at both rates, takes about half a minute on a laptop with 11 worker processes; the
    paper's full table is codes=("BB144", "BB70", "GB102") and takes about 4x longer. Returns the per-basis rows of
    all cells; prints the pooled table next to the paper's values."""
    frames = [offset_fraction(c, p, num_shots=num_shots, num_workers=num_workers) for c in codes for p in rates]
    df = pd.concat(frames, ignore_index=True)
    paper = load_paper_offset_fraction()
    print(f"\n{'code (n)':>16s} {'p_CNOT':>7s} {'measured':>9s} {'paper':>7s}   (fraction of bits with a nonzero offset per tilt round, x + z pooled)")
    for (c, p), sub in df.groupby(["code", "p"], sort=False):
        ref = paper[(paper.code == c) & np.isclose(paper.p, p)]
        print(f"{c + ' (' + format(int(sub.num_error_mechanisms.iloc[0]), ',') + ')':>16s} {p:>7g} "
              f"{100 * _pooled_offset_fraction(sub):>8.2f}% {100 * _pooled_offset_fraction(ref):>6.2f}%")
    return df


# ---------------------------------------------------------------------------------------------------------------------
# XYZ decoding of BB144 at p_CNOT = 0.003 (the paper's correlated-decoding experiment)
# ---------------------------------------------------------------------------------------------------------------------

#: The (code, p_CNOT) pairs with XYZ circuits, schedule and stored results.
XYZ_CASES = {("BB144", 0.003)}
XYZ_MAX_ROUNDS = 6000


def _xyz_case(code, p_CNOT: float):
    name = _code_name(code)
    p = _closest_listed_rate(name, p_CNOT)
    if (name, p) not in XYZ_CASES:
        raise ValueError(f"XYZ decoding is provided only for {sorted(XYZ_CASES)}, got ({name}, {p:g})")
    return name, p


def xyz_circuit_file(basis: str = "z", code="BB144", p_CNOT: float = 0.003) -> Path:
    """Path of the XYZ-decoding circuit: the memory experiment in `basis` whose detectors after the first round
    compare all stabiliser measurements (X and Z), so that the decoder sees the full detector error model."""
    name, p = _xyz_case(code, p_CNOT)
    c = CODES[name]
    if basis not in ("x", "z"):
        raise ValueError("basis must be 'x' or 'z'")
    return CIRCUIT_DIR / f"{c['family']}[[{c['n']},{c['k']},{c['d']}]],xyz_memory_{basis.upper()},error_rate={p:g},syndrome_rounds={c['syndrome_rounds']}.stim"


def load_xyz_circuit(basis: str = "z", code="BB144", p_CNOT: float = 0.003) -> stim.Circuit:
    return stim.Circuit.from_file(str(xyz_circuit_file(basis, code, p_CNOT)))


def xyz_decoder(code="BB144", p_CNOT: float = 0.003, max_rounds: int = XYZ_MAX_ROUNDS, precision: str = "f64", seed: int = -1) -> SinterDecoder_PriorTilt:
    """sinter decoder of the paper's XYZ experiment: PriorTilt three-phase mode with the XYZ schedule, one solution,
    6000 rounds (the paper's setting)."""
    name, p = _xyz_case(code, p_CNOT)
    sched = xyz_schedule(name, p)
    return SinterDecoder_PriorTilt(schedule={"ends": sched["ends"], "phases": sched["phases"]}, max_rounds=max_rounds, precision=precision, seed=seed)


def xyz_decoder_name(code="BB144", p_CNOT: float = 0.003) -> str:
    return f"PriorTilt_3phase_XYZ_{_code_name(code)}_p{p_CNOT:g}"


def load_paper_xyz(code="BB144", p_CNOT: float = 0.003) -> pd.DataFrame:
    """The paper's XYZ-decoding results (one row per decoder and basis) with the LER per cycle and its 1-sigma error:
    PriorTilt three-phase (6000 rounds, one solution) and the Relay-BP reference (601 legs, 5 solutions)."""
    name, p = _xyz_case(code, p_CNOT)
    return load_sinter_csv(PAPER_DATA_DIR / f"{name}_XYZ_p{p:g}.csv")


def print_xyz_table(code="BB144", p_CNOT: float = 0.003) -> pd.DataFrame:
    """Print the paper's XYZ-decoding table: LER per cycle of each decoder for the X and Z memory experiments and
    their sum, with the failure counts and the PriorTilt / Relay-BP ratio."""
    name, p = _xyz_case(code, p_CNOT)
    df = load_paper_xyz(name, p)
    piv = df.pivot_table(index="decoder", columns="basis", values=["ler_per_cycle", "errors", "shots"])
    print(f"XYZ decoding of {CODES[name]['title']} at p_CNOT = {p:g} (LER per cycle; failures / shots)")
    print(f"{'decoder':>18s} {'xyz_x':>10s} {'xyz_z':>10s} {'x + z':>10s}   {'failures / shots (x; z)':>40s}")
    tot = {}
    for dec in ("PriorTilt_3phase", "RelayBP_5results"):
        lx, lz = piv.loc[dec, ("ler_per_cycle", "xyz_x")], piv.loc[dec, ("ler_per_cycle", "xyz_z")]
        tot[dec] = lx + lz
        fx = f"{int(piv.loc[dec, ('errors', 'xyz_x')])} / {int(piv.loc[dec, ('shots', 'xyz_x')]):,}"
        fz = f"{int(piv.loc[dec, ('errors', 'xyz_z')])} / {int(piv.loc[dec, ('shots', 'xyz_z')]):,}"
        print(f"{dec:>18s} {lx:>10.2e} {lz:>10.2e} {lx + lz:>10.2e}   {fx + ';  ' + fz:>40s}")
    print(f"PriorTilt / Relay-BP (5 results): {tot['PriorTilt_3phase'] / tot['RelayBP_5results']:.2f}")
    return df


def xyz_one_point_simulation(
    basis: str = "z",
    maximum_shots: int = 2_000_000,
    maximum_errors: int = 100,
    code="BB144",
    p_CNOT: float = 0.003,
    max_rounds: int = XYZ_MAX_ROUNDS,
    num_workers: Optional[int] = None,
) -> pd.DataFrame:
    """Re-simulate the paper's XYZ point with PriorTilt (three-phase mode, XYZ schedule, one solution, 6000 rounds)
    on the `basis` memory experiment until `maximum_errors` failures or `maximum_shots` shots, and compare with the
    paper's value.

    WARNING: this is NOT a laptop-scale computation. The XYZ detector error model has about 68,000 error mechanisms
    and a shot takes about 0.6 s on a laptop core (2.5 s on the cluster nodes used for the paper); the paper's 100
    failures needed 1.6 million shots per basis, i.e. hundreds of CPU-hours per basis (9.4 hours on 238 cores for
    both bases). A laptop can only afford a few thousand shots, which at an LER of 5e-6 per cycle (6e-5 per shot)
    will most likely see zero failures. Use `maximum_shots` to bound the run."""
    name, p = _xyz_case(code, p_CNOT)
    bases = {"z": ("z",), "x": ("x",), "xz": ("x", "z"), "zx": ("x", "z")}.get(str(basis).lower())
    if bases is None:
        raise ValueError(f"basis must be 'x', 'z' or 'xz', got {basis!r}")
    rounds = CODES[name]["syndrome_rounds"]
    dec_name = xyz_decoder_name(name, p)
    decoders = {dec_name: xyz_decoder(name, p, max_rounds=max_rounds)}
    sched = xyz_schedule(name, p)
    print(f"PriorTilt three-phase mode, XYZ decoding of {CODES[name]['title']} at p_CNOT = {p:g} ({max_rounds} rounds)")
    print(f"  schedule: ends={sched['ends']}, phases={sched['phases']}")
    paper = load_paper_xyz(name, p)
    rows = []
    for xz in bases:
        t0 = time.time()
        md = {"code": name, "p_CNOT": p, "x_or_z": f"xyz_{xz}", "syndrome_rounds": rounds, "experiment": "xyz"}
        stats = sinter.collect(
            num_workers=_num_workers(num_workers), max_shots=maximum_shots, max_errors=maximum_errors,
            tasks=[sinter.Task(circuit=load_xyz_circuit(xz, name, p), json_metadata=md)],
            decoders=[dec_name], custom_decoders=decoders, print_progress=True,
        )
        s = stats[0]
        ler = s.errors / s.shots / rounds
        ref = paper[(paper.decoder == "PriorTilt_3phase") & (paper.basis == f"xyz_{xz}")]
        ref_ler = float(ref.ler_per_cycle.iloc[0]) if len(ref) else float("nan")
        rows.append(dict(code=name, basis=f"xyz_{xz}", p=p, syndrome_rounds=rounds, shots=s.shots, errors=s.errors, seconds=s.seconds,
                         ler_per_cycle=ler, ler_err_1sigma=np.sqrt(max(s.errors, 1)) / s.shots / rounds, paper_ler_per_cycle=ref_ler))
        print(f"  xyz_{xz}: {s.errors} failures in {s.shots} shots -> LER per cycle {ler:.3e}; paper: {ref_ler:.3e}; "
              f"wall time {time.time() - t0:.0f} s")
    return pd.DataFrame(rows)
