// This file is part of PriorTiltDecoder.
//
// Copyright (c) 2026 IonQ, Inc., all rights reserved
//
// Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
// 4.0 International License (CC BY-NC-SA 4.0).
//
// You may obtain a copy of the License at:
// https://creativecommons.org/licenses/by-nc-sa/4.0/

//! PriorTilt: min-sum belief propagation with transient prior tilts (single-solution decoder), Rust core.
//!
//! Algorithm:
//!   Phase 1: `initial_iters` iterations of parallel min-sum BP; return at once if the syndrome is satisfied.
//!   Phase 2: BP continues without resetting messages, in rounds of `round_len` iterations, for at most
//!     `max_rounds` rounds. Per bit, over a round: LLR_sums = sum of the posterior; run_min / run_max = min / max
//!     over the posterior and all outgoing bit->check messages. At the end of a round, a bit with
//!     LLR_sums < threshold_factor * init gets a prior offset for the next round:
//!       with probability prob_positive:  +pos_coef * max(0, -run_min)
//!       otherwise:                       -neg_coef * max(0,  run_max)
//!     Offsets are overwritten every round. The first syndrome-consistent hard decision is returned.
//!   Three-phase mode: `set_round_schedule` installs block boundaries and one (prob_positive, pos_coef,
//!     neg_coef, threshold_factor) tuple per block; the tilt computed after `completed` finished tilt rounds
//!     uses the block that contains `completed`.
//!
//! Data layout: messages in two flat arrays indexed by edge id, edges numbered in column (bit) order so each
//! bit's edges are contiguous; each check holds an index list (u32) of its edge ids. One fused pass per check
//! and one fused pass per bit per iteration; hot loops use unchecked indexing on indices that are valid by
//! construction.
//!
//! Two Python classes: `PriorTiltDecoder` (f64 messages, same arithmetic as the C++) and `PriorTiltDecoderF32`
//! (f32 messages: faster, results equal only statistically).

use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn make_rng(seed: i64) -> StdRng {
    if seed < 0 {
        StdRng::from_entropy()
    } else {
        StdRng::seed_from_u64(seed as u64)
    }
}

/// Extract (rows, cols, indptr, indices) of a scipy sparse (or dense) binary matrix in CSC form.
fn csc_from_python(matrix: &Bound<'_, PyAny>) -> PyResult<(usize, usize, Vec<usize>, Vec<usize>)> {
    let py = matrix.py();
    let scipy_sparse = PyModule::import(py, "scipy.sparse")?;
    let m = if scipy_sparse.getattr("issparse")?.call1((matrix,))?.is_truthy()? {
        matrix.call_method0("tocsc")?
    } else {
        scipy_sparse.getattr("csc_matrix")?.call1((matrix,))?
    };
    let m = m.call_method1("astype", ("uint8",))?;
    let m = m.call_method0("sorted_indices")?;
    let shape = m.getattr("shape")?;
    let rows: usize = shape.get_item(0)?.extract()?;
    let cols: usize = shape.get_item(1)?.extract()?;
    let indptr: Vec<usize> = m.getattr("indptr")?.call_method0("tolist")?.extract()?;
    let indices: Vec<usize> = m.getattr("indices")?.call_method0("tolist")?.extract()?;
    let data: Vec<u8> = m.getattr("data")?.call_method0("tolist")?.extract()?;
    if data.iter().any(|&d| d != 1) {
        return Err(PyValueError::new_err("check matrix must be binary"));
    }
    if indptr.len() != cols + 1 {
        return Err(PyTypeError::new_err("unexpected CSC indptr length"));
    }
    Ok((rows, cols, indptr, indices))
}

#[derive(Clone, Copy)]
pub struct Params {
    pub initial_iters: usize,
    pub round_len: usize,
    pub max_rounds: usize,
    pub threshold_factor: f64,
    pub neg_coef: f64,
    pub pos_coef: f64,
    pub prob_positive: f64,
}

macro_rules! make_core {
    ($core:ident, $t:ty) => {
        pub struct $core {
            check_count: usize,
            bit_count: usize,
            col_ptr: Vec<u32>,
            edge_row: Vec<u32>,
            row_ptr: Vec<u32>,
            row_edges: Vec<u32>,
            b2c: Vec<$t>,
            c2b: Vec<$t>,
            init_llr: Vec<$t>,
            post: Vec<$t>,
            offsets: Vec<$t>,
            llr_sums: Vec<$t>,
            run_min: Vec<$t>,
            run_max: Vec<$t>,
            decoding: Vec<u8>,
            cand_syn: Vec<u8>,
            // round schedule (three-phase mode): block k covers completed rounds in [ends[k-1], ends[k]);
            // the last tuple applies beyond the last end; empty = uniform mode
            sched_ends: Vec<usize>,
            sched_ppos: Vec<f64>,
            sched_pos: Vec<f64>,
            sched_neg: Vec<f64>,
            sched_thr: Vec<f64>,
            rng: StdRng,
            pub params: Params,
            pub seed: i64,
            pub iterations: usize,
            pub tilt_rounds: usize,
            pub converged: bool,
            // diagnostic counters, accumulated over decodes until reset_tilt_statistics():
            pub stat_uncertain: u64,  // (bit, tilt round) pairs that received a nonzero prior offset
            pub stat_bit_rounds: u64, // bits examined in those offset computations (= bit_count per tilt round)
            pub stat_tilts: u64,      // offset computations (= tilt rounds)
        }

        impl $core {
            fn new(check_matrix: &Bound<'_, PyAny>, priors: &[f64], params: Params, seed: i64) -> PyResult<Self> {
                let (check_count, bit_count, col_ptr, edge_row) = csc_from_python(check_matrix)?;
                if priors.len() != bit_count {
                    return Err(PyValueError::new_err(format!(
                        "error_priors has length {} but the check matrix has {} columns",
                        priors.len(),
                        bit_count
                    )));
                }
                let edge_count = edge_row.len();
                if edge_count >= u32::MAX as usize || check_count >= u32::MAX as usize {
                    return Err(PyValueError::new_err("matrix too large for 32-bit indices"));
                }
                let mut row_ptr = vec![0usize; check_count + 1];
                for &r in &edge_row {
                    row_ptr[r + 1] += 1;
                }
                for j in 0..check_count {
                    row_ptr[j + 1] += row_ptr[j];
                }
                let mut fill = row_ptr[..check_count].to_vec();
                let mut row_edges = vec![0u32; edge_count];
                for (e, &r) in edge_row.iter().enumerate() {
                    row_edges[fill[r]] = e as u32;
                    fill[r] += 1;
                }
                let init_llr: Vec<$t> = priors.iter().map(|&p| ((((1.0 - p) / p).ln()) as $t)).collect();
                Ok(Self {
                    check_count,
                    bit_count,
                    col_ptr: col_ptr.iter().map(|&v| v as u32).collect(),
                    edge_row: edge_row.iter().map(|&v| v as u32).collect(),
                    row_ptr: row_ptr.iter().map(|&v| v as u32).collect(),
                    row_edges,
                    b2c: vec![(0.0 as $t); edge_count],
                    c2b: vec![(0.0 as $t); edge_count],
                    init_llr,
                    post: vec![(0.0 as $t); bit_count],
                    offsets: vec![(0.0 as $t); bit_count],
                    llr_sums: vec![(0.0 as $t); bit_count],
                    run_min: vec![(0.0 as $t); bit_count],
                    run_max: vec![(0.0 as $t); bit_count],
                    decoding: vec![0u8; bit_count],
                    cand_syn: vec![0u8; check_count],
                    sched_ends: Vec::new(),
                    sched_ppos: Vec::new(),
                    sched_pos: Vec::new(),
                    sched_neg: Vec::new(),
                    sched_thr: Vec::new(),
                    rng: make_rng(seed),
                    params,
                    seed,
                    iterations: 0,
                    tilt_rounds: 0,
                    converged: false,
                    stat_uncertain: 0,
                    stat_bit_rounds: 0,
                    stat_tilts: 0,
                })
            }

            fn reseed(&mut self, seed: i64) {
                self.seed = seed;
                self.rng = make_rng(seed);
            }

            fn reset_round_stats(&mut self) {
                for i in 0..self.bit_count {
                    self.llr_sums[i] = (0.0 as $t);
                    self.run_min[i] = <$t>::MAX;
                    self.run_max[i] = -<$t>::MAX;
                }
            }

            /// One parallel min-sum iteration with each bit's prior shifted by offsets[i]. Returns true on convergence.
            ///
            /// Unchecked indexing: every index comes from `row_ptr`/`col_ptr`/`row_edges`/`edge_row`, built in the
            /// constructor from a valid CSC structure; `syndrome.len() == check_count` is verified by the caller.
            #[inline(never)]
            fn bp_iteration(&mut self, syndrome: &[u8]) -> bool {
                debug_assert_eq!(syndrome.len(), self.check_count);
                let inf = <$t>::MAX;
                let zero = (0.0 as $t);
                let check_count = self.check_count;
                let bit_count = self.bit_count;
                // ---- check -> bit (min-sum): forward running min, backward combine ----
                unsafe {
                    let re = self.row_edges.as_ptr();
                    let rp = self.row_ptr.as_ptr();
                    let b = self.b2c.as_ptr();
                    let c = self.c2b.as_mut_ptr();
                    let cs = self.cand_syn.as_mut_ptr();
                    let syn = syndrome.as_ptr();
                    for j in 0..check_count {
                        *cs.add(j) = 0;
                        let lo = *rp.add(j) as usize;
                        let hi = *rp.add(j + 1) as usize;
                        let mut total_sgn: u32 = *syn.add(j) as u32;
                        let mut temp = inf;
                        for k in lo..hi {
                            let e = *re.add(k) as usize;
                            let v = *b.add(e);
                            total_sgn += (v <= zero) as u32;
                            *c.add(e) = temp;
                            let a = v.abs();
                            if a < temp {
                                temp = a;
                            }
                        }
                        temp = inf;
                        let mut k = hi;
                        while k > lo {
                            k -= 1;
                            let e = *re.add(k) as usize;
                            let v = *b.add(e);
                            let sgn = total_sgn + (v <= zero) as u32;
                            let mut cv = *c.add(e);
                            if temp < cv {
                                cv = temp;
                            }
                            *c.add(e) = if sgn & 1 == 0 { cv } else { -cv };
                            let a = v.abs();
                            if a < temp {
                                temp = a;
                            }
                        }
                    }
                }
                // ---- bit side, fused pass per bit ----
                unsafe {
                    let cp = self.col_ptr.as_ptr();
                    let er = self.edge_row.as_ptr();
                    let b = self.b2c.as_mut_ptr();
                    let c = self.c2b.as_ptr();
                    let cs = self.cand_syn.as_mut_ptr();
                    let init = self.init_llr.as_ptr();
                    let off = self.offsets.as_ptr();
                    let post = self.post.as_mut_ptr();
                    let rmin = self.run_min.as_mut_ptr();
                    let rmax = self.run_max.as_mut_ptr();
                    let sums = self.llr_sums.as_mut_ptr();
                    let dec = self.decoding.as_mut_ptr();
                    for i in 0..bit_count {
                        let lo = *cp.add(i) as usize;
                        let hi = *cp.add(i + 1) as usize;
                        let mut temp = *init.add(i) + *off.add(i);
                        for k in lo..hi {
                            *b.add(k) = temp;
                            temp = temp + *c.add(k);
                        }
                        let p = temp;
                        *post.add(i) = p;
                        let mut mn = *rmin.add(i);
                        let mut mx = *rmax.add(i);
                        temp = zero;
                        let mut k = hi;
                        while k > lo {
                            k -= 1;
                            let m = *b.add(k) + temp;
                            *b.add(k) = m;
                            temp = temp + *c.add(k);
                            if m < mn {
                                mn = m;
                            }
                            if m > mx {
                                mx = m;
                            }
                        }
                        if p <= zero {
                            *dec.add(i) = 1;
                            for k in lo..hi {
                                *cs.add(*er.add(k) as usize) ^= 1;
                            }
                        } else {
                            *dec.add(i) = 0;
                        }
                        *sums.add(i) = *sums.add(i) + p;
                        if p < mn {
                            mn = p;
                        }
                        if p > mx {
                            mx = p;
                        }
                        *rmin.add(i) = mn;
                        *rmax.add(i) = mx;
                    }
                }
                self.iterations += 1;
                if self.cand_syn == syndrome {
                    self.converged = true;
                    return true;
                }
                false
            }

            /// Tilt for the round that follows `completed` completed tilt rounds: the schedule block containing
            /// `completed` in three-phase mode, the scalar parameters otherwise.
            fn compute_offsets(&mut self, completed: usize) {
                if !self.sched_ppos.is_empty() {
                    let mut k = 0usize;
                    while k < self.sched_ends.len() && completed >= self.sched_ends[k] {
                        k += 1;
                    }
                    let (pp, pos, neg, thr) = (self.sched_ppos[k], self.sched_pos[k], self.sched_neg[k], self.sched_thr[k]);
                    self.compute_offsets_with(pp, pos, neg, thr);
                } else {
                    let p = self.params;
                    self.compute_offsets_with(p.prob_positive, p.pos_coef, p.neg_coef, p.threshold_factor);
                }
            }

            fn compute_offsets_with(&mut self, p_pos: f64, pos_coef: f64, neg_coef: f64, thr_factor: f64) {
                let thr = (thr_factor as $t);
                let neg = (neg_coef as $t);
                let pos = (pos_coef as $t);
                let zero = (0.0 as $t);
                self.stat_tilts += 1;
                self.stat_bit_rounds += self.bit_count as u64;
                for i in 0..self.bit_count {
                    let init = self.init_llr[i];
                    if self.llr_sums[i] < thr * init {
                        self.stat_uncertain += 1;
                        let u: f64 = self.rng.gen();
                        self.offsets[i] = if u < p_pos {
                            pos * (-self.run_min[i]).max(zero)
                        } else {
                            -(neg * self.run_max[i].max(zero))
                        };
                    } else {
                        self.offsets[i] = zero;
                    }
                }
            }

            fn decode(&mut self, syndrome: &[u8]) -> &[u8] {
                self.converged = false;
                self.iterations = 0;
                self.tilt_rounds = 0;
                let p = self.params;
                for i in 0..self.bit_count {
                    let init = self.init_llr[i];
                    for k in self.col_ptr[i] as usize..self.col_ptr[i + 1] as usize {
                        self.b2c[k] = init;
                    }
                    self.offsets[i] = (0.0 as $t);
                }
                self.reset_round_stats();
                // ---------------- phase 1: standard BP ----------------
                for it in 1..=p.initial_iters {
                    if it + p.round_len == p.initial_iters + 1 {
                        self.reset_round_stats();
                    }
                    if self.bp_iteration(syndrome) {
                        return &self.decoding;
                    }
                }
                let mut completed = 0usize;
                self.compute_offsets(completed);
                // ---------------- phase 2: rounds with prior offsets ----------------
                for _round in 0..p.max_rounds {
                    completed += 1;
                    self.tilt_rounds = completed;
                    self.reset_round_stats();
                    for _k in 1..=p.round_len {
                        if self.bp_iteration(syndrome) {
                            return &self.decoding;
                        }
                    }
                    self.compute_offsets(completed);
                }
                &self.decoding
            }
        }
    };
}

make_core!(CoreF64, f64);
make_core!(CoreF32, f32);

macro_rules! make_pyclass {
    ($name:ident, $core:ident, $t:ty, $doc:expr) => {
        #[doc = $doc]
        #[pyclass]
        pub struct $name {
            core: $core,
        }

        #[pymethods]
        impl $name {
            #[new]
            #[pyo3(signature = (check_matrix, error_priors, initial_iters=60, round_len=6, max_rounds=6000,
                                threshold_factor=2.0, neg_coef=0.8, pos_coef=0.5, prob_positive=0.1, seed=-1))]
            #[allow(clippy::too_many_arguments)]
            pub fn new(
                check_matrix: &Bound<'_, PyAny>,
                error_priors: PyReadonlyArray1<'_, f64>,
                initial_iters: usize,
                round_len: usize,
                max_rounds: usize,
                threshold_factor: f64,
                neg_coef: f64,
                pos_coef: f64,
                prob_positive: f64,
                seed: i64,
            ) -> PyResult<Self> {
                let params = Params { initial_iters, round_len, max_rounds, threshold_factor, neg_coef, pos_coef, prob_positive };
                Ok(Self { core: $core::new(check_matrix, error_priors.as_slice()?, params, seed)? })
            }

            #[getter] fn check_count(&self) -> usize { self.core.check_count }
            #[getter] fn bit_count(&self) -> usize { self.core.bit_count }
            /// BP iterations of the last decode.
            #[getter] fn iterations(&self) -> usize { self.core.iterations }
            /// Tilt rounds completed in the last decode.
            #[getter] fn tilt_rounds(&self) -> usize { self.core.tilt_rounds }
            /// Whether the last decode reached a syndrome-consistent solution.
            #[getter] fn converge(&self) -> bool { self.core.converged }
            /// Diagnostic counters accumulated since the last reset: (number of (bit, tilt round) pairs that received
            /// a nonzero prior offset, number of (bit, tilt round) pairs examined, number of tilt rounds). Their ratio
            /// is the fraction of bits with a nonzero offset per tilt round.
            fn tilt_statistics(&self) -> (u64, u64, u64) { (self.core.stat_uncertain, self.core.stat_bit_rounds, self.core.stat_tilts) }
            fn reset_tilt_statistics(&mut self) { self.core.stat_uncertain = 0; self.core.stat_bit_rounds = 0; self.core.stat_tilts = 0; }
            #[getter] fn seed(&self) -> i64 { self.core.seed }
            #[setter] fn set_seed(&mut self, v: i64) { self.core.reseed(v) }
            #[getter] fn initial_iters(&self) -> usize { self.core.params.initial_iters }
            #[setter] fn set_initial_iters(&mut self, v: usize) { self.core.params.initial_iters = v }
            #[getter] fn round_len(&self) -> usize { self.core.params.round_len }
            #[setter] fn set_round_len(&mut self, v: usize) { self.core.params.round_len = v }
            #[getter] fn max_rounds(&self) -> usize { self.core.params.max_rounds }
            #[setter] fn set_max_rounds(&mut self, v: usize) { self.core.params.max_rounds = v }
            #[getter] fn threshold_factor(&self) -> f64 { self.core.params.threshold_factor }
            #[setter] fn set_threshold_factor(&mut self, v: f64) { self.core.params.threshold_factor = v }
            #[getter] fn neg_coef(&self) -> f64 { self.core.params.neg_coef }
            #[setter] fn set_neg_coef(&mut self, v: f64) { self.core.params.neg_coef = v }
            #[getter] fn pos_coef(&self) -> f64 { self.core.params.pos_coef }
            #[setter] fn set_pos_coef(&mut self, v: f64) { self.core.params.pos_coef = v }
            #[getter] fn prob_positive(&self) -> f64 { self.core.params.prob_positive }
            #[setter] fn set_prob_positive(&mut self, v: f64) { self.core.params.prob_positive = v }

            /// Posterior log-likelihood ratios after the last decode (as float64).
            fn log_prob_ratios<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
                PyArray1::from_vec(py, self.core.post.iter().map(|&v| v as f64).collect())
            }

            /// Decode one syndrome (uint8, length check_count); returns the error estimate (uint8, bit_count).
            fn decode<'py>(&mut self, py: Python<'py>, detectors: PyReadonlyArray1<'_, u8>) -> PyResult<Bound<'py, PyArray1<u8>>> {
                let syn = detectors.as_slice()?;
                if syn.len() != self.core.check_count {
                    return Err(PyValueError::new_err(format!(
                        "syndrome has length {} but the check matrix has {} rows", syn.len(), self.core.check_count)));
                }
                let out = self.core.decode(syn).to_vec();
                Ok(PyArray1::from_vec(py, out))
            }

            /// Decode a batch (shots x check_count, uint8); returns shots x bit_count (uint8).
            fn decode_batch<'py>(&mut self, py: Python<'py>, detectors: PyReadonlyArray2<'_, u8>) -> PyResult<Bound<'py, PyArray2<u8>>> {
                let shape = detectors.shape();
                let (shots, m) = (shape[0], shape[1]);
                if m != self.core.check_count {
                    return Err(PyValueError::new_err("syndrome width does not match the check matrix"));
                }
                let arr = detectors.as_array();
                let n = self.core.bit_count;
                let mut flat = Vec::with_capacity(shots * n);
                for k in 0..shots {
                    let syn: Vec<u8> = arr.row(k).iter().copied().collect();
                    flat.extend_from_slice(self.core.decode(&syn));
                }
                Ok(PyArray1::from_vec(py, flat).reshape([shots, n])?)
            }

            /// Block schedule of tilt parameters (three-phase mode): `ends` = strictly increasing positive block end
            /// rounds (completed tilt rounds); value k of each parameter applies to the tilts computed after a number of
            /// completed rounds in [ends[k-1], ends[k]), the last value beyond the last end (len = len(ends) + 1).
            /// Overrides the scalar parameters. clear_round_schedule() returns to the uniform mode.
            fn set_round_schedule(&mut self, ends: PyReadonlyArray1<'_, i64>, prob_positive: PyReadonlyArray1<'_, f64>, pos_coef: PyReadonlyArray1<'_, f64>, neg_coef: PyReadonlyArray1<'_, f64>, threshold_factor: PyReadonlyArray1<'_, f64>) -> PyResult<()> {
                let e = ends.as_slice()?;
                let (a, b, c, d) = (prob_positive.as_slice()?, pos_coef.as_slice()?, neg_coef.as_slice()?, threshold_factor.as_slice()?);
                let nb = e.len() + 1;
                if a.len() != nb || b.len() != nb || c.len() != nb || d.len() != nb {
                    return Err(PyValueError::new_err("schedule needs len(ends) + 1 parameter tuples"));
                }
                if e.windows(2).any(|w| w[1] <= w[0]) || e.iter().any(|&v| v <= 0) {
                    return Err(PyValueError::new_err("schedule ends must be positive and strictly increasing"));
                }
                self.core.sched_ends = e.iter().map(|&v| v as usize).collect();
                self.core.sched_ppos = a.to_vec();
                self.core.sched_pos = b.to_vec();
                self.core.sched_neg = c.to_vec();
                self.core.sched_thr = d.to_vec();
                Ok(())
            }
            fn clear_round_schedule(&mut self) {
                self.core.sched_ends.clear();
                self.core.sched_ppos.clear();
                self.core.sched_pos.clear();
                self.core.sched_neg.clear();
                self.core.sched_thr.clear();
            }
            #[getter] fn round_schedule_set(&self) -> bool { !self.core.sched_ppos.is_empty() }
        }
    };
}

make_pyclass!(PriorTiltDecoder, CoreF64, f64, "PriorTilt decoder with f64 messages (reference implementation).");
make_pyclass!(PriorTiltDecoderF32, CoreF32, f32, "PriorTilt decoder with f32 messages (faster; results agree with f64 only statistically).");

#[pymodule]
fn priortilt_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PriorTiltDecoder>()?;
    m.add_class::<PriorTiltDecoderF32>()?;
    Ok(())
}
