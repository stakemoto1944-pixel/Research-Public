# Code & Data — Data-Driven Identification Protocol for NEIP

Reproduction package for the manuscript
*Non-equilibrium Integrated Phases (NEIP): A Data-Driven Identification Protocol —
Identifiability, Geometric Singularities, and the Resolution of the Computational Bottleneck*.

Everything in this directory is the **actual code that produced the numbers in the
manuscript**, together with the raw result tables and the complete stdout logs of
every run. No numbers have been re-entered by hand: the result tables are written by
the scripts, and the logs are the verbatim console output of the production runs.

> **A note on naming.** The author's working scripts carry Japanese filenames. For
> this public archive they have been renamed to ASCII, **without changing any
> code**, and every internal reference (dynamic module loads, result-table reads,
> log paths) has been rewritten to match. The Japanese names are listed in the
> mapping table below so the archive can be matched against the author's private
> working notes.

---

## 1. What is here

| Category | Count | Files |
| :--- | :--- | :--- |
| Driver scripts | 16 | `experiment1_fixed.py` … `experiment14_fixed_positive_control.py`, `15_theory_verification.py`, `16_theory_verification.py` |
| Post-analysis | 1 | `post16_theory_fit.py` |
| Figure generation | 2 | `figure_maker.py`, `figure_data_check.py` |
| Diagnostics / pilots | 3 | `diag_exponent.py`, `pilot_floor.py`, `pilot2_harmonic.py` |
| Result tables | 10 | `*_result.csv` |
| Run logs | 8 | `logs/experiment*_log.txt`, `logs/post16_log.txt` |

## 2. Requirements

```
python >= 3.11
numpy
scipy
matplotlib
```

No compiled extensions and no external data are needed; every run is seeded and
self-contained.

## 3. How to reproduce the pipeline

The stages are numbered in the order they were executed. Stages 9–16 are the
**robustness and theory-verification** stages: they are not needed to obtain the
main result, but every corrected claim in the manuscript rests on them, and they
are the evidence for the corrections listed in §6.

```bash
# Stage 1-2  parameter scan and Phase-1 screening definition
python3 experiment1_fixed.py
python3 experiment2_fixed_param_search.py

# Stage 3  intervention / noise sweeps (Phase 2)
python3 experiment3_fixed_intervention_noise.py

# Stage 4  core numerics: SPDE integrator + KDE-Fisher geometry
#          (this module provides `simulate` and `compute_geometry`, which the
#           later stages import dynamically)
python3 experiment4_fixed_info_geometry.py

# Stage 5-8  boundary sweep, k-NN cross-check, PCA submanifolds, axes/phase
python3 experiment5_fixed_boundary_sweep.py
python3 experiment6_fixed_knn_crosscheck.py
python3 experiment7_fixed_highdim_metric.py
python3 experiment8_fixed_axes_phase.py

# Stage 9   FWER control for the multiplicity in Phase 1
python3 experiment9_fixed_fwer_control.py

# Stage 10  estimator calibration on synthetic densities
python3 experiment10_fixed_estimator_calibration.py

# Stage 11  bootstrap confidence intervals (B >= 1000, moving block)
python3 experiment11_fixed_bootstrap_ci.py

# Stage 12  moving-block length sensitivity
python3 experiment12_fixed_blocklen_sensitivity.py

# Stage 13  scaling benchmark (reused by stage 14)
python3 experiment13_fixed_scaling_benchmark.py

# Stage 14  positive control on a testbed with a known winding number
python3 experiment14_fixed_positive_control.py

# Stage 15/16  theory verification (negative run, then redesigned run)
python3 15_theory_verification.py      # negative result: no verdict reached
python3 16_theory_verification.py      # redesigned estimator
python3 post16_theory_fit.py           # quantified fit and refutation

# Figures and the cross-check that they agree with the result tables
python3 figure_maker.py
python3 figure_data_check.py
```

`experiment10`, `11`, `12`, `13`, and `14` load `experiment4` (and `experiment13`)
at run time from the same directory, so keep the files together and run them from
this directory.

### Fastest useful checks

If you only want to confirm that the corrected claims reproduce, run:

```bash
python3 post16_theory_fit.py                    # theory verification, seconds
python3 experiment9_fixed_fwer_control.py       # multiplicity correction
python3 experiment11_fixed_bootstrap_ci.py --quick
```

`post16_theory_fit.py` reads the shipped `theory_verification16_result.csv`, so it
reproduces the reported fit and the refutation of the superseded theory **without
re-running the 276 s simulation**.

## 4. Expected output

| Quantity | Expected value | Where |
| :--- | :--- | :--- |
| ROI operating point | `lambda_min = 0.207` (seed-paired ROI/matched-Gaussian `= 1.40`) | `highdim_metric_result.csv`, `logs/experiment*_log.txt` |
| FWER, raw vs corrected | `0.9275 -> 0.0225` (nominal 0.025) | `fwer_control_result.csv`, `logs/experiment09_log.txt` |
| Bootstrap CI at ROI | `0.1994 +/- 0.0304`, CI width `0.037`-`0.082`, 20/20 seeds exclude zero | `bootstrap_ci_result.csv` |
| Positive control | winding number recovered to `< 0.01 %` relative error, `30/30` detections | `positive_control_result.csv`, `logs/experiment14_log.txt` |
| Theory (P1c) | measured/predicted `0.894`-`1.118` over the resolvable range | `theory_verification16_result.csv`, `logs/post16_log.txt` |
| Theory (R1) | fixed arm `z >= 5.3`, co-rotating arm `abs(z) <= 0.60` | same |
| Theory (W1) | width exponent `-2.0000` | same |
| Theory (F1) | noise floor exponent `-1.425` (partial confirmation only) | same |

The `logs/` directory holds the verbatim output of the runs that produced these
numbers. `post16_log.txt` ends with the summary verdict block.

## 5. Filename mapping (Japanese working names -> public names)

| Working name | Public name |
| :--- | :--- |
| `数値実験1_修正版.py` | `experiment1_fixed.py` |
| `数値実験2_修正版_パラメータ探索.py` | `experiment2_fixed_param_search.py` |
| `数値実験3_修正版_介入ノイズ掃引.py` | `experiment3_fixed_intervention_noise.py` |
| `数値実験4_修正版_情報幾何検証.py` | `experiment4_fixed_info_geometry.py` |
| `数値実験5_修正版_相境界掃引.py` | `experiment5_fixed_boundary_sweep.py` |
| `数値実験6_修正版_kNN交差検証.py` | `experiment6_fixed_knn_crosscheck.py` |
| `数値実験7_修正版_高次元計量検証.py` | `experiment7_fixed_highdim_metric.py` |
| `数値実験8_修正版_境界軸と位相変数.py` | `experiment8_fixed_axes_phase.py` |
| `数値実験9_修正版_FWER制御.py` | `experiment9_fixed_fwer_control.py` |
| `数値実験10_修正版_推定器校正.py` | `experiment10_fixed_estimator_calibration.py` |
| `数値実験11_修正版_ブートストラップCI.py` | `experiment11_fixed_bootstrap_ci.py` |
| `数値実験12_修正版_ブロック長感度_n5000.py` | `experiment12_fixed_blocklen_sensitivity.py` |
| `数値実験13_修正版_スケーリングbenchmark.py` | `experiment13_fixed_scaling_benchmark.py` |
| `数値実験14_修正版_正の対照.py` | `experiment14_fixed_positive_control.py` |
| `数値実験15_修正版_理論検証.py` | `15_theory_verification.py` |
| `数値実験16_修正版_理論検証_再設計.py` | `16_theory_verification.py` |
| `post16.py` | `post16_theory_fit.py` |
| `図版作成.py` | `figure_maker.py` |
| `図版データ整合性チェック.py` | `figure_data_check.py` |
| `相境界掃引_result.csv` | `boundary_sweep_result.csv` |
| `高次元計量_result.csv` | `highdim_metric_result.csv` |
| `境界軸位相_result.csv` | `axis_phase_result.csv` |
| `FWER制御_result.csv` | `fwer_control_result.csv` |
| `推定器校正_result.csv` | `estimator_calibration_result.csv` |
| `ブートストラップCI_result.csv` | `bootstrap_ci_result.csv` |
| `スケーリング_result.csv` | `scaling_result.csv` |
| `正の対照_result.csv` | `positive_control_result.csv` |
| `理論検証_result.csv` | `theory_verification15_result.csv` |
| `理論検証16_result.csv` | `theory_verification16_result.csv` |

## 6. Corrections included in this package

These are not cosmetic. The package documents the corrections applied to the
manuscript after its first submission draft, and it ships the **negative results**
that motivated them.

1. **Confidence intervals were too narrow.** The original intervals used `B = 12`
   with i.i.d. resampling, which both under-resolve the quantile tail and ignore the
   `tau ~ 31` correlation time of the quadrature series. Recomputed with `B = 1000`
   and a moving block of length `L = 64` over 20 seeds, the intervals are `1.6`-`3.5`
   times wider (mean `2.4` times). **No conclusion changes**: zero is excluded in
   `20/20` seeds. -> `experiment11`, `bootstrap_ci_result.csv`
2. **The k-NN estimator cannot serve as a second line of evidence.** Its
   overestimation of `lambda_min` is a *resolution floor*, not a calibration offset:
   `24.4x` at the ROI working point, rising to `231x` as `lambda_min -> 1e-3`. All
   quantitative claims rest on the calibrated KDE. -> `experiment10`,
   `estimator_calibration_result.csv`
3. **Multiplicity in Phase 1 was not controlled.** Raw family-wise error was
   `0.9275`; with a max-statistic correction it is `0.0225` against a nominal `0.025`.
   -> `experiment9`, `fwer_control_result.csv`
4. **The published record-length scaling was wrong.** The endpoint is `0.204`, not
   `0.275`, and the dependence is **non-monotonic**: `0.1691`, `0.1772`, `0.1734`,
   `0.1850`, `0.1974`, `0.2036` for `M = 3000 ... 20000` (total change `+20 %`).
   -> `experiment13`, `scaling_result.csv`
5. **The theory was superseded, and the old form is refuted.** The first draft's
   `g_theta_theta = omega_1^2 / (2 D_theta^2)` turned out to be the `omega_0 -> 0`
   limit of the correct expression
   `g_theta_theta = omega_1^2 / (2 (D_theta^2 + omega_0^2))`. The negative run
   `15_theory_verification.py` is shipped precisely because it **failed to reach a
   verdict** — the estimator noise floor exceeded the signal. -> `15_...`, `16_...`,
   `post16_theory_fit.py`
6. **The metric determinant cannot carry a universal exponent.** `det g` is not a
   coordinate invariant: under `x = x(q)` it transforms as
   `det g_ab = (det J)^2 det g`. Only the null-eigenvalue statement is invariant.

**Scope note.** Items 5 and 6 concern the *theory*, and the quantitative
confirmation of the corrected theory was obtained on a **separate** controlled
system with a uniform rotation rate `omega_0`. Only the *structure* of the
condition (co-rotating drive => degeneracy; lab-frame-fixed drive => non-degeneracy)
transfers to the many-body system studied here; the `omega_0`-dependent scaling has
**not** been verified for that system.

## 7. Reproducibility notes

- All runs are seeded; the `SEEDS` tuples are visible in each driver script.
- `experiment16` takes ~276 s on a laptop-class CPU.
- Verdict strings are **computed from the measured values at run time**, never
  hard-coded, so a changed measurement changes the verdict rather than silently
  contradicting it.
- `figure_data_check.py` verifies that the figures agree with the shipped result
  tables (`ALL PASS` as of 2026-09-16).
- Requirement versions used for the archived runs are recorded in `logs/`.

## License

Copyright (c) 2026 Satoshi Takemoto.
Code and data are released under the **CC BY 4.0** license
(<https://creativecommons.org/licenses/by/4.0/>).

## Archive record

- DOI: [10.5281/zenodo.23050819](https://doi.org/10.5281/zenodo.23050819)
- Concept DOI: `10.5281/zenodo.23050818`
- Deposited 2026-09-30 as *Software*, version 1.0
- Archive file: `NEIP_protocol_code_v1.zip` (164,045 bytes,
  md5 `dc19d89eef8b3d1ffc5c1c75eddb5b91`)
- Mirror: <https://github.com/stakemoto1944-pixel/Research-Public>

To archive a later revision, create a **new version** of the concept DOI rather
than editing record 23050819, so that citations to v1.0 continue to resolve to the
code that produced the numbers they cite.
