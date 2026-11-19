# Code & Data — Data-Driven Identification Protocol for NEIP

> **ERRATUM (2026-11-18) — noise convention of the 30-component SPDE.**
> The SPDE integrator used by stages 1–13 (archived as v4.0) multiplied the noise
> increment by `dt` twice, `psi += (-Q + sqrt(2 D dt) * xi) * dt`. The effective
> diffusion constant was therefore `D*dt = 1.0e-5` (for `D = 0.02`, `dt = 5e-4`)
> instead of `D`, and the simulated system was essentially deterministic. The
> integrator is now the standard Euler–Maruyama step
> `psi += -Q*dt + sqrt(2 D dt) * xi`, and **every 30-component number below has
> been recomputed**. See §8 for what changed, what did not, and how it was found.
> The previous results are kept unchanged in `legacy_noise_v4.0/` and in Zenodo v4.0.

Reproduction package for the manuscript
*Non-equilibrium Integrated Phases (NEIP): A Data-Driven Identification Protocol —
Identifiability, Geometric Singularities, and the Resolution of the Computational Bottleneck*.

> **Status (2026-11-14).** The manuscript is **not under review at any journal**.
> Its most recent submission was to *JSTAT* (`JSTAT_044P_1026`, submitted 2026-10-23)
> under the title *Data-Driven Identification of Non-equilibrium Integrated Phases: A
> Three-Stage Protocol and a Refined Information-Geometric Criterion*; JSTAT
> desk-rejected it on 2026-11-14 without external review. It was previously submitted
> to *Physical Review E* (`EX12494`; desk reject 2026-10-06) and to *Physica A*
> (`PHYSA-264230`; desk reject 2026-10-16). No version has produced a referee report.
> The reported numbers are identical in
> every version from the 2026-09-30 revision onward (the 2026-09-19 first draft's
> numbers were corrected on 2026-09-30); the submission record is in the repository
> [README](../README.md#submission-history). The code and the current manuscript
> are archived as **v5.0** of the concept DOI
> ([10.5281/zenodo.24361781](https://doi.org/10.5281/zenodo.24361781), 2026-11-19); v4.0
> ([10.5281/zenodo.23650287](https://doi.org/10.5281/zenodo.23650287)) holds the earlier
> integrator.

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
| Integrator | 1 | `fastsim.py` (standard Euler–Maruyama; `legacy=True` reproduces the old step) |
| Standard-noise additions | 11 | `standard_noise_additions/` (extra checks, calibration work points, integrator audits) |
| Previous results | — | `legacy_noise_v4.0/` (old CSVs, logs and figure scripts, for comparison only) |
| Result tables | 10 | `*_result.csv` |
| Run logs | 8 | `logs/experiment*_log.txt`, `logs/post16_log.txt` |

## 2. Requirements

```
python >= 3.11
numpy
scipy
matplotlib
numba        # strongly recommended: fastsim.py compiles the SPDE loop
```

No external data are needed; every run is seeded and self-contained. `fastsim.py`
falls back to the same loop in plain Python if `numba` is missing (identical result,
much slower). The SPDE driver scripts (`experiment1`–`9`) call
`fastsim.integrate`, so keep `fastsim.py` next to them.

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

Run lengths: `experiment11_fixed_bootstrap_ci.py` is shipped with the settings that
produced the reported numbers (`B = 500` main, `B = 200` auxiliary, 2 worker
processes, and the S0/S1 point-estimate stages removed because the 20-seed point
estimates come from `standard_noise_additions/pt20.py`). The earlier `B = 1000`
setting is in `legacy_noise_v4.0/`. `experiment12_fixed_blocklen_sensitivity.py`
was **not** re-run; the block-length sensitivity is reported by stage S3 of
`experiment11` (`logs/experiment11_log.txt`). In `experiment13`, part B
(record-length scaling of `lambda_min`) is replaced by
`standard_noise_additions/exp13B_std.py`; its timing parts (FFT cost) do not depend
on the noise convention and were not re-run.

Standard-noise additions (run from `standard_noise_additions/`):

```bash
python3 audit_fastsim.py        # integrator check on a closed-form diffusion: standard 0.998, legacy 2.5e-7
python3 pt20.py                 # 20-seed lambda_min for ROI / EQ / DD / matched Gaussian
python3 pt40.py                 # 40-seed ROI vs DD test
python3 cal_std.py              # estimator calibration at the standard-noise work points
python3 exp13B_std.py           # record-length scaling of lambda_min
python3 off_null.py             # drive-OFF reference ensemble for the Phase-1 statistic
python3 gain_scan.py            # drive-gain scan (chaos vs circulation)
python3 lyap.py                 # largest Lyapunov exponent at D = 0
python3 audit_github_all.py <dir>   # audits the update line of every SPDE script in <dir>
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
| ROI operating point | `lambda_min = 0.1737 +/- 0.0067` (20 seeds, SE); matched Gaussian `0.1294`; EQ `1.063`; DD `6.48` (median `3.37`) | `logs/pt20_lambda_min_log.txt`, `highdim_metric_result.csv` |
| ROI vs DD (40 seeds) | ROI `0.1776 +/- 0.0052`, DD `8.10 +/- 1.33`, Welch `p = 6e-7`, Mann–Whitney `p = 1.4e-14` | `logs/pt40_roi_vs_dd_log.txt`, `pt40_perseed.csv` |
| Dynamics at `D = 0` | largest Lyapunov exponent: ROI `+0.93`, EQ `-0.05`, DD `-0.64` (ROI is deterministic chaos) | `logs/lyapunov_D0_log.txt` |
| FWER, raw vs corrected | `0.9275 -> 0.0225` (nominal 0.025; white-noise null, independent of the integrator) | `fwer_control_result.csv`, `logs/experiment09_log.txt` |
| Drive-OFF reference | `max|I|` of the OFF ensemble exceeds the channel-mean iid threshold at all four grid points; ON/OFF mean ratio `25`–`53` | `OFF_null_result.csv`, `logs/off_null_log.txt` |
| Bootstrap CI at ROI | `B = 500`, `L = 64`: mean CI width `0.047` (`0.031`–`0.061`), 20/20 seeds exclude zero; controls EQ `[0.93, 1.22]`, DD `[3.77, 4.72]`, Gauss `[0.119, 0.144]` | `bootstrap_ci_result.csv`, `logs/experiment11_log.txt` |
| Estimator calibration | KDE `0.942x` of truth at every work point; k-NN `14.8x` at the ROI point (`24.3x` at the old point) | `estimator_calibration_result.csv`, `logs/experiment10_workpoints_log.txt` |
| Record-length scaling | `0.138, 0.159, 0.173, 0.169, 0.177, 0.179` for `M = 3000, 5000, 8000, 12000, 16000, 20000` (6 seeds, non-monotonic) | `logs/experiment13B_scaling_log.txt` |
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
| `数値実験*_修正版*.py` (2026-11-18 standard-noise reruns) | `experiment*_fixed*.py` (same public names, updated in place) |
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
   `tau ~ 31` correlation time of the quadrature series. Recomputed with a moving block of length `L = 64` over 20 seeds (`B = 1000` in
   the archived v4.0 run, `B = 500` in the standard-noise rerun), zero is excluded in
   `20/20` seeds in both. -> `experiment11`, `bootstrap_ci_result.csv`
2. **The k-NN estimator cannot serve as a second line of evidence.** Its
   overestimation of `lambda_min` is a *resolution floor*, not a calibration offset:
   `24.4x` at the old ROI working point (`14.8x` at the standard-noise ROI point),
   rising to `231x` as `lambda_min -> 1e-3`. All
   quantitative claims rest on the calibrated KDE. -> `experiment10`,
   `estimator_calibration_result.csv`
3. **Multiplicity in Phase 1 was not controlled.** Raw family-wise error was
   `0.9275`; with a max-statistic correction it is `0.0225` against a nominal `0.025`.
   -> `experiment9`, `fwer_control_result.csv`
4. **The published record-length scaling was wrong.** The endpoint is not `0.275`;
   with the standard-noise integrator it is `0.179` (6 seeds), and the dependence is
   **non-monotonic**: `0.138`, `0.159`, `0.173`, `0.169`, `0.177`, `0.179` for
   `M = 3000 ... 20000` (total change `+30 %`). (Archived v4.0 values with the
   old integrator: `0.169 ... 0.204`.) -> `standard_noise_additions/exp13B_std.py`
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
- `figure_maker_standard.py` regenerates the figures from the standard-noise tables. The
  old `figure_maker.py` / `figure_data_check.py` (which checked the old tables, `ALL PASS`
  as of 2026-09-16) are in `legacy_noise_v4.0/` and do not apply to the current CSVs.
- Requirement versions used for the archived runs are recorded in `logs/`.

## 8. Erratum: noise convention of the 30-component SPDE (2026-11-18)

**What was wrong.** In `experiment1`–`9` (and everything that loads `experiment4`,
i.e. `10`–`13`) the update was
`psi += (-Q + np.sqrt(2.0*D*DT) * np.random.randn(N)) * DT`. The noise already carries
`sqrt(DT)`; multiplying by `DT` again makes its variance per step `2 D DT^3`. The
effective diffusion constant is `D*DT = 1.0e-5` (a factor `5e-4` below the nominal
`D = 0.02`), and the dynamics of the ROI condition are deterministic chaos. The
comment in the old code ("`dt` explicitly multiplied = regular continuous-time
discretisation") was incorrect.

**How it was found.** A closed-form audit of the update line (`standard_noise_additions/`):
for free diffusion the ratio `Var(psi_T) / (2 D T)` is `1` for a correct integrator.

| Update | Free diffusion ratio | Verdict |
| :--- | :--- | :--- |
| standard EM (`fastsim.integrate`, `legacy=False`) | `0.998` | pass |
| old step (`legacy=True`, and `experiment1`–`9` as archived in v4.0) | `2.5e-7 = dt^2` | fail |
| `experiment14`, `15_`, `16_`, `pilot_*`, `diag_exponent` | standard EM form (`0.996`) | pass |

(`logs/audit_fastsim_log.txt`, `logs/audit_legacy_code_log.txt`.)

**What is not affected.** `experiment14` (positive control), `15_theory_verification`,
`16_theory_verification`, `post16`, the pilots, the estimator calibration on
synthetic densities (`experiment10`, apart from the added work points), and the FWER
control (white-noise null) do not use the 30-component integrator.

**What changed in the 30-component results** (old archive -> standard noise):

| Quantity | Old (v4.0) | Standard noise |
| :--- | :--- | :--- |
| ROI `lambda_min` (20 seeds) | `0.1994` | `0.1737 +/- 0.0067` |
| matched Gaussian (20 seeds) | see `legacy_noise_v4.0/` | `0.1294` |
| DD `lambda_min` (20 seeds) | see `legacy_noise_v4.0/` | `6.48` (median `3.37`) |
| Bootstrap CI | `B=1000`, 20/20 exclude zero | `B=500`, 20/20 exclude zero, mean width `0.047` |
| Record-length endpoint (`M = 20000`) | `0.204` | `0.179` |
| k-NN over-estimate at the ROI point | `24.4x` | `14.8x` |

**What the conclusion is now.** The qualitative picture survives: `lambda_min` stays
positive and finite in every condition, the ROI is separated from the equilibrium and
dissipative controls, and no multi-order gap to the matched Gaussian appears. The
interpretation changes: with `D = 0` the ROI has a largest Lyapunov exponent of
`+0.93`, i.e. the near-deterministic dynamics are chaotic, while EQ and DD relax
(`-0.05`, `-0.64`). The drive gain is the only control parameter that moves the system
between regimes (`standard_noise_additions/gain_scan.py`). The manuscript text has been
rewritten accordingly.

**Provenance.** `legacy_noise_v4.0/` keeps the old CSVs, logs and figure scripts for
comparison; they were produced by the old integrator and should not be cited as
results of the stated model. Zenodo v4.0 contains the old code; this package is archived as v5.0.

## License

Copyright (c) 2026 Satoshi Takemoto.
Code and data are released under the **CC BY 4.0** license
(<https://creativecommons.org/licenses/by/4.0/>).

## Archive record

- DOI (cite this): [10.5281/zenodo.23050818](https://doi.org/10.5281/zenodo.23050818)
  — concept DOI, always resolves to the latest version
- Current latest: **v5.0** = [10.5281/zenodo.24361781](https://doi.org/10.5281/zenodo.24361781),
  published 2026-11-19 as *Software*, CC BY 4.0
- Archive file in v5.0: `NEIP_protocol_code_v5.zip` (742,610 bytes,
  md5 `e937a1c0e0a9933c155da8ae7d87227d`); manuscript
  `NEIP_protocol_manuscript_standard_noise_20261119.pdf`
  (md5 `448f66f9cd9cc155862171ad1c0342c4`), also in the GitHub mirror —
  [`../論文/NEIP_protocol_manuscript_standard_noise_20261119.pdf`](../論文/NEIP_protocol_manuscript_standard_noise_20261119.pdf)
- v4.0 = [10.5281/zenodo.23650287](https://doi.org/10.5281/zenodo.23650287)
  (2026-10-26, *Software*, CC BY 4.0): archive file `NEIP_protocol_code_v1.zip`
  (164,045 bytes, md5 `dc19d89eef8b3d1ffc5c1c75eddb5b91`), identical in v1.0–v4.0
- Manuscript file in v4.0: `NEIP_protocol_manuscript_JSTAT_submission_20261023.pdf`,
  the manuscript submitted to *JSTAT* on 2026-10-23 (`JSTAT_044P_1026`); also in the
  GitHub mirror —
  [`../論文/NEIP_protocol_manuscript_JSTAT_submission_20261023.pdf`](../論文/NEIP_protocol_manuscript_JSTAT_submission_20261023.pdf)
- Manuscript file in v3.0 (kept as the PRE record):
  `NEIP_protocol_manuscript_PRE_submission.pdf`, compiled for submission to
  *Physical Review E* (2026-10-05, `EX12494`). The reported numbers are unchanged
  between the two; see the repository
  [README](../README.md#submission-history) for the full record
- Mirror: <https://github.com/stakemoto1944-pixel/Research-Public>

Version history of the concept DOI:

| Version | Record | Date | Files |
| :--- | :--- | :--- | :--- |
| v1.0 | `…zenodo.23050819` | 2026-09-30 | code archive |
| v2.0 | `…zenodo.23134914` | 2026-10-04 | manuscript PDF only — **code archive missing; do not cite** |
| v3.0 | `…zenodo.23135347` | 2026-10-04 | code archive + PRE-submission manuscript PDF |
| v4.0 | `…zenodo.23650287` | 2026-10-26 | code archive + JSTAT-submission manuscript PDF (earlier integrator) |
| **v5.0 (latest)** | `…zenodo.24361781` | 2026-11-19 | standard-noise code archive + revised manuscript PDF |

To archive a later revision, create a **new version** of the concept DOI rather
than editing an existing record, so that citations continue to resolve to the
code that produced the numbers they cite. Always verify the **file list** before
publishing: the concept DOI points at the latest version, so a version published
without the code archive makes the archive un-reproducible for anyone who
follows the concept DOI (this is exactly what happened in v2.0).

