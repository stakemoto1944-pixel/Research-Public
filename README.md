# Satoshi Takemoto — Public Research Archive

Research publications and supplementary documents by **Satoshi Takemoto (竹本 聡)**, Independent Researcher, Tokyo, Japan.

- **ORCID**: [0009-0005-6401-3534](https://orcid.org/0009-0005-6401-3534)
- **Zenodo (publications)**: [search results on Zenodo](https://zenodo.org/search?q=creators.name:%22Takemoto,%20Satoshi%22)
- **Contact**: s.takemoto.1944@gmail.com

> **Status as of 2026-11-14**: the data-driven identification manuscript is **not
> under review at any journal**. It was **desk-rejected by *JSTAT*** on 2026-11-14
> (`JSTAT_044P_1026`); the editors stated that manuscripts declared "not suitable"
> "have not been judged on their scientific value", the decision concerning subject
> and emphasis rather than the results. Earlier submissions to *Physical Review E*
> and *Physica A* were also desk-rejected. **No version has reached external peer
> review, and no referee report exists for any of them.** The next venue has not been
> chosen. The full record is in [Submission history](#submission-history).

> **Erratum (2026-11-18).** The 30-component SPDE integrator used for the numerical
> results in all earlier versions of the manuscript and in Zenodo v4.0 applied the
> time step to the noise twice (`psi += (-Q + sqrt(2 D dt) xi) dt`), so the effective
> diffusion constant was `D*dt` instead of `D`. The integrator has been corrected to
> standard Euler–Maruyama and all 30-component results have been recomputed; see
> [`コード/README.md`](コード/README.md) §8. The previous results are kept in
> `コード/legacy_noise_v4.0/`. Results that do not use this integrator (positive
> control, theory verification, estimator calibration on synthetic densities, FWER
> control) are unchanged.

---

## 1. Main Papers (Zenodo preprints)

Works are listed by **title and DOI**. An earlier version of this file used
"Paper (1)/(2)/(3)" numbering, which collided with the numbering used inside the
manuscripts themselves; the numbering has been removed to avoid the ambiguity.

| Title | Date | DOI | File |
| :--- | :--- | :--- | :--- |
| *Systematization of the Nonequilibrium Integrated Phase (NEIP)* | 2026-08-25 | [10.5281/zenodo.22088694](https://doi.org/10.5281/zenodo.22088694) | [`論文/Systematization of the Nonequilibrium Integrated Phase (NEIP.pdf`](論文/Systematization%20of%20the%20Nonequilibrium%20Integrated%20Phase%20(NEIP.pdf) |
| *Emergence of Macroscopic Coherence based on Extended Integrated Field Equation and Predictive Error Minimization in Complex Vector Fields* | 2026-08-28 | [10.5281/zenodo.22138906](https://doi.org/10.5281/zenodo.22138906) | [`論文/Emergence of Macroscopic Coherence...pdf`](論文/Emergence%20of%20Macroscopic%20Coherence%20based%20on%20Extended%20Integrated%20Field%20Equation%20and%20Predictive%20Error%20Minimization%20in%20Complex%20Vector%20Fields.pdf) |
| *Systematization of the Nonequilibrium Integrated Phase (NEIP) in Complex Vector Fields: Foundational Equations, NEIC Criteria, Topological Solenoidal Driving, and Testable Theoretical Predictions* | 2026-09-10 | [10.5281/zenodo.22682953](https://doi.org/10.5281/zenodo.22682953) | [`論文/Systematization ... in Complex Vector Fields.pdf`](論文/Systematization%20of%20the%20Nonequilibrium%20Integrated%20Phase%20(NEIP)%20in%20Complex%20Vector%20Fields.pdf) |

### Manuscript

**Data-Driven Identification of Non-equilibrium Integrated Phases: A Three-Stage
Protocol and a Refined Information-Geometric Criterion** — the current text is the
version submitted to the *Journal of Statistical Mechanics: Theory and Experiment*
(JSTAT, IOP × SISSA) on **2026-10-23**, manuscript number **`JSTAT_044P_1026`**.
JSTAT **desk-rejected** it on **2026-11-14** without external review, and the
manuscript is not currently under review anywhere. There is no article DOI yet;
none will exist until a journal accepts the paper.

The work was previously submitted under a longer title, *Non-equilibrium Integrated
Phases (NEIP): A Data-Driven Identification Protocol — Identifiability, Geometric
Singularities, and the Resolution of the Computational Bottleneck*. That is the
title of the Zenodo preprint and of the PRE-submission PDF retained in this
repository, so earlier citations to that title refer to the same paper.

#### Submission history

| Submitted | Journal | Manuscript no. | Outcome |
| :--- | :--- | :--- | :--- |
| 2026-10-05 | *Physical Review E* (APS) | `EX12494` | **Desk reject, 2026-10-06** — rejected at editorial screening ("fails to match the minimal threshold for the APS journals"). No individual reason was disclosed; an inquiry received the same standard text. |
| 2026-10-14 | *Physica A* (Elsevier) | `PHYSA-264230` | **Desk reject, 2026-10-16** — scope/fit, before review. |
| 2026-10-23 | *JSTAT* (IOP × SISSA) | `JSTAT_044P_1026` | **Desk reject, 2026-11-14** — declared "not suitable"; the editors stated that such manuscripts "have not been judged on their scientific value" and that the decision reflects the paper's subject and emphasis rather than its results. No referee report. |

Each row is a separate submission of a **revised** manuscript, not the same file
resent: the Physica A version was retitled and reframed for that journal's scope,
and the JSTAT version is a re-titled revision with an expanded introduction. The
reported numbers are identical in all three. All three are listed because their
outcome is part of the paper's history; **none is pending, and none produced a
referee report.**

- **Code and data DOI**: [10.5281/zenodo.23050818](https://doi.org/10.5281/zenodo.23050818)
  (concept DOI — always resolves to the latest version; CC BY 4.0. Current latest
  is **v5.0** = [10.5281/zenodo.24361781](https://doi.org/10.5281/zenodo.24361781),
  published 2026-11-19, which bundles the standard-noise code archive **and** the
  revised manuscript PDF. v4.0 = [10.5281/zenodo.23650287](https://doi.org/10.5281/zenodo.23650287)
  (2026-10-26) holds the JSTAT-submission PDF and the earlier integrator. v3.0 = [10.5281/zenodo.23135347](https://doi.org/10.5281/zenodo.23135347)
  retains the PRE-submission PDF)

| Version | Date | File |
| :--- | :--- | :--- |
| First draft (superseded) | 2026-09-19 | [`論文/NEIP_20260917zenodo_new.pdf`](論文/NEIP_20260917zenodo_new.pdf) |
| Revised (superseded) | 2026-09-30 | [`論文/NEIP_data-driven-protocol_20260930_revised.pdf`](論文/NEIP_data-driven-protocol_20260930_revised.pdf) |
| PRE submission version (superseded as a manuscript; retained as the PRE record) | 2026-10-04 | [`論文/NEIP_protocol_manuscript_PRE_submission.pdf`](論文/NEIP_protocol_manuscript_PRE_submission.pdf) |
| JSTAT submission version (superseded: numbers from the earlier integrator; retained as the submission record) | 2026-10-23 | [`論文/NEIP_protocol_manuscript_JSTAT_submission_20261023.pdf`](論文/NEIP_protocol_manuscript_JSTAT_submission_20261023.pdf) |
| **Revised manuscript, standard-noise results (current manuscript)** | 2026-11-19 | [`論文/NEIP_protocol_manuscript_standard_noise_20261119.pdf`](論文/NEIP_protocol_manuscript_standard_noise_20261119.pdf) |

> **Preprint (2026-11-20).** The intermediate report *What does Fisher-metric
> degeneracy measure as a detector of nonequilibrium phases? A test on a
> 30-component system* is deposited on Zenodo as a **Preprint** (not peer
> reviewed, not under consideration at any journal):
> [10.5281/zenodo.24382246](https://doi.org/10.5281/zenodo.24382246). It states
> what the Fisher-metric criterion can and cannot detect; the code and data
> behind it are in the software record above.

> **The 2026-11-19 version is the current manuscript and supersedes all earlier
> files.** It reports the 30-component results recomputed with the standard
> Euler–Maruyama integrator (see the erratum above). The 2026-10-23 PDF is the
> version submitted to JSTAT; its numbers (identical to the 2026-10-04 version) were
> produced by the earlier integrator, which the code deposited in v4.0 reproduces.
> The corrected numbers are in the current `コード/`.

> **The 2026-10-04 PRE submission version remains here as a record, not as the
> current text.** It is the file submitted to *Physical Review E* (`EX12494`) and
> the exact source-compiled PDF deposited in v3.0 of the concept DOI above (md5
> `7069079fc438bf31f4b1dd2872884e63`; the file in this repository is byte-identical
> to the Zenodo copy). v4.0 replaced the bundled manuscript PDF with the JSTAT
> version; v3.0 keeps this file. It had superseded the 2026-09-30 revision, which corrected
> several quantitative claims and replaced the original theoretical prediction; the
> 2026-10-04 file reformatted the manuscript to journal style (single unstructured
> abstract, PACS and keywords, journal-style reference list via BibTeX). None of
> these changes alter the reported numbers, so the code deposited in v4.0
> reproduces every version (with the integrator caveat in the erratum above). The corrections are listed in [`コード/README.md`](コード/README.md) §6.
> The superseded files are retained only as a record; they are **not** the citable
> versions.

**Research themes**: Nonequilibrium Integrated Phase (NEIP) / Consciousness Critical Field Theory (CCFT) / Nonequilibrium Statistical Field Theory of Mind (NCSFT)

---

## 2. Explanation & Theory Slides (`論文解説/`)

Slide-style explanations and theory documents:

| File | Content |
| :--- | :--- |
| `NEIP_1.pdf` | NEIP basics & NEIC 4 requirements (Zenodo 22088694) |
| `NEEP_2.pdf` | Extended Integrated Field Equation & macroscopic coherence (Zenodo 22138906) |
| `NEIP_3.pdf` | Data-driven identification, 3-stage protocol (slides based on the JSTAT submission version; numbers from the earlier integrator, not updated) |
| `NEIP_Systematization.pdf` | NEIP systematization overview (Zenodo 22682953) |
| `統合意識理論１.pdf` | Consciousness as a nonequilibrium critical predictive information field (Ver.1.1.0) |
| `精神の非平衡統計場理論.pdf` | Spin-glass hierarchical self, quantum criticality, semantic phase transition (Ver.1.1.0) |
| `意識の基礎理論と脳科学的アプローチ.pdf` | Foundations of consciousness theory & neuroscientific approaches (Ver.1.0.0) |
| `Resonance_Blueprint.pdf` | Resonance structure / field design |

---

## 3. Code & Data (`コード/`)

Reproduction package for the manuscript: the driver scripts that
produced every number in the current manuscript, the raw result tables, and the
verbatim stdout log of each run. The numerical results of the 30-component SPDE
(stages 1–13) were recomputed on 2026-11-18 with a corrected integrator (see the
erratum above). The package of Zenodo v4.0, which reproduces the earlier numbers
(2026-10-04 PRE and 2026-10-23 JSTAT versions), is superseded for these stages.

- **DOI**: [10.5281/zenodo.23050818](https://doi.org/10.5281/zenodo.23050818)
  (concept DOI, CC BY 4.0, deposited as *Software*; always resolves to the latest
  version — currently **v5.0** = [10.5281/zenodo.24361781](https://doi.org/10.5281/zenodo.24361781),
  which bundles the revised standard-noise manuscript PDF; v4.0 =
  `10.5281/zenodo.23650287` holds the JSTAT-submission PDF (earlier integrator); v3.0 = `10.5281/zenodo.23135347`
  keeps the PRE-submission PDF)
- **Entry point**: [`コード/README.md`](コード/README.md) — requirements,
  execution order, expected values, and a list of the corrections included.
- **Contents**: 22 Python scripts, 10 result tables, 8 run logs.
- **Theoretical verification**: `15_theory_verification.py` is shipped *because it
  failed*; `16_theory_verification.py` is the redesigned run, and
  `post16_theory_fit.py` quantifies the fit and the refutation of the superseded
  theory.
- **Verdicts are computed at run time** from the measured values, never hard-coded.

The scripts are ASCII-renamed copies of the author's Japanese-named working files;
the code itself is unchanged and every internal reference was rewritten to match.
The mapping is tabulated in [`コード/README.md`](コード/README.md) §5.

---

## 4. Philosophical & Intellectual Works (Books)

### 「知を得て無知を識る」series (6 volumes, 2023) — `論文/知を得て無知を識る/`

| Vol | Title | File |
| :--- | :--- | :--- |
| 2 | 自然と生命〜自然とは何か？・ヒトとは何か？〜 | `20230122_Ver003_A4_自然と生命.pdf` |
| 3 | 知覚と感覚 | `20230201_Ver003_A4_知覚と感覚.pdf` |
| 4 | 自然の数学的解釈 | `20230313_Ver003_A4_自然の数学的解釈.pdf` |
| 5 | 脳科学と人間科学のあいだ | `20230801_Ver002_脳科学と人間科学のあいだ.pdf` |
| 6 | 生物科学と人間科学のあいだ | `20231004_Ver001_生物科学と人間科学のあいだ.pdf` |

> Note: Volume 1 is not included in this public archive.

### 「身体と脳と意識のあいだ」series (4 volumes) — `論文/身体と脳と意識のあいだ/`

| Vol | Version | Date | File |
| :--- | :--- | :--- | :--- |
| 1 | Ver.1.1.1 | 2024-06-19 | `身体と脳と意識のあいだ.pdf` |
| 2 | Ver.1.0.0 | 2025-01-17 | `身体と脳と意識のあいだ_2.pdf` |
| 3 | Ver.1.0.0 | 2025-03-31 | `身体と脳と意識のあいだ_3.pdf` |
| 4 | Ver.1.0.0 | 2025-09-06 | `身体と脳と意識のあいだ_4.pdf` |

---

## Structure

```
Research-Public/
├── README.md
├── 論文/          # Papers (Zenodo) & books
│   ├── *.pdf
│   ├── 知を得て無知を識る/
│   └── 身体と脳と意識のあいだ/
├── 論文解説/      # Explanation & theory slides
└── コード/        # Code & data for the manuscript
    ├── README.md
    ├── *.py           # 22 driver / post-analysis / figure scripts
    ├── *_result.csv   # 10 raw result tables
    └── logs/          # 8 verbatim run logs
```

## License

- **Papers and books** (`論文/`, `論文解説/`): Copyright © 2026 Satoshi Takemoto.
  All rights reserved.
- **Code and data** (`コード/`): Copyright © 2026 Satoshi Takemoto, released under
  the **CC BY 4.0** license (<https://creativecommons.org/licenses/by/4.0/>).

## Citation

If you use the code or data, please cite the work you actually used and link to
this repository. The manuscript has **no journal DOI yet** — its most recent
submission (to *JSTAT*, `JSTAT_044P_1026`) was desk-rejected on 2026-11-14, and no
venue has been chosen for the next one. The **2026-11-19 revised manuscript**
(standard-noise results) is the current text and supersedes the 2026-10-23 JSTAT
submission version, the 2026-10-04 PRE submission version and the two earlier drafts. The quantitative corrections listed in [`コード/README.md`](コード/README.md)
§6 are not optional — the first draft's numbers and its original theoretical
prediction have both been withdrawn. Cite the concept DOI
`10.5281/zenodo.23050818` so that your reference resolves to the complete
archive; v5.0
([10.5281/zenodo.24361781](https://doi.org/10.5281/zenodo.24361781)) archives the current manuscript and the standard-noise
code, and v4.0
([10.5281/zenodo.23650287](https://doi.org/10.5281/zenodo.23650287)) the 2026-10-23
JSTAT version with the earlier integrator.