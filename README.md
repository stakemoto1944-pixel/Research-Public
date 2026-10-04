# Satoshi Takemoto — Public Research Archive

Research publications and supplementary documents by **Satoshi Takemoto (竹本 聡)**, Independent Researcher, Tokyo, Japan.

- **ORCID**: [0009-0005-6401-3534](https://orcid.org/0009-0005-6401-3534)
- **Zenodo (publications)**: [search results on Zenodo](https://zenodo.org/search?q=creators.name:%22Takemoto,%20Satoshi%22)
- **Contact**: s.takemoto.1944@gmail.com

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

### Manuscript under review

**Non-equilibrium Integrated Phases (NEIP): A Data-Driven Identification
Protocol — Identifiability, Geometric Singularities, and the Resolution of the
Computational Bottleneck**. In preparation for submission to *Physical Review E*.

- **Code and data DOI**: [10.5281/zenodo.23050818](https://doi.org/10.5281/zenodo.23050818)
  (concept DOI — always resolves to the latest version; CC BY 4.0. Current latest
  is v3.0 = [10.5281/zenodo.23135347](https://doi.org/10.5281/zenodo.23135347),
  which bundles the code archive **and** the manuscript PDF)

| Version | Date | File |
| :--- | :--- | :--- |
| First draft (superseded) | 2026-09-19 | [`論文/NEIP_20260917zenodo_new.pdf`](論文/NEIP_20260917zenodo_new.pdf) |
| Revised (superseded) | 2026-09-30 | [`論文/NEIP_data-driven-protocol_20260930_revised.pdf`](論文/NEIP_data-driven-protocol_20260930_revised.pdf) |
| **PRE submission version (current)** | 2026-10-04 | [`論文/NEIP_protocol_manuscript_PRE_submission.pdf`](論文/NEIP_protocol_manuscript_PRE_submission.pdf) |

> **The 2026-10-04 version supersedes both earlier files and is the one to cite.**
> It is the exact source-compiled PDF deposited as v3.0 of the concept DOI above
> (md5 `7069079fc438bf31f4b1dd2872884e63`; the file in this repository is
> byte-identical to the Zenodo copy). The 2026-09-30 revision corrected several
> quantitative claims and replaced the original theoretical prediction; the
> 2026-10-04 version reformats the manuscript to the journal style (single
> unstructured abstract, PACS and keywords, journal-style reference list via
> BibTeX). None of these changes alter the reported numbers, so the deposited
> code reproduces every version. The corrections are listed in
> [`コード/README.md`](コード/README.md) §6. The superseded files are retained
> only as a record; they are **not** the citable versions.

**Research themes**: Nonequilibrium Integrated Phase (NEIP) / Consciousness Critical Field Theory (CCFT) / Nonequilibrium Statistical Field Theory of Mind (NCSFT)

---

## 2. Explanation & Theory Slides (`論文解説/`)

Slide-style explanations and theory documents:

| File | Content |
| :--- | :--- |
| `NEIP_1.pdf` | NEIP basics & NEIC 4 requirements (Zenodo 22088694) |
| `NEEP_2.pdf` | Extended Integrated Field Equation & macroscopic coherence (Zenodo 22138906) |
| `NEIP_3.pdf` | Data-driven identification, 3-stage protocol (manuscript under review) |
| `NEIP_Systematization.pdf` | NEIP systematization overview (Zenodo 22682953) |
| `統合意識理論１.pdf` | Consciousness as a nonequilibrium critical predictive information field (Ver.1.1.0) |
| `精神の非平衡統計場理論.pdf` | Spin-glass hierarchical self, quantum criticality, semantic phase transition (Ver.1.1.0) |
| `意識の基礎理論と脳科学的アプローチ.pdf` | Foundations of consciousness theory & neuroscientific approaches (Ver.1.0.0) |
| `Resonance_Blueprint.pdf` | Resonance structure / field design |

---

## 3. Code & Data (`コード/`)

Reproduction package for the manuscript under review: the driver scripts that
produced every number, the raw result tables, and the verbatim stdout log of each
production run.

- **DOI**: [10.5281/zenodo.23050818](https://doi.org/10.5281/zenodo.23050818)
  (concept DOI, CC BY 4.0, deposited as *Software*; always resolves to the latest
  version — currently v3.0 = [10.5281/zenodo.23135347](https://doi.org/10.5281/zenodo.23135347))
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
└── コード/        # Code & data for the manuscript under review
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
this repository. Note that the **2026-10-04 PRE submission version** of the
manuscript supersedes the 2026-09-30 revision and the 2026-09-19 draft, and that
the quantitative corrections listed in [`コード/README.md`](コード/README.md) §6
are not optional — the first draft's numbers and its original theoretical
prediction have both been withdrawn. Cite the concept DOI
`10.5281/zenodo.23050818` so that your reference resolves to the complete
archive.