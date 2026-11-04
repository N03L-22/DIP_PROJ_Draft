---
type: project
title: DIP Course Project
status: todo
due: 2026-11-12
priority: high
---

# DIP Course Project

**Final due:** 2026-11-12 (project presentations) · **30% of the grade**

> A collaborative final project, discussed and evaluated through the semester. Dates from the
> lecture plan in [[_syllabus]] — confirm details in class.

## Goal
**Topic (proposal v2):** *Restoring Visibility Through Haze, Water, Glass and Plastic* — which classical,
non-learning methods restore visibility best in each medium, why, and how far a synthetic degradation can
stand in for a real capture. Dark Channel Prior (DCP) is the baseline. **"Done" =** real captures, DCP plus
2 to 3 classical alternatives, matched synthetic data, and the real-vs-synthetic ranking comparison, with a
recommended pipeline per medium. Stages: [[Subjects/DIP/Projects/DIP-Project/docs/stages|stages]] ·
datasets: [[Subjects/DIP/Projects/DIP-Project/data/README|data/README]] · proposals in `proposal/`.

**Team: 3 people.** *(The Lec01 slides say "teams of 2" — confirmed as allowed.)*

**~30 researched candidate topics, explained in plain language: [[Project-Ideas]].**
Leading options: DCP dehazing assumption audit · low-light reproduction audit · Fourier trilogy.

> [!warning] Registration (Aug 17) and proposal (Aug 20) dates have passed
> Confirm the actual status with the instructor before doing anything else.

## Submissions
- [ ] Project group registration 📅 2026-08-17
- [ ] Project proposal 📅 2026-08-20
- [ ] Project Report 1 📅 2026-09-30
- [ ] Project Report 2 📅 2026-10-26
- [ ] Project presentation 📅 2026-11-12

## Sections
- [ ] Pick topic + form group 📅 2026-08-17
- [ ] Literature / approach survey 📅 2026-08-20
- [ ] Baseline implementation 📅 2026-09-25
- [ ] Main implementation 📅 2026-10-20
- [ ] Evaluation + results 📅 2026-11-05
- [ ] Slides + demo 📅 2026-11-10

## Team
| Member | Roll number | Role in the proposal |
|---|---|---|
| Noel Alex Jacob (listed as "Noel Jacob" on the proposal) | 2025201085 | not assigned in writing |
| Shaik Areesh | 2026909008 | not assigned in writing |
| Sri Harshit K | 2024102022 | not assigned in writing |

The proposal splits the work into A (data and captures), B (restoration and evaluation) and C (synthesis and
comparison) without naming who takes which. Names and roll numbers as on the team table of the report (2026-11-04).

## Log
Findings and changes: [[Subjects/DIP/Projects/DIP-Project/log|log]]. Latest (2026-11-03): proposal files moved to `proposal/`,
folder structure created, stages and data plan written. Later the same day: public data downloaded and curated, forward models and the demo page written and tested (see log).
2026-11-04: all stages run on original and calibrated presets, real tank-water data (TUB) added, local-against-global, subtraction and chain-length studies done, detailed report written (`report/final/report.pdf`, `docs/model-equations.md`); findings F1 to F60 in the log. Still open: own captures (water, glass fog and dirt, plastic, stacks), more diverse data (see the log's open items).

---
[[_syllabus|DIP syllabus]] · [[Dashboard]]
