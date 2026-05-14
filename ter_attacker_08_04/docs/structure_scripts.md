# Script structure

## Purpose of this page

This page gives a clear view of how the project's scripts are organized, to quickly understand:

- which files play a central role;
- which scripts are used to run the main steps;
- which files serve as support;
- which scripts are mostly used for automation or report generation.

---

## Overview

The `scripts/` folder mainly contains:

1. dataset preparation scripts;
2. main execution scripts;
3. attack preparation scripts;
4. shared utility files;
5. schema-matching scripts;
6. benchmark scripts;
7. report generation scripts.

---

## Main scripts

In the current project state, the most important scripts are:

- `prepare_dataset_with_record_id.py`
- `run_ano.py`
- `make_auxiliary_base.py`
- `run_linkage_attack.py`
- `make_mia_targets.py`
- `make_mia_targets_post_ano.py`
- `run_mia_attack.py`

They correspond directly to the main pipeline documented on the other pages.

---

## 1. Dataset preparation

### `prepare_dataset_with_record_id.py`

This is the most important preparation script before anonymization.

#### Role
- add `record_id` if missing;
- check its uniqueness;
- produce a stable version of the dataset;
- optionally generate an updated copy of a configuration.

#### Why it matters
The current pipeline relies heavily on `record_id` to link together:

- anonymization;
- linkage attack;
- MIA;
- internal evaluation.

In particular, `record_id` must be declared as `insensitive_attributes` in the runtime config (and removed from quasi-identifiers / sensitive attributes) so that it is not generalized or suppressed during anonymization.

---

## 2. Main execution scripts

### `run_ano.py`

Main entry point for anonymization.

#### Role
- load an experiment configuration;
- prepare the runtime configuration;
- run the anonymization;
- save the produced outputs.

#### Typical outputs
- executed configuration;
- public anonymized dataset;
- evaluation anonymized dataset;
- metrics.

---

### `run_linkage_attack.py`

Main entry point to run the linkage attack.

#### Role
- load the auxiliary base;
- load the anonymized datasets;
- build `attacker_knowledge`;
- split attributes between phase 1 and phase 2 based on `visible_level`;
- optionally run the Valentine / Jaccard schema-matching step between the two phases;
- build the phase 1 and phase 2 equivalence classes;
- infer the sensitive attribute;
- save the results and the HTML report.

#### Current specificity
The script applies a two-phase equivalence class logic based on `visible_level`, with an optional schema-matching step inserted between phase 1 and phase 2 when `--obfuscate-refine-attrs` is set.

---

### `run_mia_attack.py`

Main entry point to run the membership inference attack.

#### Role
- load the MIA targets;
- load the anonymized datasets;
- build `attacker_knowledge`;
- split attributes between phase 1 and phase 2;
- compute the compatible candidates via phase 1 / phase 2 equivalence classes;
- predict IN or OUT;
- save the results and the HTML report.

---

## 3. Attack-data preparation scripts

### `make_auxiliary_base.py`

Prepares the auxiliary base used by the linkage attack.

#### Role
- start from a dataset already prepared with `record_id`;
- select the attributes known by the attacker;
- sample individuals;
- optionally keep only the individuals still present in `anonymized_eval` via `--released-eval`.

#### Why it matters
In the current version, the linkage attack is used in strict mode: targets must in practice still be present in `anonymized_eval`.

---

### `make_mia_targets.py`

Prepares the MIA pre-anonymization split.

#### Role
- start from the original dataset;
- create a `published subset`;
- create an `OUT holdout pool`;
- write a JSON with the split metadata;
- optionally produce a config retargeted to the published subset.

#### Important note
This script no longer creates the final IN/OUT targets directly.

---

### `make_mia_targets_post_ano.py`

Prepares the final MIA targets after anonymization.

#### Role
- read the `published subset`;
- read the `OUT holdout pool`;
- read `anonymized_eval`;
- identify the survivors;
- build the attacker base;
- build balanced IN/OUT targets.

#### Why it matters
It guarantees that IN targets actually correspond to records still present in the final export.

---

## 4. Shared utility files

### `common.py`

General toolbox of the project.

#### Role
- path handling;
- JSON read/write;
- folder creation;
- helper functions reused across several scripts.

---

### `attack_common.py`

Utilities common to both attacks.

#### Role
- normalized CSV loading;
- runtime configuration loading;
- `attacker_knowledge` construction;
- `visible_level` inference;
- shared helpers for validation and summary writing.

---

### `linkage_helpers.py`

Helpers specific to the linkage attack.

#### Role
- value-index construction;
- compatibility logic (exact, generalized, suppressed, privJedAI fuzzy);
- refinement (exact or fuzzy) for phase 2;
- summary of the sensitive attribute distributions.

---

### `privjedai_utils.py`

Integration layer with `privJedAI`.

#### Role
- fuzzy-mode configuration;
- similarity computation;
- Bloom-filter comparisons for clear-text attributes used in phase 2.

---

## 5. Schema-matching scripts

The schema-matching step is inserted between phase 1 and phase 2 of the linkage attack. It is used when the anonymized release has obfuscated column names on the clear-text columns that phase 2 would otherwise rely on.

### `schema_matcher.py`

Module providing the schema-matching primitives.

#### Role
- `build_valentine_matcher(name)` — factory for Valentine matchers: `coma` (`valentine.algorithms.Coma`), `jaccard` (`valentine.algorithms.JaccardDistanceMatcher`), `distribution` (`valentine.algorithms.DistributionBased`);
- `obfuscate_columns(...)` — rename a subset of columns to `col_0`, `col_1`, ... and keep the ground-truth mapping aside for evaluation;
- `recover_column_mapping(...)` — run Valentine on the two column subsets and return a one-to-one mapping `anon_column -> (kb_column, score)`, filtered by a minimum score;
- `jaccard_baseline(...)` — pure-Python greedy Jaccard baseline over unique-value sets, useful as a sanity check against Valentine;
- `apply_recovered_mapping(...)` — rename the obfuscated columns back to the auxiliary-base names so phase 2 can run unchanged;
- `evaluate_mapping(...)` — compare the recovered mapping against the ground truth and return coverage / accuracy / recall metrics.

#### Why it matters
This is the module that makes the attack robust to a realistic scenario where the attacker does not know the labels of the release's clear-text columns. It is invoked by `run_linkage_attack.py` when `--obfuscate-refine-attrs` is set.

---

### `demo_schema_matching.py`

Standalone demo that runs only the schema-matching step.

#### Role
- load an anonymized public CSV and an auxiliary KB CSV;
- obfuscate a chosen list of `refine_attrs`;
- run the chosen matcher (`coma`, `jaccard`, `distribution`, or `baseline_jaccard`);
- print the recovered mapping and the evaluation metrics;
- show how the renamed DataFrame can be plugged back into `run_linkage_attack._evaluate_target()`.

#### Why it matters
Useful to calibrate `--schema-matcher` and `--schema-matcher-min-score` on new data, outside of the full attack pipeline.

---

## 6. Report generation scripts

### `generate_linkage_attack_report.py`

Generates an HTML report for a linkage attack.

#### Role
- read `summary.json`;
- read `targets.csv`;
- optionally read the config and the anonymization metrics;
- produce a synthetic HTML report.

---

### `generate_mia_attack_report.py`

Generates an HTML report for a MIA.

#### Role
- read `summary.json`;
- read `targets.csv`;
- compute the classification indicators;
- produce an HTML report.

---

## 7. Benchmark scripts

### `run_benchmark.py`
Automates several anonymization runs.

### `run_linkage_benchmark.py`
Automates batches of linkage attacks.

### `run_mia_benchmark.py`
Automates batches of MIA attacks.

### `run_linkage_phase_curve.py`
Sweeps a pool of quasi-identifiers, runs the full pipeline (anonymization → auxiliary base → linkage attack) for each combination, and aggregates phase 1 and phase 2 equivalence-class sizes. Produces a plot of the two curves as a function of the number of generalized attacker-known attributes.

These scripts are useful to produce several comparable experiments, but they are not the logical core of the attacks themselves.

---

## Logical organization

### Dataset preparation block
- `prepare_dataset_with_record_id.py`

### Anonymization block
- `run_ano.py`

### Linkage preparation block
- `make_auxiliary_base.py`

### Linkage attack block
- `run_linkage_attack.py`
- `linkage_helpers.py`
- `privjedai_utils.py`
- `schema_matcher.py` *(optional schema-matching step between phase 1 and phase 2)*
- `demo_schema_matching.py` *(standalone demo)*

### MIA preparation block
- `make_mia_targets.py`
- `make_mia_targets_post_ano.py`

### MIA attack block
- `run_mia_attack.py`

### Shared utilities block
- `common.py`
- `attack_common.py`

### Reports block
- `generate_linkage_attack_report.py`
- `generate_mia_attack_report.py`

### Benchmarks block
- `run_benchmark.py`
- `run_linkage_benchmark.py`
- `run_mia_benchmark.py`
- `run_linkage_phase_curve.py`

---

## Summary

The current script organization reflects a pipeline structure:

1. prepare the dataset;
2. anonymize;
3. prepare the attack inputs;
4. run the attacks (with the optional Valentine / Jaccard schema-matching step for the linkage attack);
5. generate the reports.

The main recent evolutions are:

- the MIA is now split into a pre-anonymization phase and a post-anonymization phase;
- both attacks use an explicit **phase 1 / phase 2 equivalence class** engine driven by `visible_level`;
- the linkage attack now supports a schema-matching step based on Valentine (`coma`, `jaccard`, `distribution`) or a pure-Python `baseline_jaccard` to handle obfuscated column names in the release.
