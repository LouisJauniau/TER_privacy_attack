# Input / output file formats

## Purpose of this page

This page describes the main files handled by the project.

The goal is to understand:

- which files serve as inputs;
- which files are produced;
- what they are used for in the pipeline;
- what the difference is between public versions, evaluation versions and intermediate files.

---

## Overview

The project handles several families of files:

- source datasets;
- datasets prepared with `record_id`;
- configuration files;
- generalization hierarchies;
- intermediate files for the linkage attack;
- intermediate files for the MIA;
- anonymization output files;
- attack output files;
- schema-matching output files (linkage attack);
- HTML reports.

---

## 1. Source and prepared datasets

### Source dataset

Typical examples:

- `data/adult.csv`
- `data/adult_with_record_id.csv`

#### Role
Starting point of the experiments.

#### Format
CSV with header.

---

### Dataset prepared with `record_id`

Typical example:

- `data/adult_with_record_id.csv`

Associated metadata:

- `data/adult_with_record_id.json`

#### Role
Stable version of the dataset used by the current pipeline.

#### Content
It contains:

- `record_id`;
- the quasi-identifiers;
- the sensitive attribute;
- the other columns of the dataset.

#### Why it matters
This is the clean base file that feeds all downstream steps.

---

## 2. Configuration files

### Base configuration

Example:

- `configs/adult_base.json`

#### Role
Describes a starting anonymization experiment.

---

### Executed runtime configuration

Typical folder:

- `outputs/configs/`

Possible examples:

- `outputs/configs/qi_age-sex__k_5__l_2__t_None__supp_10__arx.json`
- `outputs/configs/adult.runtime.json`
- `outputs/configs/adult_base_with_record_id.json`

#### Role
Keeps exactly the configuration that was actually used during the run.

#### Format
JSON.

---

## 3. Generalization hierarchies

Typical examples:

- `hierarchies/age.csv`
- `hierarchies/sex.csv`
- `hierarchies/race.csv`
- `hierarchies/native-country.csv`

#### Role
Describe the successive generalization levels for the quasi-identifiers.

#### Format
CSV where each row links a source value to its generalization levels.

---

## 4. Files produced by the anonymization

### Public anonymized dataset

Typical folder:

- `outputs/anonymized/`

Example:

- `outputs/anonymized/<experiment_id>.csv`

#### Role
Represents the published version of the dataset.

#### Important note
Columns such as `record_id` can be removed from this version.

---

### Evaluation anonymized dataset

Typical folder:

- `outputs/anonymized_eval/`

Example:

- `outputs/anonymized_eval/<experiment_id>.csv`

#### Role
Keeps an internal version that allows linking rows to `record_id`.

#### Important note
This version is **not** meant to be visible to the attacker.

---

### Anonymization metrics

Typical folder:

- `outputs/metrics/`

Example:

- `outputs/metrics/<experiment_id>.json`

#### Role
Summarize the result of an anonymization run.

#### Typical content
You can find:

- the export paths;
- the columns dropped from the public version;
- the suppression statistics;
- the number of rows removed because all QIs were `*`.

---

### CSV summary of runs

Typical file:

- `outputs/benchmark_summary.csv`

#### Role
Aggregate several anonymization runs into a single table.

---

## 5. Linkage attack preparation files

### Auxiliary base

Typical folder:

- `outputs/auxiliary/`

Example default names:

- `outputs/auxiliary/<dataset>__aux__known_<attrs>__released_only__n_<n>.csv`
- `outputs/auxiliary/<dataset>__aux__known_<attrs>__all_records__n_<n>.csv`

#### Role
Contains the targets and the attributes known by the attacker.

#### Format
CSV.

---

### Auxiliary base metadata

Same base name with a `.json` extension.

Example:

- `outputs/auxiliary/<...>.json`

#### Role
Keep track of:

- the source dataset used;
- the known attributes;
- the sample size;
- the target-population mode;
- optionally the link with `released_eval`.

---

## 6. Linkage attack output files

Each run usually produces a folder:

- `outputs/attacks/linkage/<attack_id>/`

### `summary.json`

#### Role
Global summary of the attack.

#### Notably contains
- the list of known attributes;
- the phase 1 / phase 2 split (both legacy `qid_filter_attrs` / `refine_attrs` and new `stage1_filter_attrs` / `stage2_refine_attrs` keys are present);
- class sizes for phase 1 and phase 2;
- the sensitive inference metrics;
- `operation_counter`;
- the schema-matching configuration and output paths, when enabled (`schema_matching_enabled`, `schema_matcher_name`, `schema_matcher_min_score`, `schema_matching_results_json`, `schema_matching_pairs_csv`).

---

### `targets.csv`

#### Role
Detailed per-target results.

#### Typical content
- phase 1 class size;
- phase 2 (final) class size;
- reduction;
- presence of the true record at both phases;
- sensitive attribute distribution.

---

### `equivalence_class_candidates.csv`

#### Role
List the final candidates kept in the phase 2 equivalence classes.

#### Format
One row per `(target, final candidate)` pair.

---

### `attacker_knowledge.json`

#### Role
Describe the attacker's view of the known attributes.

#### Typical content
For each attribute:

- `visible_level`;
- `observed_values`;
- `projection`.

---

### `schema_matching_results.json` (optional)

#### Role
Produced only when the schema-matching step ran (i.e. `--obfuscate-refine-attrs` was set).

#### Typical content
- `matcher` used (`coma`, `jaccard`, `distribution`, `baseline_jaccard`);
- `min_score` threshold;
- `obfuscated_columns` list;
- `anon_unknown_cols` (the `col_0`, `col_1`, ... names);
- `kb_candidate_cols` (auxiliary-base candidate columns);
- `truth` (ground-truth mapping, for evaluation only);
- `recovered` (mapping `anon_col → { predicted_column, score }`);
- `metrics` (`n_obfuscated`, `n_mapped`, `n_correct`, `coverage`, `accuracy_on_mapped`, `recall`);
- `pairs` (the same information in a flat, per-pair form).

---

### `schema_matching_pairs.csv` (optional)

#### Role
Flat per-obfuscated-column view of the schema-matching result. Easy to ingest for comparisons across runs.

#### Typical columns
- `anon_column` — `col_0`, `col_1`, ...;
- `true_column` — ground truth, for evaluation only;
- `predicted_column` — column picked on the KB side, or empty if below threshold;
- `score` — similarity score returned by the matcher;
- `is_mapped` — whether the matcher returned any mapping above `min_score`;
- `is_correct` — whether the mapping matches the ground truth;
- `matcher` — matcher name;
- `min_score` — threshold used.

---

### `prefilter_debug/` (optional)

#### Role
Store debug exports for some targets (rows kept at phase 1 and phase 2, compatible values by attribute, etc.).

---

### HTML report

Typical name:

- `outputs/attacks/linkage/<attack_id>/<attack_id>__report.html`

#### Role
Present the main results in a readable form.

---

### Aggregated linkage attack summary

Typical file:

- `outputs/attacks/linkage/linkage_summary.csv`

#### Role
Aggregate several linkage runs into a single table. Also exposes the schema-matching configuration columns when present.

---

## 7. MIA preparation files

### Published subset

Typical folder:

- `outputs/prepared_data/`

Example:

- `outputs/prepared_data/<n>.published.csv`

#### Role
Subset of the original dataset that will be anonymized for the MIA.

---

### OUT holdout pool

Typical folder:

- `outputs/prepared_data/`

Example:

- `outputs/prepared_data/<n>.out.csv`

#### Role
Pool of individuals absent from the published dataset.

---

### MIA split metadata

Example:

- `outputs/prepared_data/<n>.published.json`

#### Role
Keep track of:

- the path of the published subset;
- the path of the OUT pool;
- the expected size of the IN pool;
- the seed;
- the size of the attacker knowledge base.

---

### MIA attacker base

Example:

- `outputs/prepared_data/<n>.attacker_base.csv`

#### Role
Attacker knowledge base built after anonymization, containing both OUT candidates and IN survivors.

---

### Post-anonymization MIA targets

Typical folder:

- `outputs/mia_targets/`

Example:

- `outputs/mia_targets/<n>.targets_post_ano.csv`

Associated metadata:

- `outputs/mia_targets/<n>.targets_post_ano.json`

#### Role
Contain the final balanced IN/OUT targets with the `is_member` label.

---

## 8. MIA output files

Each run usually produces a folder:

- `outputs/attacks/mia/<attack_id>/`

### `summary.json`

#### Role
Global summary of the MIA attack.

#### Notably contains
- `known_qids`;
- the phase 1 and phase 2 attributes;
- the confusion matrix;
- accuracy, precision, recall, F1;
- average class sizes for phase 1 and phase 2.

---

### `targets.csv`

#### Role
Detailed per-target results.

#### Typical content
- ground truth;
- final prediction;
- phase 1 class size;
- final number of compatible candidates (phase 2);
- compatible fraction;
- textual reason of the decision.

---

### HTML report

Typical name:

- `outputs/attacks/mia/<attack_id>/<attack_id>__report.html`

#### Role
Present the MIA results in a readable form.

---

### Aggregated MIA summary

Typical file:

- `outputs/attacks/mia/mia_summary.csv`

#### Role
Aggregate several MIA runs into a single table.

---

## 9. Difference between public, internal and intermediate files

### Public files
They represent the attacker's plausible view.

Examples:
- `outputs/anonymized/...`

### Internal evaluation files
They are used to evaluate the attacks without being published.

Examples:
- `outputs/anonymized_eval/...`

### Intermediate files
They serve to prepare the attacks or to document the pipeline.

Examples:
- `outputs/prepared_data/...`
- `outputs/auxiliary/...`
- `outputs/mia_targets/...`

### Schema-matching trace files
They are produced only when the schema-matching step ran, and document what the matcher saw, recovered and evaluated.

Examples:
- `outputs/attacks/linkage/<attack_id>/schema_matching_results.json`
- `outputs/attacks/linkage/<attack_id>/schema_matching_pairs.csv`

---

## Summary

The project therefore handles several file levels:

1. source files;
2. prepared files;
3. anonymization files;
4. attack preparation files;
5. detailed attack results (including schema-matching traces for the linkage attack);
6. HTML reports.

Understanding where each family fits is essential to follow the full pipeline.
