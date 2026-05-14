# Linkage attack

## Role of this step

The **linkage attack** tries to link a target known to the attacker to one or more rows of the published anonymized dataset.

In this project, its role is not only to search for an exact re-identification.
It also serves to measure what an attacker can still learn about the **sensitive attribute** from the remaining compatible candidates.

---

## General idea

The attack starts from a realistic assumption:

- the attacker has an auxiliary base;
- they know some attributes of a target;
- they compare this knowledge with the published anonymized dataset.

For each target, the attack builds an **attacker equivalence class**, that is, the set of anonymized rows still compatible with the known information.

It then tries to **reduce** this class as much as possible, and finally looks at the sensitive attribute values among the remaining candidates.

The full pipeline is structured in **two equivalence-class phases**, with an optional schema-matching step between them:

1. **equivalence class phase 1** — build the initial class from generalized/suppressed attributes;
2. **(optional) schema matching** — recover column names in the release when they are obfuscated;
3. **equivalence class phase 2** — refine the class with clear-text attributes (exact or fuzzy).

---

## Main scripts

The most important scripts at this step are:

- `scripts/make_auxiliary_base.py`
- `scripts/run_linkage_attack.py`
- `scripts/schema_matcher.py`
- `scripts/demo_schema_matching.py`
- `scripts/generate_linkage_attack_report.py`

### `make_auxiliary_base.py`
Prepares the attacker auxiliary base from a dataset that already has `record_id`.

### `run_linkage_attack.py`
Runs the attack, produces the detailed files, the `summary.json`, and if possible the HTML report. It also drives the optional schema-matching step.

### `schema_matcher.py`
Schema-matching module used between phase 1 and phase 2. Wraps the Valentine library (COMA, JaccardDistanceMatcher, DistributionBased) and provides a pure-Python Jaccard baseline. Also exposes utilities to obfuscate column names, evaluate recovered mappings, and rename columns back.

### `demo_schema_matching.py`
Standalone demo that showcases the schema-matching step alone (outside of the full attack), useful to check matcher behavior on a given anonymized release and auxiliary base.

### `generate_linkage_attack_report.py`
Generates an HTML report from the attack outputs.

---

## Data used by the attack

The linkage attack relies on three main data sets.

### 1. The attacker auxiliary base

It contains the information the attacker knows about the targets.

Common examples:

- `age`
- `sex`
- `race`
- `marital-status`
- `native-country`

### 2. The public anonymized dataset

This is the attacker's assumed realistic view.

### 3. The evaluation anonymized dataset

This version is used internally to:

- recover `record_id`;
- check whether the true record survived;
- fetch the true sensitive value;
- measure the real quality of the attack.

---

## What the attacker knows

The attacker only knows a subset of attributes.

The script accepts this list through `--known-attrs`.

This partial knowledge is then interpreted through an important internal object:

- `attacker_knowledge.json`

For each known attribute, this file describes:

- the `visible_level` observed in the published dataset;
- the values observed in the anonymized version;
- the projection from attacker-raw value to attacker-visible value.

---

## The notion of `visible_level`

The `visible_level` indicates up to which hierarchy level an attribute remains visible in the published data.

### `visible_level != 0`
The attribute appears in a generalized or suppressed form in the release.

It therefore belongs to **equivalence class phase 1**.

### `visible_level == 0`
The attribute remains visible as clear text in the release.

It can then be used to **refine** the class built in phase 1, that is, it belongs to **equivalence class phase 2**.

This separation is central to the current logic of the attack: the attribute split is **attacker-view based**, not config-based.

---

## Attack logical flow

### Equivalence class phase 1 — initial class

The script selects the known attributes with `visible_level != 0` (these are called `qid_filter_attrs` or `stage1_filter_attrs` in the summary).

For each of them, it projects the target's raw value into the attacker-visible value.

Example:

- raw target value: `United-States`
- published visible value: `North America`

It then keeps only the anonymized rows compatible with all these visible values. Compatibility is evaluated through an inverted index (one entry per attribute value) built once over the anonymized release.

The result is the **phase 1 equivalence class** (also called `qid_equivalence_class` or `stage1_equivalence_class`).

### (Optional) Schema matching between the two phases

If the attacker does not know the column names of the clear-text columns in the release, a schema-matching step can be inserted here. See the dedicated section below.

### Equivalence class phase 2 — refinement

The script then selects the known attributes with `visible_level == 0` (called `refine_attrs` or `stage2_refine_attrs` in the summary).

Starting from the phase 1 class, it further reduces the candidates:

- either by exact match;
- or, if enabled, with `privJedAI` in fuzzy mode.

The result is the **phase 2 equivalence class** (also called `reduced_equivalence_class` or `stage2_equivalence_class`).

---

## Phase 1 class reuse (cache)

The script does not recompute the phase 1 class for every target.

If several targets share exactly the same projection on the phase 1 attributes, they reuse the same phase 1 class through a cache (`qid_stage1_cache`), keyed on a tuple of `(attribute, projected value)` pairs.

This is what the variable `n_distinct_stage1_groups` counts.

This optimization has two effects:

- it speeds up the attack;
- it influences the complexity counters.

---

## Schema matching between phase 1 and phase 2

### Motivation

Phase 2 refines the equivalence class using attacker-known attributes that stay in clear text in the release. For this to work, the attacker needs to know **which column** in the release corresponds to each clear-text attribute they have in their auxiliary base.

In a realistic scenario, the data publisher may obfuscate or rename these columns. Phase 2 cannot proceed as-is: the attacker sees values but not labels.

The schema-matching step recovers, from values alone, a mapping from each obfuscated anonymized column to its corresponding auxiliary-base column. It is instance-based: it compares values, not names.

### Module involved

The logic lives in `scripts/schema_matcher.py`. It is wired into `run_linkage_attack.py` via three CLI flags:

- `--obfuscate-refine-attrs age,hours-per-week,capital-gain` — the comma-separated list of clear-text attributes to treat as unknown-named in `df_public` (they will be renamed to `col_0`, `col_1`, ... before matching);
- `--schema-matcher {coma, jaccard, distribution, baseline_jaccard}` — the matcher to use, defaulting to `jaccard`;
- `--schema-matcher-min-score <float>` — minimum similarity score required to keep a recovered mapping.

When `--obfuscate-refine-attrs` is not passed, the schema-matching step is skipped and phase 2 runs as usual on known column names.

### Available matchers

The `schema_matcher.build_valentine_matcher` factory exposes three Valentine matchers plus a baseline:

| Matcher name        | Backend                                            | Notes                                                                                                                     |
|---------------------|----------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------|
| `coma`              | `valentine.algorithms.Coma`                        | Instance-based COMA. Compares values, not just names. Parameter `max_n=0` disables the cap on returned matches.           |
| `jaccard`           | `valentine.algorithms.JaccardDistanceMatcher`      | Valentine's Jaccard-distance matcher. Default `threshold_dist=0.8`. Strong baseline for categorical columns with overlapping vocabularies. |
| `distribution`      | `valentine.algorithms.DistributionBased`           | Compares value distributions. Useful on continuous/numeric-like columns. Default `threshold1=threshold2=0.15`.            |
| `baseline_jaccard`  | Pure Python in `schema_matcher.jaccard_baseline`   | Greedy Jaccard over unique-value sets. Does not depend on Valentine. Useful as sanity check against Valentine's matchers. |

The recommended default is `jaccard` because it is simple, fast, and works well on the Adult dataset's categorical columns. `baseline_jaccard` is kept as a reference to confirm that Valentine's `JaccardDistanceMatcher` behaves as expected.

### Schema-matching logical flow

Inside `run_linkage_attack`, when `--obfuscate-refine-attrs` is set:

1. rename the listed columns in `df_public` to `col_0`, `col_1`, ... (`obfuscate_columns`), keeping the ground truth aside for evaluation only;
2. collect the candidate columns from the auxiliary base (`df_aux`), excluding `target_id_col` and the sensitive attribute;
3. call `recover_column_mapping(...)` (Valentine matchers) or `jaccard_baseline(...)` for the pure-Python fallback;
4. drop any recovered mapping whose score is below `--schema-matcher-min-score`;
5. rename the obfuscated columns of `df_public` back to the auxiliary-base names using the recovered mapping (`apply_recovered_mapping`);
6. remove from `known_attrs` any originally obfuscated column whose name could not be recovered — the attacker cannot use it in phase 2;
7. evaluate the recovered mapping against the ground truth (`evaluate_mapping`) and save both a JSON and a CSV trace.

Phase 2 then proceeds as if the names had never been hidden, operating only on columns that were actually recovered.

### Outputs of the schema-matching step

When the step runs, the attack directory contains:

- `schema_matching_results.json` — full trace: matcher used, `min_score`, obfuscated columns, candidate columns on the KB side, ground truth, recovered pairs with scores, metrics (`n_obfuscated`, `n_mapped`, `n_correct`, `coverage`, `accuracy_on_mapped`, `recall`);
- `schema_matching_pairs.csv` — one row per obfuscated column with `anon_column`, `true_column`, `predicted_column`, `score`, `is_mapped`, `is_correct`, `matcher`, `min_score`.

The `summary.json` of the attack also gains the following fields: `schema_matching_enabled`, `schema_matcher_name`, `schema_matcher_min_score`, `schema_matching_results_json`, `schema_matching_pairs_csv`.

### Standalone demo

`scripts/demo_schema_matching.py` runs the schema-matching step in isolation, outside of the full attack. It loads an anonymized CSV and an auxiliary CSV, obfuscates a chosen list of `refine_attrs`, runs a chosen matcher, prints the recovered mapping and the evaluation metrics, and shows how the renamed DataFrame can be plugged back into the attack pipeline. Useful when calibrating `--schema-matcher` and `--schema-matcher-min-score` on new data.

---

## Phase 2 compatibility and refinement types

### Exact match
Used when the refinement attribute remained visible as-is in the published dataset.

### Fuzzy match with `privJedAI`
Optional, enabled with `--use-privjedai-fuzzy`.

This mode is useful when two values should be treated as close despite form variations.

It does not replace phase 1: it only kicks in during phase 2 on clear-text attributes, after the optional schema-matching step.

### Suppression during phase 2
During refinement, an anonymized value that is a pure suppression token (e.g. `*`, `**`, empty string) stays compatible with any attacker value. This avoids collapsing the phase 2 class to size 0 when an attribute was mistakenly routed to refinement because its column is entirely suppressed.

---

## Sensitive attribute inference

Once the phase 2 class is obtained, the attack looks at the sensitive attribute values among the candidates.

### Case 1: empty class
No compatible candidate remains.

### Case 2: a single candidate
The target is re-identified uniquely in the scope of the attack.

### Case 3: multiple candidates, a single sensitive value
The sensitive attribute is inferred with certainty, even without unique re-identification.

### Case 4: multiple candidates, multiple sensitive values
The script produces a probability distribution from the observed frequencies.

Example:

- `<=50K`: 75 %
- `>50K`: 25 %

---

## Outputs produced

Each linkage run usually produces a folder of the form:

- `outputs/attacks/linkage/<attack_id>/`

It typically contains:

### `summary.json`
Global summary of the attack.

### `targets.csv`
Per-target results.

### `equivalence_class_candidates.csv`
One row per candidate kept in the final (phase 2) equivalence classes.

### `attacker_knowledge.json`
Description of the attacker's visibility over the known attributes.

### `schema_matching_results.json` (optional)
Only present when `--obfuscate-refine-attrs` was set. Trace of the matcher run, recovered mapping and evaluation metrics.

### `schema_matching_pairs.csv` (optional)
Flat, per-column view of the schema-matching result.

### `prefilter_debug/` (optional)
Debug files detailing the rows kept at the different phases.

### `<attack_id>__report.html`
HTML report generated automatically if generation succeeds.

In addition, an aggregated summary can be appended to:

- `outputs/attacks/linkage/linkage_summary.csv`

---

## Logical flow of `make_auxiliary_base.py`

The script roughly follows these steps:

1. load a dataset already prepared with `record_id`;
2. select the columns known by the attacker;
3. optionally keep only the individuals still present in `anonymized_eval`;
4. sample the targets;
5. produce an auxiliary CSV and a JSON metadata file.

---

## Logical flow of `run_linkage_attack.py`

The script roughly follows these steps:

1. load the runtime configuration;
2. load the auxiliary base;
3. load `anonymized` and `anonymized_eval`;
4. build `attacker_knowledge`;
5. split attributes between phase 1 and phase 2 based on `visible_level`;
6. **(optional)** run the schema-matching step on the release columns corresponding to phase 2 attributes, using Valentine (`coma`, `jaccard`, `distribution`) or the pure-Python `baseline_jaccard`, then rename the release columns back and drop unrecovered attributes from `known_attrs`;
7. build or reuse the phase 1 equivalence classes (with `qid_stage1_cache`);
8. apply the phase 2 refinement (exact, or fuzzy with privJedAI);
9. measure the size of the final classes;
10. infer the sensitive attribute;
11. save the artifacts and the report.

---

## Main variables of `summary.json`

| Variable | Meaning |
|---|---|
| `attack_id` | Unique identifier of the linkage run. |
| `known_attrs` | Attributes known by the attacker. |
| `qid_filter_attrs` / `stage1_filter_attrs` | Attributes used in phase 1 (seen as generalized or suppressed). |
| `refine_attrs` / `stage2_refine_attrs` | Attributes used in phase 2 (still clear-text). |
| `skipped_refine_attrs` | Phase-2 attributes dropped because their visibility could not be resolved. |
| `n_targets` | Number of attacked targets. |
| `n_anonymized_rows` | Number of rows in the anonymized dataset used by the attack. |
| `n_distinct_stage1_groups` | Number of distinct phase-1 groups after caching. |
| `unique_reidentification_rate` | Share of targets whose phase 2 class contains exactly one candidate that matches the true target. |
| `false_unique_match_rate` | Share of targets whose phase 2 class contains a single candidate that is **not** the true record. |
| `true_record_kept_after_refinement_rate` | Share of targets whose true record is still present after phase 2. |
| `avg_qid_equivalence_class_size` / `avg_stage1_equivalence_class_size` | Average size of the phase 1 classes. |
| `median_qid_equivalence_class_size` / `median_stage1_equivalence_class_size` | Median size of the phase 1 classes. |
| `avg_equivalence_class_size` | Average size of the phase 2 classes. |
| `median_equivalence_class_size` | Median size of the phase 2 classes. |
| `max_equivalence_class_size` | Largest phase 2 class observed. |
| `avg_reduction_rate` | Average reduction from phase 1 size to phase 2 size. |
| `certainty_sensitive_inference_rate` | Share of targets for which the sensitive attribute is inferred with certainty. |
| `avg_true_sensitive_probability` | Average probability assigned to the true sensitive value. |
| `median_true_sensitive_probability` | Median probability assigned to the true sensitive value. |
| `avg_top_sensitive_probability` | Average probability of the most likely sensitive value in the phase 2 class. |
| `schema_matching_enabled` | True when the schema-matching step ran. |
| `schema_matcher_name` | Name of the matcher used (`coma`, `jaccard`, `distribution`, `baseline_jaccard`). |
| `schema_matcher_min_score` | Minimum score threshold applied to recovered mappings. |
| `schema_matching_results_json` | Path to the full schema-matching trace JSON. |
| `schema_matching_pairs_csv` | Path to the per-column schema-matching CSV. |

---

## Main variables of `targets.csv`

| Variable | Meaning |
|---|---|
| `target_id` | Internal identifier of the target. |
| `qid_equivalence_class_size` / `stage1_equivalence_class_size` | Size of the phase 1 class. |
| `reduced_equivalence_class_size` / `stage2_equivalence_class_size` / `equivalence_class_size` | Size of the final (phase 2) class. |
| `equivalence_class_reduction` | Absolute reduction between phase 1 and phase 2 sizes. |
| `equivalence_class_reduction_rate` | Relative reduction between phase 1 and phase 2 sizes. |
| `true_record_in_qid_class` | Whether the true record was present at phase 1. |
| `true_record_in_reduced_class` | Whether the true record is still present after phase 2. |
| `unique_reidentified` | Whether the target is uniquely and correctly re-identified. |
| `false_unique_match` | Whether a single but incorrect candidate was obtained. |
| `true_sensitive_value` | True value of the sensitive attribute. |
| `predicted_sensitive_top_value` | Most likely sensitive value in the phase 2 class. |
| `predicted_sensitive_top_probability` | Probability of that top sensitive value. |
| `true_sensitive_probability` | Probability assigned to the true sensitive value. |
| `sensitive_value_certain` | Whether the sensitive inference is certain. |
| `n_distinct_sensitive_values` | Number of distinct sensitive values in the phase 2 class. |

---

## Complexity variables

These counters give an **estimate of the amount of logical work** performed by the attack.

| Variable | Meaning |
|---|---|
| `value_index_row_visits` | Number of passes over rows of the anonymized dataset when building or using the value index for filtering. |
| `targets_evaluated` | Number of targets actually evaluated. |
| `array_cells_initialized` | Total number of temporary-array cells initialized while processing targets. Indicator of memory cost and mask preparation cost. |
| `attribute_positive_mask_cells` | Number of positively marked cells in the attribute masks (positions recognized as compatible for a given attribute). |
| `matching_row_visits` | Number of row visits during the phase 1 compatibility filtering. |
| `mask_and_cells` | Number of logical AND operations between masks to combine several attribute constraints. |
| `final_mask_reads` | Number of reads of the final mask to extract the kept candidates. |
| `equivalence_class_candidate_rows_output` | Total number of candidates output by phase 1, summed over all targets. |
| `refinement_candidate_row_visits` | Total number of candidate visits during phase 2. |
| `refinement_exact_tests` | Number of exact tests performed during phase 2. |
| `refinement_fuzzy_tests` | Number of fuzzy tests performed (e.g. via `privJedAI`) when that option is enabled. |
| `refinement_mask_cells` | Number of cells handled in the phase 2 masks. |
| `reduced_equivalence_class_candidate_rows_output` | Total number of candidates remaining after phase 2, summed over all targets. |
| `estimated_total_operations` | Global estimate of the number of logical operations performed by the attack. |

---

## Configuration variables

| Variable | Meaning |
|---|---|
| `attack_id` | Unique identifier of the attack and of its output folder. |
| `known_attrs` | Attributes known by the attacker. |
| `qid_filter_attrs` / `stage1_filter_attrs` | Attributes used in phase 1 to build the initial equivalence class. |
| `refine_attrs` / `stage2_refine_attrs` | Attributes used in phase 2 to reduce the phase 1 class. |
| `target_id_col` | Name of the column identifying the targets. |
| `sensitive_attr` | Sensitive attribute that the attack tries to infer. |
| `n_anonymized_rows` | Number of rows of the anonymized dataset actually used by the attack. |
| `use_privjedai_fuzzy` | Whether fuzzy comparisons were enabled during phase 2. |
| `seed` | Random seed used for reproducibility. |
| `schema_matching_enabled` | Whether the schema-matching step ran. |
| `schema_matcher_name` | `coma`, `jaccard`, `distribution` or `baseline_jaccard`. |
| `schema_matcher_min_score` | Minimum score required to keep a recovered mapping. |

---

## Summary

The current linkage attack relies on a precise logic:

1. build a phase 1 equivalence class using attributes seen as generalized or suppressed;
2. optionally recover obfuscated column names in the release using Valentine (COMA, JaccardDistanceMatcher, DistributionBased) or a Jaccard baseline;
3. reduce that class in phase 2 using attributes still visible in clear text (exact match or privJedAI fuzzy);
4. measure what the final class still reveals about the sensitive attribute.
