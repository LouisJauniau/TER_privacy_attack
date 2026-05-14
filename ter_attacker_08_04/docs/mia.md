# Membership Inference Attack (MIA)

## Role of this step

The **Membership Inference Attack (MIA)** tries to determine whether a given target was part of the published dataset before anonymization.

In other words, the attacker is not trying to recover a specific row here, nor to infer a sensitive attribute as a priority. They want to answer the question:

**"Was this individual part of the dataset used to produce the published anonymized data?"**

Within this project, this attack therefore evaluates a **membership leakage** risk.

---

## General idea

The general logic of the MIA is:

1. build a balanced set of targets between members and non-members;
2. compare the information known about each target with the published anonymized dataset;
3. build a class of compatible candidates (using the same phase 1 / phase 2 equivalence class logic as the linkage attack);
4. decide whether this class is a membership signal or not;
5. compare the prediction with the ground truth.

---

## Main scripts

The main scripts at this step are:

- `scripts/make_mia_targets.py`
- `scripts/make_mia_targets_post_ano.py`
- `scripts/run_mia_attack.py`
- `scripts/generate_mia_attack_report.py`

### `make_mia_targets.py`
Prepares the pre-anonymization split.

It produces:

- a `published subset` which will be anonymized;
- an `OUT holdout pool` which stays out of the publication;
- a JSON file with the split metadata.

### `make_mia_targets_post_ano.py`
Builds the final MIA targets after anonymization.

It keeps IN targets only among the individuals of the `published subset` that actually survived in `anonymized_eval`.

### `run_mia_attack.py`
Runs the attack itself.

### `generate_mia_attack_report.py`
Generates an HTML report from the attack outputs.

---

## What the MIA predicts

The main output of the MIA is a binary prediction:

- **IN**: the target is predicted as a member of the published dataset;
- **OUT**: the target is predicted as a non-member.

In the files, the ground truth is usually encoded as:

- `is_member = 1` for IN;
- `is_member = 0` for OUT.

---

## Difference with the linkage attack

### Linkage attack
The linkage attack mainly tries to know:

- which anonymized rows are compatible with a target;
- what can be deduced about the sensitive attribute.

### MIA
The MIA mainly tries to know:

- whether a target is part of the published dataset or not.

In other words:

- the linkage attack addresses the risk of linkage and sensitive inference;
- the MIA addresses the risk of membership disclosure.

Both attacks share the same phase 1 / phase 2 equivalence class engine: the difference lies in what they do with the final class.

---

## Data used by the attack

The MIA relies on three main data sets.

### 1. The MIA targets

This file contains the individuals tested by the attack.

Each row contains:

- `record_id`;
- the attributes known by the attacker;
- the `is_member` label.

### 2. The public anonymized dataset

This is the version visible to the attacker.

### 3. The evaluation anonymized dataset

This version is used internally to:

- check which `record_id` survived;
- evaluate predictions properly;
- keep the steps linked together.

---

## Building the MIA targets

### Step 1: `make_mia_targets.py`

The script roughly follows these steps:

1. load the original dataset;
2. ensure the dataset contains `record_id`;
3. determine the size of the future attacker base via `--attacker-frac` or `--attacker-size`;
4. derive an **OUT holdout pool** from it;
5. keep the rest as the **published subset**;
6. write:
   - `*.published.csv`
   - `*.out.csv`
   - a JSON metadata file.

The current principle is:

- the total size of the attacker base is chosen;
- it is then split evenly between OUT and IN;
- the IN part will be sampled later, after anonymization.

### Step 2: `make_mia_targets_post_ano.py`

This script then follows these steps:

1. load the `published subset`;
2. load the `OUT holdout pool`;
3. load `anonymized_eval`;
4. fetch the `record_id`s that actually survived;
5. form the IN pool from these survivors;
6. build a balanced attacker base;
7. sample an equal number of IN and OUT targets;
8. write:
   - `targets_post_ano.csv`
   - `attacker_base.csv`
   - a JSON metadata file.

---

## What the attacker knows

Just like for the linkage attack, the attacker only knows a subset of attributes.

Common examples:

- `age`
- `sex`
- `race`

These columns are passed via `--known-qids`.

---

## Compatibility notion

The MIA also relies on a compatibility logic.

For a given target, the attack searches the anonymized rows that remain consistent with the known attributes.

### Simple example

If the attacker knows:

- `age = 27`
- `sex = Male`
- `race = White`

and an anonymized row contains:

- `age = [20-29]`
- `sex = Male`
- `race = White`

then this row is compatible with the target.

---

## Current filtering logic (phase 1 / phase 2 equivalence class)

As for the linkage attack, the MIA splits known attributes into two groups based on `visible_level`. The split is attacker-view based, not config-based.

### Equivalence class phase 1
Attributes with `visible_level != 0`.

They serve to build the **initial (phase 1) equivalence class** from generalized/suppressed visible values. The same `qid_stage1_cache` mechanism as in the linkage attack is used to reuse phase 1 classes across targets that share the same projection.

### Equivalence class phase 2
Attributes with `visible_level == 0`.

They serve to **refine** the phase 1 class to the **phase 2 class**, either:

- by exact match;
- or optionally with `privJedAI` if `--use-privjedai-fuzzy` is enabled.

---

## IN / OUT decision in the current version

In the current code, the final decision is deliberately simple.

It mainly relies on two criteria:

- `compatible_candidate_count`
- `compatible_candidate_fraction`

### Current rule

The target is predicted **IN** if:

1. at least one compatible candidate remains after phase 2;
2. the compatible fraction in the dataset is less than or equal to `max_compatible_fraction`.

Otherwise, the target is predicted **OUT**.

This rule is applied by `decide_membership(...)` in `run_mia_attack.py` and its reasoning is exposed in the `decision_reason` field of `targets.csv` for auditability.

---

## Logical flow of `run_mia_attack.py`

The script roughly follows these steps.

### 1. Loading files
The script loads:

- the runtime configuration;
- the targets file;
- the public anonymized dataset;
- the evaluation anonymized dataset.

### 2. Inferring known columns
If needed, it derives `known_qids` from the targets file.

### 3. Building `attacker_knowledge`
As for the linkage attack, the script builds an attacker view of the known attributes and of their `visible_level`.

### 4. Phase 1 / phase 2 split
Known attributes are split between:

- `qid_filter_qids` (phase 1);
- `refine_qids` (phase 2).

### 5. Building the phase 1 class
The script keeps the rows compatible with the generalized attributes, using the `qid_stage1_cache` for reuse across targets.

### 6. Phase 2 refinement
It then reduces that class with the clear-text attributes (exact or privJedAI fuzzy).

### 7. Membership decision
It computes:

- `compatible_candidate_count`;
- `compatible_candidate_fraction`;

then applies the IN/OUT decision rule.

### 8. Evaluation
It finally computes:

- accuracy;
- precision;
- recall;
- F1;
- confusion matrix;
- separate statistics for members and non-members.

---

## Outputs produced

Each MIA run usually produces a folder of the form:

- `outputs/attacks/mia/<attack_id>/`

It typically contains:

### `summary.json`
Global summary of the attack.

### `targets.csv`
Per-target results.

### `<attack_id>__report.html`
HTML report generated automatically if possible.

In addition, an aggregated summary can be appended to:

- `outputs/attacks/mia/mia_summary.csv`

---

## Main variables of `summary.json`

| Variable | Meaning |
|---|---|
| `attack_id` | Identifier of the MIA run. |
| `known_qids` | Attributes known by the attacker. |
| `qid_filter_qids` | Attributes used in phase 1. |
| `refine_qids` | Attributes used in phase 2. |
| `n_targets` | Total number of evaluated targets. |
| `n_members` | Number of IN targets in the ground truth. |
| `n_non_members` | Number of OUT targets in the ground truth. |
| `max_compatible_fraction` | Maximum compatible-fraction threshold allowed to predict IN. |
| `tp` | True positives: IN targets correctly predicted IN. |
| `tn` | True negatives: OUT targets correctly predicted OUT. |
| `fp` | False positives: OUT targets predicted IN. |
| `fn` | False negatives: IN targets predicted OUT. |
| `accuracy` | Overall share of correct predictions. |
| `precision` | Among IN predictions, share of correct ones. |
| `recall` | Among true IN targets, share found as IN. |
| `f1` | Harmonic mean of precision and recall. |
| `member_recall` | Recall restricted to the member class. |
| `non_member_true_negative_rate` | True negative rate on the OUT class. |
| `member_avg_stage1_equivalence_class_size` | Average size of the phase 1 class for members. |
| `non_member_avg_stage1_equivalence_class_size` | Average size of the phase 1 class for non-members. |
| `member_avg_compatible_candidate_count` | Average final (phase 2) compatible-candidate count for members. |
| `non_member_avg_compatible_candidate_count` | Average final (phase 2) compatible-candidate count for non-members. |
| `member_avg_equivalence_class_reduction` | Average reduction from phase 1 to phase 2 for members. |
| `non_member_avg_equivalence_class_reduction` | Average reduction from phase 1 to phase 2 for non-members. |

---

## Complexity variables

These counters estimate the amount of logical work performed by the attack.

| Variable | Meaning |
|---|---|
| `row_index_row_visits` | Number of row visits linked to the index used during the initial filtering phase (phase 1). |
| `targets_evaluated` | Number of targets actually processed. |
| `candidate_row_refs_loaded` | Number of candidate-row references loaded in memory during the attack. |
| `refinement_candidate_row_visits` | Total number of candidate visits during phase 2. |
| `refinement_exact_tests` | Number of exact tests performed during phase 2. |
| `refinement_fuzzy_tests` | Number of fuzzy tests performed when the corresponding option is enabled. |
| `refinement_mask_cells` | Number of cells handled in the phase 2 masks. |
| `membership_decisions` | Number of final IN/OUT decisions taken by the attack. |
| `estimated_total_operations` | Global estimate of the number of logical operations performed. |

---

## Main variables of `targets.csv`

| Variable | Meaning |
|---|---|
| `target_id` | Internal identifier of the target. |
| `ground_truth_member` | Ground truth: 1 if IN, 0 if OUT. |
| `predicted_member` | Final attack prediction. |
| `qid_filter_qids` | Attributes used in phase 1 for this attack. |
| `refine_qids` | Attributes used in phase 2 for this attack. |
| `stage1_equivalence_class_size` | Size of the phase 1 class. |
| `compatible_candidate_count` | Final number of compatible candidates (phase 2). |
| `reduced_equivalence_class_size` | Same information as the final class size. |
| `equivalence_class_reduction` | Absolute reduction from phase 1 to phase 2. |
| `equivalence_class_reduction_rate` | Relative reduction from phase 1 to phase 2. |
| `compatible_candidate_fraction` | Fraction of anonymized rows still compatible. |
| `target_present_in_anonymized` | Whether the target's `record_id` is actually present in `anonymized_eval`. |
| `true_record_in_stage1_class` | Whether the true record is present in the phase 1 class. |
| `true_record_in_reduced_class` | Whether the true record is still present after phase 2. |
| `decision_reason` | Textual trace of the decision logic applied by the script. |

---

## Configuration variables

| Variable | Meaning |
|---|---|
| `attack_id` | Unique identifier of the MIA experiment. |
| `known_qids` | Attributes known by the attacker. |
| `qid_filter_qids` | Attributes used in phase 1 for the initial filtering. |
| `refine_qids` | Attributes used in phase 2 to refine the candidates. |
| `target_id_col` | Name of the column identifying the target. |
| `member_col` | Name of the column carrying the IN/OUT ground truth label. |
| `max_compatible_fraction` | Maximum compatible-fraction threshold used in the decision logic. |
| `use_privjedai_fuzzy` | Whether fuzzy refinement was enabled. |
| `seed` | Random seed for reproducibility. |

---

## Summary

The current MIA relies on a consistent chain:

1. prepare a split before anonymization;
2. rebuild targets after anonymization, keeping only survivors for IN;
3. build a compatible class via phase 1 / phase 2 equivalence-class logic;
4. decide IN or OUT based on the relative size of that class.
