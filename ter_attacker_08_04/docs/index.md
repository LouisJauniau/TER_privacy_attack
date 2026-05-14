# TER Documentation

## Purpose of this documentation

This documentation describes how the TER project works end to end, from dataset preparation to attack evaluation after anonymization.

The goal is to understand:

- how the dataset is prepared;
- how anonymization is executed;
- how the linkage attack and the membership inference attack (MIA) are built;
- which files are produced at each step.

---

## Project context

The project studies the effect of dataset anonymization on privacy protection.

Starting from a source dataset, an anonymized version is produced, and the project then measures what an attacker can still learn from the published data.

The work focuses on three main blocks:

- anonymization;
- the **linkage attack**;
- the **membership inference attack (MIA)**.

Benchmark scripts also exist in the project, but they mostly serve to automate batches of experiments.

---

## Documentation layout

### Project overview
Presents the overall pipeline, the main steps, and the logical chaining between preparation, anonymization and attacks.

### Anonymization
Explains how a source dataset is transformed into an anonymized dataset, with a public version and an evaluation version.

### Linkage attack
Describes how an attacker auxiliary base is used to build equivalence classes, reduce them, and then infer a sensitive attribute. It also covers the schema-matching step based on Valentine (COMA, JaccardDistanceMatcher, DistributionBased) that sits between phase 1 and phase 2.

### Membership Inference Attack (MIA)
Describes how IN and OUT targets are built, then how the attack decides whether a target belonged to the published dataset.

### Script structure
Gives an overview of the `scripts/` folder and of the role of each main file.

### File formats
Describes the main families of files handled by the project and their place in the pipeline.

---

## Overall pipeline

The project roughly follows this logic:

1. prepare a dataset with a stable internal identifier `record_id`;
2. run an anonymization;
3. produce a public anonymized dataset and an evaluation anonymized dataset;
4. prepare the files required for the linkage attack and/or the MIA;
5. run the attacks;
6. save the detailed results and the HTML reports under `outputs/`.

---

## Important note about `record_id`

The project uses a stable internal identifier, usually `record_id`, to cleanly link the different steps.

This identifier:

- is useful for internal evaluation;
- is kept in `anonymized_eval`;
- can be removed from the public dataset using `--public-drop-columns record_id`.

It must therefore **not** be considered as information published to the attacker.

---

## Important note about the linkage attack

In the current state of the project, the linkage attack follows a two-phase logic on equivalence classes:

- **phase 1 (equivalence class phase 1)**: filtering with attacker-known attributes whose values appear **generalized or suppressed** in the published dataset (`visible_level != 0`);
- **phase 2 (equivalence class phase 2)**: refinement with attacker-known attributes that remain **visible in clear text** (`visible_level == 0`), either via exact match or optionally with `privJedAI` in fuzzy mode.

An optional **schema-matching step** can be inserted between phase 1 and phase 2 using the Valentine library. It simulates an attacker who does not know the column names of the anonymized release but still wants to use their values during phase 2. Schema matchers available include `coma`, `jaccard` (JaccardDistanceMatcher from Valentine), `distribution`, and a pure-Python `baseline_jaccard`.

The detailed documentation of this logic is on the dedicated linkage attack page.

---

## Important note about the MIA

In the current state of the project, the MIA is done in two steps:

1. `make_mia_targets.py` prepares a **published subset** and an **OUT holdout pool** before anonymization;
2. `make_mia_targets_post_ano.py` then builds the **final targets** after anonymization, taking IN targets only from records that actually survived in `anonymized_eval`.

This split is important to keep a consistent ground truth when some rows are removed from the final export.

The MIA also uses the same phase 1 / phase 2 equivalence class logic as the linkage attack.

## Integrated documentation structure

This documentation now combines two complementary parts of the TER project.

- The **privacy attack part** documents the global pipeline, anonymization outputs, linkage attack, MIA, attacker knowledge, and implementation details.
- The **utility evaluation part** documents ARX utility metrics, quality models, aggregate functions, classification benchmarks, and the associated scripts and file formats.

The navigation menu separates these concerns into four main blocks: **Overview**, **Anonymization**, **Attacks**, **Utility metrics and classification**, and **Implementation**.

