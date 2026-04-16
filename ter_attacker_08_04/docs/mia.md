# Membership Inference Attack (MIA)

## Rôle de cette étape

La **Membership Inference Attack (MIA)** cherche à déterminer si une cible donnée appartenait ou non au dataset publié avant anonymisation.

Autrement dit, l'attaquant ne cherche pas ici à retrouver directement une ligne précise ni à inférer en priorité un attribut sensible.
Il cherche à répondre à la question suivante :

**"Cette personne faisait-elle partie du dataset utilisé pour produire les données anonymisées publiées ?"**

Dans le cadre du projet, cette attaque permet donc d'évaluer un risque de fuite d'**appartenance**.

---

## Idée générale

La logique générale de la MIA est la suivante :

1. construire un ensemble de cibles équilibré entre membres et non-membres ;
2. comparer les informations connues sur chaque cible au dataset anonymisé publié ;
3. construire une classe de candidats compatibles ;
4. décider si cette classe constitue un signal d'appartenance ou non ;
5. comparer la prédiction avec la vérité terrain.

---

## Scripts principaux

Les scripts principaux de cette étape sont :

- `scripts/make_mia_targets.py`
- `scripts/make_mia_targets_post_ano.py`
- `scripts/run_mia_attack.py`
- `scripts/generate_mia_attack_report.py`

### `make_mia_targets.py`
Prépare le split pré-anonymisation.

Il produit :

- un `published subset` qui sera anonymisé ;
- un `OUT holdout pool` qui restera hors publication ;
- un JSON de métadonnées de split.

### `make_mia_targets_post_ano.py`
Construit les cibles MIA finales après anonymisation.

Il ne garde les cibles IN que parmi les individus du `published subset` qui ont réellement survécu dans `anonymized_eval`.

### `run_mia_attack.py`
Exécute l'attaque elle-même.

### `generate_mia_attack_report.py`
Génère un rapport HTML à partir des sorties de l'attaque.

---

## Ce que cherche à prédire la MIA

La sortie principale de la MIA est une prédiction binaire :

- **IN** : la cible est prédite comme membre du dataset publié ;
- **OUT** : la cible est prédite comme non-membre.

Dans les fichiers, la vérité terrain est généralement codée ainsi :

- `is_member = 1` pour IN ;
- `is_member = 0` pour OUT.

---

## Différence avec la linkage attack

### Linkage attack
La linkage attack cherche surtout à savoir :

- quelles lignes anonymisées sont compatibles avec une cible ;
- ce qu'on peut en déduire sur l'attribut sensible.

### MIA
La MIA cherche surtout à savoir :

- si une cible faisait partie ou non du dataset publié.

Autrement dit :

- la linkage attack traite surtout le risque de liaison et d'inférence sensible ;
- la MIA traite le risque d'appartenance.

---

## Données utilisées par l'attaque

La MIA repose sur trois ensembles de données principaux.

### 1. Les cibles MIA

Ce fichier contient les individus testés par l'attaque.

Chaque ligne contient :

- `record_id` ;
- les attributs connus par l'attaquant ;
- le label `is_member`.

### 2. Le dataset anonymisé public

C'est la version visible par l'attaquant.

### 3. Le dataset anonymisé d'évaluation

Cette version est utilisée en interne pour :

- vérifier quels `record_id` ont survécu ;
- évaluer correctement les prédictions ;
- relier les étapes entre elles.

---

## Construction des cibles MIA

### Étape 1 : `make_mia_targets.py`

Le script suit globalement les étapes suivantes :

1. charger le dataset original ;
2. s'assurer que le dataset contient `record_id` ;
3. déterminer la taille de la future attacker base via `--attacker-frac` ou `--attacker-size` ;
4. en déduire un **OUT holdout pool** ;
5. conserver le reste comme **published subset** ;
6. écrire :
   - `*.published.csv`
   - `*.out.csv`
   - un JSON de métadonnées.

Le principe actuel est le suivant :

- la taille totale de l'attacker base est choisie ;
- elle est ensuite répartie de façon équilibrée entre OUT et IN ;
- la partie IN sera échantillonnée plus tard, après anonymisation.

### Étape 2 : `make_mia_targets_post_ano.py`

Ce script suit ensuite les étapes suivantes :

1. charger le `published subset` ;
2. charger le `OUT holdout pool` ;
3. charger `anonymized_eval` ;
4. récupérer les `record_id` qui ont réellement survécu ;
5. former le pool IN à partir de ces survivants ;
6. construire une attacker base équilibrée ;
7. échantillonner un nombre égal de cibles IN et OUT ;
8. écrire :
   - `targets_post_ano.csv`
   - `attacker_base.csv`
   - un JSON de métadonnées.

---

## Ce que sait l'attaquant

Comme pour la linkage attack, l'attaquant ne connaît qu'un sous-ensemble d'attributs.

Exemples fréquents :

- `age`
- `sex`
- `race`

Ces colonnes sont passées via `--known-qids`.

---

## Notion de compatibilité

La MIA repose elle aussi sur une logique de compatibilité.

Pour une cible donnée, on cherche les lignes du dataset anonymisé qui restent cohérentes avec les attributs connus.

### Exemple simple

Si l'attaquant connaît :

- `age = 27`
- `sex = Male`
- `race = White`

et qu'une ligne anonymisée contient :

- `age = [20-29]`
- `sex = Male`
- `race = White`

alors cette ligne est compatible avec la cible.

---

## Logique actuelle de filtrage

Comme pour la linkage attack, la MIA sépare les attributs connus en deux groupes selon `visible_level`.

### Stade 1
Attributs dont `visible_level != 0`.

Ils servent à construire la classe initiale à partir des valeurs généralisées visibles.

### Stade 2
Attributs dont `visible_level == 0`.

Ils servent à raffiner la classe initiale, soit :

- en exact ;
- soit éventuellement avec `privJedAI` si `--use-privjedai-fuzzy` est activé.

---

## Décision IN / OUT dans la version actuelle

Dans le code actuel, la décision finale est volontairement simple.

Elle repose principalement sur deux critères :

- `compatible_candidate_count`
- `compatible_candidate_fraction`

### Règle actuelle

La cible est prédite **IN** si :

1. il reste au moins un candidat compatible ;
2. la fraction compatible dans le dataset est inférieure ou égale à `max_compatible_fraction`.

Sinon, la cible est prédite **OUT**.

---

## Déroulement logique de `run_mia_attack.py`

Le script suit globalement les étapes suivantes.

### 1. Chargement des fichiers
Le script charge :

- la configuration runtime ;
- le fichier de cibles ;
- le dataset anonymisé public ;
- le dataset anonymisé d'évaluation.

### 2. Inférence des colonnes connues
Si besoin, il déduit les `known_qids` à partir du fichier de cibles.

### 3. Construction de `attacker_knowledge`
Comme pour la linkage, le script construit une vue attaquant des attributs connus et de leur `visible_level`.

### 4. Séparation stade 1 / stade 2
Les attributs connus sont répartis entre :

- `qid_filter_qids`
- `refine_qids`

### 5. Construction de la classe de stade 1
Le script conserve les lignes compatibles avec les attributs généralisés.

### 6. Raffinement final
Il réduit ensuite cette classe avec les attributs restés en clair.

### 7. Décision de membership
Il calcule :

- `compatible_candidate_count`
- `compatible_candidate_fraction`

puis applique la règle de décision IN/OUT.

### 8. Évaluation
Il calcule enfin :

- accuracy
- precision
- recall
- F1
- matrice de confusion
- statistiques séparées pour membres et non-membres

---

## Sorties produites

Chaque run de MIA produit généralement un dossier de la forme :

- `outputs/attacks/mia/<attack_id>/`

On y trouve notamment :

### `summary.json`
Résumé global de l'attaque.

### `targets.csv`
Résultats par cible.

### `<attack_id>__report.html`
Rapport HTML généré automatiquement si possible.

En plus, un résumé agrégé peut être ajouté dans :

- `outputs/attacks/mia/mia_summary.csv`

---

## Variables principales du `summary.json`

| Variable | Signification |
|---|---|
| `attack_id` | Identifiant du run MIA. |
| `known_qids` | Attributs connus par l'attaquant. |
| `qid_filter_qids` | Attributs utilisés au stade 1. |
| `refine_qids` | Attributs utilisés au stade 2. |
| `n_targets` | Nombre total de cibles évaluées. |
| `n_members` | Nombre de cibles IN dans la vérité terrain. |
| `n_non_members` | Nombre de cibles OUT dans la vérité terrain. |
| `max_compatible_fraction` | Seuil maximal de fraction compatible autorisé pour prédire IN. |
| `tp` | Nombre de vrais positifs : cibles IN correctement prédites IN. |
| `tn` | Nombre de vrais négatifs : cibles OUT correctement prédites OUT. |
| `fp` | Nombre de faux positifs : cibles OUT prédites IN. |
| `fn` | Nombre de faux négatifs : cibles IN prédites OUT. |
| `accuracy` | Part totale de prédictions correctes. |
| `precision` | Parmi les prédictions IN, part de celles qui sont correctes. |
| `recall` | Parmi les vraies cibles IN, part de celles retrouvées comme IN. |
| `f1` | Moyenne harmonique entre precision et recall. |
| `member_recall` | Recall restreint à la classe membre. |
| `non_member_true_negative_rate` | Taux de vrais négatifs sur la classe OUT. |
| `member_avg_stage1_equivalence_class_size` | Taille moyenne de la classe de stade 1 pour les membres. |
| `non_member_avg_stage1_equivalence_class_size` | Taille moyenne de la classe de stade 1 pour les non-membres. |
| `member_avg_compatible_candidate_count` | Nombre moyen de candidats compatibles finaux pour les membres. |
| `non_member_avg_compatible_candidate_count` | Nombre moyen de candidats compatibles finaux pour les non-membres. |
| `member_avg_equivalence_class_reduction` | Réduction moyenne entre stade 1 et stade final pour les membres. |
| `non_member_avg_equivalence_class_reduction` | Réduction moyenne entre stade 1 et stade final pour les non-membres. |

---

## Variables de complexité

Ces compteurs servent à estimer la quantité de travail logique effectuée par l'attaque.

| Variable | Signification |
|---|---|
| `row_index_row_visits` | Nombre de visites de lignes liées à l'index utilisé au stade de filtrage initial. |
| `targets_evaluated` | Nombre de cibles effectivement traitées. |
| `candidate_row_refs_loaded` | Nombre de références de lignes candidates chargées en mémoire pendant l'attaque. |
| `refinement_candidate_row_visits` | Nombre total de visites de candidats pendant la phase de raffinement. |
| `refinement_exact_tests` | Nombre de tests exacts réalisés pendant ce raffinement. |
| `refinement_fuzzy_tests` | Nombre de tests fuzzy réalisés lorsque l'option correspondante est activée. |
| `refinement_mask_cells` | Nombre de cellules manipulées dans les masques logiques pendant le raffinement. |
| `membership_decisions` | Nombre de décisions finales IN/OUT prises par l'attaque. |
| `estimated_total_operations` | Estimation globale du nombre d'opérations logiques réalisées. |

---

## Variables principales de `targets.csv`

| Variable | Signification |
|---|---|
| `target_id` | Identifiant interne de la cible. |
| `ground_truth_member` | Vérité terrain : 1 si IN, 0 si OUT. |
| `predicted_member` | Prédiction finale de l'attaque. |
| `qid_filter_qids` | Attributs utilisés au stade 1 pour cette attaque. |
| `refine_qids` | Attributs utilisés au stade 2 pour cette attaque. |
| `stage1_equivalence_class_size` | Taille de la classe obtenue au stade 1. |
| `compatible_candidate_count` | Nombre final de candidats compatibles. |
| `reduced_equivalence_class_size` | Même information que la taille finale de classe. |
| `equivalence_class_reduction` | Réduction absolue entre stade 1 et stade final. |
| `equivalence_class_reduction_rate` | Réduction relative entre stade 1 et stade final. |
| `compatible_candidate_fraction` | Fraction de lignes anonymisées restant compatibles. |
| `target_present_in_anonymized` | Indique si le `record_id` de la cible est réellement présent dans `anonymized_eval`. |
| `true_record_in_stage1_class` | Indique si le vrai record est présent dans la classe de stade 1. |
| `true_record_in_reduced_class` | Indique si le vrai record est encore présent après raffinement. |
| `decision_reason` | Trace textuelle de la logique de décision utilisée par le script. |

---

## Variables de configuration

| Variable | Signification |
|---|---|
| `attack_id` | Identifiant unique de l'expérience MIA. |
| `known_qids` | Attributs connus par l'attaquant. |
| `qid_filter_qids` | Attributs utilisés au stade 1 pour le premier filtrage. |
| `refine_qids` | Attributs utilisés au stade 2 pour affiner les candidats. |
| `target_id_col` | Nom de la colonne identifiant la cible. |
| `member_col` | Nom de la colonne portant le label de vérité terrain IN/OUT. |
| `max_compatible_fraction` | Seuil maximal de fraction compatible utilisé dans la logique de décision. |
| `use_privjedai_fuzzy` | Indique si un raffinement fuzzy a été activé. |
| `seed` | Graine aléatoire de reproductibilité. |


---


## Résumé

La MIA actuelle du projet repose donc sur une chaîne cohérente :

1. préparer un split avant anonymisation ;
2. reconstruire les cibles après anonymisation en ne gardant que les survivants pour les IN ;
3. construire une classe compatible ;
4. décider IN ou OUT à partir de la taille relative de cette classe.
