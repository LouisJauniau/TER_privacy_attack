# Linkage attack

## Rôle de cette étape

La **linkage attack** cherche à relier une cible connue par l'attaquant à une ou plusieurs lignes du dataset anonymisé publié.

Dans ce projet, son rôle n'est pas seulement de rechercher une ré-identification exacte.
Elle sert aussi à mesurer ce qu'un attaquant peut encore apprendre sur l'**attribut sensible** à partir des candidats compatibles restants.

---

## Idée générale

L'attaque part d'une hypothèse réaliste :

- l'attaquant dispose d'une base auxiliaire ;
- il connaît certains attributs sur une cible ;
- il compare cette connaissance avec le dataset anonymisé publié.

Pour chaque cible, l'attaque construit une **classe d'équivalence attaquant**, c'est-à-dire l'ensemble des lignes anonymisées encore compatibles avec les informations connues.

Ensuite, elle essaie de réduire cette classe autant que possible, puis observe les valeurs de l'attribut sensible sur les candidats restants.

---

## Scripts principaux

Les scripts les plus importants pour cette étape sont :

- `scripts/make_auxiliary_base.py`
- `scripts/run_linkage_attack.py`
- `scripts/generate_linkage_attack_report.py`

### `make_auxiliary_base.py`
Prépare la base auxiliaire de l'attaquant à partir d'un dataset contenant déjà `record_id`.

### `run_linkage_attack.py`
Exécute l'attaque, produit les fichiers détaillés, le `summary.json` et, si possible, le rapport HTML.

### `generate_linkage_attack_report.py`
Génère un rapport HTML à partir des sorties de l'attaque.

---

## Données utilisées par l'attaque

La linkage attack repose sur trois ensembles de données principaux.

### 1. La base auxiliaire de l'attaquant

Elle contient les informations connues par l'attaquant sur les cibles.

Exemples fréquents :

- `age`
- `sex`
- `race`
- `marital-status`
- `native-country`

### 2. Le dataset anonymisé public

C'est la vue supposée réaliste de l'attaquant.

### 3. Le dataset anonymisé d'évaluation

Cette version est utilisée uniquement en interne pour :

- retrouver `record_id` ;
- vérifier si le vrai record a survécu ;
- récupérer la vraie valeur sensible ;
- mesurer la qualité réelle de l'attaque.


---

## Ce que sait l'attaquant

L'attaquant ne connaît qu'un sous-ensemble d'attributs.

Le script accepte cette liste via `--known-attrs`.

Cette connaissance partielle est ensuite interprétée à travers un objet interne important :

- `attacker_knowledge.json`

Ce fichier décrit, pour chaque attribut connu :

- le `visible_level` observé dans le dataset publié ;
- les valeurs observées dans la version anonymisée ;
- la projection entre valeur brute attaquant et valeur visible côté publication.

---

## Notion de `visible_level`

Le `visible_level` indique jusqu'à quel niveau un attribut reste visible dans les données publiées.

### `visible_level != 0`
L'attribut apparaît sous une forme généralisée ou supprimée dans la publication.

Il sert donc à construire la **classe initiale**.

### `visible_level == 0`
L'attribut reste visible en clair dans la publication.

Il peut alors servir à **réduire** la classe construite au stade précédent.

Cette séparation est centrale dans la logique actuelle de l'attaque.

---

## Déroulement logique de l'attaque

### Stade 1 : construction de la classe initiale

Le script prend les attributs connus dont `visible_level != 0`.

Pour chacun d'eux, il projette la valeur brute de la cible vers la valeur visible côté attaquant.

Exemple :

- valeur brute cible : `United-States`
- valeur visible publiée : `North America`

Puis il conserve uniquement les lignes du dataset anonymisé compatibles avec toutes ces valeurs visibles.

Le résultat est une première classe d'équivalence.

### Stade 2 : raffinement avec les attributs en clair

Le script prend ensuite les attributs connus dont `visible_level == 0`.

Sur cette classe initiale, il essaie de réduire encore les candidats :

- soit en match exact ;
- soit, si activé, avec `privJedAI` en mode fuzzy.

Le résultat est la **classe finale**.

---

## Réutilisation des classes de stade 1

Le script ne recalcule pas systématiquement la classe initiale pour chaque cible.

Si plusieurs cibles ont exactement la même projection sur les attributs du stade 1, elles réutilisent la même classe de départ via un cache.

C'est ce qui explique la variable :

- `n_distinct_stage1_groups`

Cette optimisation a deux effets :

- elle accélère l'attaque ;
- elle influence les compteurs de complexité.

---

## Compatibilité et raffinements possibles

### Match exact
Utilisé lorsque l'attribut de raffinement est resté visible tel quel dans le dataset publié.

### Match fuzzy avec `privJedAI`
Optionnel, activé avec `--use-privjedai-fuzzy`.

Ce mode est utile lorsque deux valeurs devraient être considérées comme proches malgré des variations de forme.

Il ne remplace pas le stade 1 : il intervient uniquement pendant le raffinement des attributs restés en clair.

---

## Inférence de l'attribut sensible

Une fois la classe finale obtenue, l'attaque regarde les valeurs de l'attribut sensible parmi les candidats.

### Cas 1 : classe vide
Aucun candidat compatible ne reste.

### Cas 2 : un seul candidat
La cible est ré-identifiée de manière unique dans le cadre de l'attaque.

### Cas 3 : plusieurs candidats, une seule valeur sensible
L'attribut sensible est inféré avec certitude, même sans ré-identification unique.

### Cas 4 : plusieurs candidats, plusieurs valeurs sensibles
Le script produit une distribution de probabilité à partir des fréquences observées.

Exemple :

- `<=50K` : 75 %
- `>50K` : 25 %

---

## Sorties produites

Chaque run de linkage produit généralement un dossier de la forme :

- `outputs/attacks/linkage/<attack_id>/`

On y trouve notamment :

### `summary.json`
Résumé global de l'attaque.

### `targets.csv`
Résultats par cible.

### `equivalence_class_candidates.csv`
Une ligne par candidat final conservé dans les classes d'équivalence.

### `attacker_knowledge.json`
Description de la visibilité des attributs côté attaquant.

### `prefilter_debug/` (optionnel)
Fichiers de debug détaillant les lignes gardées aux différents stades.

### `<attack_id>__report.html`
Rapport HTML généré automatiquement si la génération réussit.

En plus, un résumé agrégé peut être ajouté dans :

- `outputs/attacks/linkage/linkage_summary.csv`

---

## Déroulement logique de `make_auxiliary_base.py`

Le script suit globalement les étapes suivantes :

1. charger un dataset déjà préparé avec `record_id` ;
2. sélectionner les colonnes connues par l'attaquant ;
3. éventuellement ne garder que les individus encore présents dans `anonymized_eval` ;
4. échantillonner les cibles ;
5. produire un CSV auxiliaire et un JSON de métadonnées.

---

## Déroulement logique de `run_linkage_attack.py`

Le script suit globalement les étapes suivantes :

1. charger la configuration runtime ;
2. charger la base auxiliaire ;
3. charger `anonymized` et `anonymized_eval` ;
4. construire `attacker_knowledge` ;
5. séparer les attributs entre stade 1 et stade 2 selon `visible_level` ;
6. construire ou réutiliser les classes initiales de stade 1 ;
7. appliquer le raffinement de stade 2 ;
8. mesurer la taille des classes finales ;
9. inférer l'attribut sensible ;
10. sauvegarder les artefacts et le rapport.

---

## Variables principales du `summary.json`

| Variable | Signification |
|---|---|
| `attack_id` | Identifiant unique du run de linkage. |
| `known_attrs` | Attributs connus par l'attaquant. |
| `qid_filter_attrs` | Attributs utilisés au stade 1, donc vus comme généralisés ou supprimés. |
| `refine_attrs` | Attributs utilisés au stade 2, donc restés visibles en clair. |
| `n_targets` | Nombre de cibles attaquées. |
| `n_anonymized_rows` | Nombre de lignes présentes dans le dataset anonymisé utilisé par l'attaque. |
| `n_distinct_stage1_groups` | Nombre de groupes distincts au stade 1 après mise en cache. |
| `unique_reidentification_rate` | Part des cibles dont la classe finale contient exactement un seul candidat correspondant à la vraie cible. |
| `false_unique_match_rate` | Part des cibles pour lesquelles la classe finale contient un seul candidat, mais pas le vrai record. |
| `true_record_kept_after_refinement_rate` | Part des cibles pour lesquelles le vrai record est encore présent après le raffinement final. |
| `avg_qid_equivalence_class_size` | Taille moyenne des classes au stade 1. |
| `median_qid_equivalence_class_size` | Médiane de la taille des classes au stade 1. |
| `avg_equivalence_class_size` | Taille moyenne des classes finales. |
| `median_equivalence_class_size` | Médiane de la taille des classes finales. |
| `max_equivalence_class_size` | Plus grande classe finale observée. |
| `avg_reduction_rate` | Réduction moyenne entre la taille de classe de stade 1 et la taille finale. |
| `certainty_sensitive_inference_rate` | Part des cibles pour lesquelles l'attribut sensible est inféré avec certitude. |
| `avg_true_sensitive_probability` | Probabilité moyenne de la vraie valeur sensible. |
| `median_true_sensitive_probability` | Médiane de la probabilité de la vraie valeur sensible. |
| `avg_top_sensitive_probability` | Probabilité moyenne de la valeur sensible la plus probable dans la classe finale. |

---

## Variables principales de `targets.csv`

| Variable | Signification |
|---|---|
| `target_id` | Identifiant interne de la cible. |
| `qid_equivalence_class_size` | Taille de la classe construite au stade 1. |
| `reduced_equivalence_class_size` | Taille de la classe finale après raffinement. |
| `equivalence_class_reduction` | Réduction absolue entre les deux tailles. |
| `equivalence_class_reduction_rate` | Réduction relative entre les deux tailles. |
| `true_record_in_qid_class` | Indique si le vrai record était présent au stade 1. |
| `true_record_in_reduced_class` | Indique si le vrai record est encore présent après raffinement. |
| `unique_reidentified` | Indique si la cible est ré-identifiée de manière unique et correcte. |
| `false_unique_match` | Indique si un unique candidat incorrect a été obtenu. |
| `true_sensitive_value` | Vraie valeur de l'attribut sensible. |
| `predicted_sensitive_top_value` | Valeur sensible la plus probable dans la classe finale. |
| `predicted_sensitive_top_probability` | Probabilité de cette valeur sensible majoritaire. |
| `true_sensitive_probability` | Probabilité attribuée à la vraie valeur sensible. |
| `sensitive_value_certain` | Indique si l'inférence sensible est certaine. |
| `n_distinct_sensitive_values` | Nombre de valeurs sensibles distinctes dans la classe finale. |

---

## Variables de complexité

Ces compteurs donnent une **estimation de la quantité de travail logique** réalisée par l'attaque.

| Variable | Signification |
|---|---|
| `value_index_row_visits` | Nombre de passages sur des lignes du dataset anonymisé lors de la construction ou de l'utilisation de l'index de valeurs pour le filtrage. |
| `targets_evaluated` | Nombre de cibles effectivement évaluées. |
| `array_cells_initialized` | Nombre total de cellules de tableaux temporaires initialisées pendant le traitement des cibles. C'est un indicateur du coût mémoire et du coût de préparation des masques. |
| `attribute_positive_mask_cells` | Nombre de cellules marquées positivement dans les masques d'attributs, c'est-à-dire les positions reconnues comme compatibles pour un attribut donné. |
| `matching_row_visits` | Nombre de visites de lignes pendant le filtrage initial par compatibilité. |
| `mask_and_cells` | Nombre d'opérations logiques de type AND effectuées entre masques pour combiner plusieurs contraintes d'attributs. |
| `final_mask_reads` | Nombre de lectures du masque final pour extraire les candidats conservés. |
| `equivalence_class_candidate_rows_output` | Nombre total de candidats produits à la sortie de l'étape de filtrage QI, en additionnant toutes les cibles. |
| `refinement_candidate_row_visits` | Nombre total de visites de candidats pendant la phase de réduction après le filtrage QI. |
| `refinement_exact_tests` | Nombre de tests exacts réalisés pendant cette phase de réduction. |
| `refinement_fuzzy_tests` | Nombre de tests fuzzy réalisés, par exemple via `privJedAI`, lorsque cette option est activée. |
| `refinement_mask_cells` | Nombre de cellules manipulées dans les masques de la phase de réduction. |
| `reduced_equivalence_class_candidate_rows_output` | Nombre total de candidats restants après réduction, en additionnant toutes les cibles. |
| `estimated_total_operations` | Estimation globale du nombre d'opérations logiques réalisées par l'attaque. |

---

## Variables de configuration

| Variable | Signification |
|---|---|
| `attack_id` | Identifiant unique de l'attaque et du dossier de sortie. |
| `known_attrs` | Attributs connus par l'attaquant. |
| `qid_filter_attrs` | Attributs utilisés pour construire la classe d'équivalence initiale. En pratique, ce sont les attributs généralisés exploités au stade 1. |
| `refine_attrs` | Attributs utilisés pour réduire la classe obtenue au stade QI. En pratique, ce sont souvent des attributs non généralisés ou testés plus directement. |
| `target_id_col` | Nom de la colonne identifiant les cibles. |
| `sensitive_attr` | Attribut sensible que l'attaque cherche à inférer. |
| `n_anonymized_rows` | Nombre de lignes du dataset anonymisé effectivement utilisées par l'attaque. |
| `use_privjedai_fuzzy` | Indique si des comparaisons fuzzy ont été activées pendant la réduction. |
| `seed` | Graine aléatoire utilisée pour rendre l'expérience reproductible. |

---


## Résumé

La linkage attack actuelle du projet repose donc sur une logique précise :

1. construire une classe initiale avec les attributs vus comme généralisés ;
2. la réduire avec les attributs restés visibles en clair ;
3. mesurer ce que cette classe finale révèle encore sur l'attribut sensible.
