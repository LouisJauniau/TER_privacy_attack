# Formats des fichiers d'entrée / sortie

## Objectif de cette page

Cette page décrit les principaux fichiers manipulés par le projet.

L'objectif est de comprendre :

- quels fichiers servent d'entrée ;
- quels fichiers sont produits ;
- à quoi ils servent dans le pipeline ;
- quelle est la différence entre les versions publiques, les versions d'évaluation et les fichiers intermédiaires.

---

## Vue d'ensemble

Le projet manipule plusieurs familles de fichiers :

- les datasets source ;
- les datasets préparés avec `record_id` ;
- les fichiers de configuration ;
- les hiérarchies de généralisation ;
- les fichiers intermédiaires pour la linkage attack ;
- les fichiers intermédiaires pour la MIA ;
- les fichiers de sortie d'anonymisation ;
- les fichiers de sortie d'attaque ;
- les rapports HTML.

---

## 1. Datasets source et datasets préparés

### Dataset source

Exemples typiques :

- `data/adult.csv`
- `data/adult_with_record_id.csv`

#### Rôle
Point de départ des expériences.

#### Format
CSV avec en-tête.

---

### Dataset préparé avec `record_id`

Exemple typique :

- `data/adult_with_record_id.csv`

Métadonnées associées :

- `data/adult_with_record_id.json`

#### Rôle
Version stable du dataset utilisée par le pipeline courant.

#### Contenu
On y trouve :

- `record_id`
- les quasi-identifiants
- l'attribut sensible
- les autres colonnes du dataset

#### Pourquoi il est important
C'est ce fichier qui sert de base propre aux étapes suivantes.

---

## 2. Fichiers de configuration

### Configuration de base

Exemple :

- `configs/adult_base.json`

#### Rôle
Décrire une expérience d'anonymisation de départ.

---

### Configuration runtime exécutée

Dossier typique :

- `outputs/configs/`

Exemples possibles :

- `outputs/configs/qi_age-sex__k_5__l_2__t_None__supp_10__arx.json`
- `outputs/configs/adult.runtime.json`
- `outputs/configs/adult_base_with_record_id.json`

#### Rôle
Conserver exactement la configuration réellement utilisée pendant le run.

#### Format
JSON.

---

## 3. Hiérarchies de généralisation

Exemples typiques :

- `hierarchies/age.csv`
- `hierarchies/sex.csv`
- `hierarchies/race.csv`
- `hierarchies/native-country.csv`

#### Rôle
Décrire les niveaux successifs de généralisation pour les quasi-identifiants.

#### Format
CSV où chaque ligne relie une valeur source à ses niveaux de généralisation.

---

## 4. Fichiers produits par l'anonymisation

### Dataset anonymisé public

Dossier typique :

- `outputs/anonymized/`

Exemple :

- `outputs/anonymized/<experiment_id>.csv`

#### Rôle
Représenter la version publiée du dataset.

#### Point important
Des colonnes comme `record_id` peuvent être retirées de cette version.

---

### Dataset anonymisé d'évaluation

Dossier typique :

- `outputs/anonymized_eval/`

Exemple :

- `outputs/anonymized_eval/<experiment_id>.csv`

#### Rôle
Conserver une version interne permettant de relier les lignes à `record_id`.

#### Point important
Cette version n'est pas censée être visible par l'attaquant.

---

### Métriques d'anonymisation

Dossier typique :

- `outputs/metrics/`

Exemple :

- `outputs/metrics/<experiment_id>.json`

#### Rôle
Résumer le résultat d'un run d'anonymisation.

#### Contenu typique
On peut y trouver :

- les chemins des exports ;
- les colonnes retirées du public ;
- les statistiques de suppression ;
- le nombre de lignes retirées parce que tous les QI valaient `*`.

---

### Résumé CSV des runs

Fichier typique :

- `outputs/benchmark_summary.csv`

#### Rôle
Agréger plusieurs runs d'anonymisation dans un même tableau.

---

## 5. Fichiers de préparation de la linkage attack

### Base auxiliaire

Dossier typique :

- `outputs/auxiliary/`

Exemple de nom par défaut :

- `outputs/auxiliary/<dataset>__aux__known_<attrs>__released_only__n_<n>.csv`
- `outputs/auxiliary/<dataset>__aux__known_<attrs>__all_records__n_<n>.csv`

#### Rôle
Contenir les cibles et les attributs connus par l'attaquant.

#### Format
CSV.

---

### Métadonnées de la base auxiliaire

Même nom de base, avec extension `.json`.

Exemple :

- `outputs/auxiliary/<...>.json`

#### Rôle
Conserver :

- le dataset source utilisé ;
- les attributs connus ;
- la taille de l'échantillon ;
- le mode de population cible ;
- éventuellement le lien avec `released_eval`.

---

## 6. Fichiers de sortie de la linkage attack

Chaque run produit généralement un dossier :

- `outputs/attacks/linkage/<attack_id>/`

### `summary.json`

#### Rôle
Résumé global de l'attaque.

#### Contient notamment
- la liste des attributs connus ;
- la séparation stade 1 / stade 2 ;
- les tailles de classes ;
- les métriques d'inférence sensible ;
- `operation_counter`.

---

### `targets.csv`

#### Rôle
Résultats détaillés par cible.

#### Contenu typique
- taille de classe de stade 1 ;
- taille de classe finale ;
- réduction ;
- présence du vrai record ;
- distribution de l'attribut sensible.

---

### `equivalence_class_candidates.csv`

#### Rôle
Lister les candidats finaux retenus dans les classes d'équivalence.

#### Format
Une ligne par couple :

- cible ;
- candidat final.

---

### `attacker_knowledge.json`

#### Rôle
Décrire la vision attaquant des attributs connus.

#### Contenu typique
Pour chaque attribut :
- `visible_level`
- `observed_values`
- `projection`

---

### `prefilter_debug/` (optionnel)

#### Rôle
Conserver des exports de debug intermédiaires pour certaines cibles.

---

### Rapport HTML

Nom typique :

- `outputs/attacks/linkage/<attack_id>/<attack_id>__report.html`

#### Rôle
Présenter les principaux résultats sous une forme lisible.

---

### Résumé agrégé des attaques linkage

Fichier typique :

- `outputs/attacks/linkage/linkage_summary.csv`

#### Rôle
Agréger plusieurs runs de linkage dans un même tableau.

---

## 7. Fichiers de préparation de la MIA

### Published subset

Dossier typique :

- `outputs/prepared_data/`

Exemple :

- `outputs/prepared_data/<name>.published.csv`

#### Rôle
Sous-ensemble du dataset original qui sera anonymisé pour la MIA.

---

### OUT holdout pool

Dossier typique :

- `outputs/prepared_data/`

Exemple :

- `outputs/prepared_data/<name>.out.csv`

#### Rôle
Pool d'individus absents du dataset publié.

---

### Métadonnées de split MIA

Exemple :

- `outputs/prepared_data/<name>.published.json`

#### Rôle
Conserver :

- le chemin du published subset ;
- le chemin du OUT pool ;
- la taille attendue du pool IN ;
- la seed ;
- la taille de l'attacker knowledge base.

---

### Attacker base MIA

Exemple :

- `outputs/prepared_data/<name>.attacker_base.csv`

#### Rôle
Base de connaissance attaquant construite après anonymisation, contenant à la fois des candidats OUT et des candidats IN survivants.

---

### Cibles MIA post-anonymisation

Dossier typique :

- `outputs/mia_targets/`

Exemple :

- `outputs/mia_targets/<name>.targets_post_ano.csv`

Métadonnées associées :

- `outputs/mia_targets/<name>.targets_post_ano.json`

#### Rôle
Contenir les cibles finales équilibrées IN/OUT avec le label `is_member`.

---

## 8. Fichiers de sortie de la MIA

Chaque run produit généralement un dossier :

- `outputs/attacks/mia/<attack_id>/`

### `summary.json`

#### Rôle
Résumé global de l'attaque MIA.

#### Contient notamment
- les `known_qids` ;
- les attributs de stade 1 et de stade 2 ;
- la matrice de confusion ;
- accuracy, precision, recall, F1 ;
- les tailles moyennes de classes.

---

### `targets.csv`

#### Rôle
Résultats détaillés par cible.

#### Contenu typique
- vérité terrain ;
- prédiction finale ;
- taille de classe de stade 1 ;
- nombre final de candidats compatibles ;
- fraction compatible ;
- raison textuelle de la décision.

---

### Rapport HTML

Nom typique :

- `outputs/attacks/mia/<attack_id>/<attack_id>__report.html`

#### Rôle
Présenter les résultats de la MIA sous une forme lisible.

---

### Résumé agrégé des MIA

Fichier typique :

- `outputs/attacks/mia/mia_summary.csv`

#### Rôle
Agréger plusieurs runs MIA dans un même tableau.

---

## 9. Différence entre fichiers publics, internes et intermédiaires

### Fichiers publics
Ils représentent la vue plausible de l'attaquant.

Exemples :
- `outputs/anonymized/...`

### Fichiers internes d'évaluation
Ils servent à évaluer les attaques sans être publiés.

Exemples :
- `outputs/anonymized_eval/...`

### Fichiers intermédiaires
Ils servent à préparer les attaques ou à documenter le pipeline.

Exemples :
- `outputs/prepared_data/...`
- `outputs/auxiliary/...`
- `outputs/mia_targets/...`

---

## Résumé

Le projet manipule donc plusieurs niveaux de fichiers :

1. les fichiers source ;
2. les fichiers préparés ;
3. les fichiers d'anonymisation ;
4. les fichiers de préparation des attaques ;
5. les résultats détaillés des attaques ;
6. les rapports HTML.

Comprendre la place de chaque famille de fichiers est essentiel pour bien suivre le pipeline complet.
