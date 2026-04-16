# Structure des scripts

## Objectif de cette page

Cette page donne une vue claire de l'organisation des scripts du projet, pour comprendre rapidement :

- quels fichiers jouent un rôle central ;
- quels scripts servent à lancer les étapes principales ;
- quels fichiers servent de support ;
- quels scripts servent surtout à automatiser ou à générer des rapports.

---

## Vue d'ensemble

Le dossier `scripts/` contient principalement :

1. les scripts de préparation du dataset ;
2. les scripts d'exécution principaux ;
3. les scripts de préparation des attaques ;
4. les fichiers utilitaires partagés ;
5. les scripts de benchmark ;
6. les scripts de génération de rapports.

---

## Scripts principaux

Les scripts les plus importants dans l'état actuel du projet sont :

- `prepare_dataset_with_record_id.py`
- `run_ano.py`
- `make_auxiliary_base.py`
- `run_linkage_attack.py`
- `make_mia_targets.py`
- `make_mia_targets_post_ano.py`
- `run_mia_attack.py`

Ce sont eux qui correspondent directement au pipeline principal documenté dans les autres pages.

---

## 1. Préparation du dataset

### `prepare_dataset_with_record_id.py`

C'est le script de préparation le plus important avant anonymisation.

#### Rôle
- ajouter `record_id` si nécessaire ;
- vérifier son unicité ;
- produire une version stable du dataset ;
- éventuellement générer une copie de configuration mise à jour.

#### Pourquoi il est important
Le pipeline actuel s'appuie fortement sur `record_id` pour relier :

- l'anonymisation ;
- la linkage attack ;
- la MIA ;
- l'évaluation interne.

---

## 2. Scripts d'exécution principaux

### `run_ano.py`

C'est le point d'entrée principal pour l'anonymisation.

#### Rôle
- charger une configuration d'expérience ;
- préparer la configuration runtime ;
- lancer l'anonymisation ;
- sauvegarder les sorties produites.

#### Sorties typiques
- configuration exécutée ;
- dataset anonymisé public ;
- dataset anonymisé d'évaluation ;
- métriques.

---

### `run_linkage_attack.py`

C'est le point d'entrée principal pour exécuter la linkage attack.

#### Rôle
- charger la base auxiliaire ;
- charger les datasets anonymisés ;
- construire `attacker_knowledge` ;
- séparer les attributs entre stade 1 et stade 2 ;
- construire les classes d'équivalence ;
- inférer l'attribut sensible ;
- sauvegarder les résultats et le rapport HTML.

#### Particularité actuelle
Le script applique une logique en deux étages basée sur `visible_level`.

---

### `run_mia_attack.py`

C'est le point d'entrée principal pour exécuter la membership inference attack.

#### Rôle
- charger les cibles MIA ;
- charger les datasets anonymisés ;
- construire `attacker_knowledge` ;
- séparer les attributs entre stade 1 et stade 2 ;
- calculer les candidats compatibles ;
- prédire IN ou OUT ;
- sauvegarder les résultats et le rapport HTML.

---

## 3. Scripts de préparation des données d'attaque

### `make_auxiliary_base.py`

Prépare la base auxiliaire utilisée par la linkage attack.

#### Rôle
- partir d'un dataset déjà préparé avec `record_id` ;
- sélectionner les attributs connus par l'attaquant ;
- échantillonner les individus ;
- éventuellement ne garder que les individus encore publiés via `--released-eval`.

#### Pourquoi il est important
Dans la version actuelle, la linkage attack est utilisée en mode strict : les cibles doivent en pratique être encore présentes dans `anonymized_eval`.

---

### `make_mia_targets.py`

Prépare le split pré-anonymisation pour la MIA.

#### Rôle
- partir du dataset original ;
- créer un `published subset` ;
- créer un `OUT holdout pool` ;
- écrire un JSON de métadonnées de split ;
- éventuellement produire une config retargetée vers le subset publié.

#### Point important
Ce script ne crée plus directement les cibles finales IN/OUT.

---

### `make_mia_targets_post_ano.py`

Prépare les cibles finales de la MIA après anonymisation.

#### Rôle
- lire le `published subset` ;
- lire le `OUT holdout pool` ;
- lire `anonymized_eval` ;
- identifier les survivants ;
- construire l'attacker base ;
- construire des cibles équilibrées IN/OUT.

#### Pourquoi il est important
Il garantit que les cibles IN correspondent réellement à des enregistrements encore présents dans l'export final.

---

## 4. Fichiers utilitaires partagés

### `common.py`

Boîte à outils générale du projet.

#### Rôle
- gestion des chemins ;
- lecture et écriture JSON ;
- création de dossiers ;
- fonctions utilitaires réutilisées dans plusieurs scripts.

---

### `attack_common.py`

Utilitaires communs aux attaques.

#### Rôle
- lecture normalisée des CSV ;
- chargement de la configuration runtime ;
- construction de `attacker_knowledge` ;
- inférence de `visible_level` ;
- helpers partagés de validation et d'écriture de résumés.

---

### `linkage_helpers.py`

Helpers spécifiques à la linkage attack.

#### Rôle
- construction d'index de valeurs ;
- logique de compatibilité ;
- raffinement exact ou fuzzy ;
- résumé des distributions de l'attribut sensible.

---

### `privjedai_utils.py`

Couche d'intégration avec `privJedAI`.

#### Rôle
- configuration du mode fuzzy ;
- calcul de similarité ;
- support des comparaisons Bloom-filter pour les attributs restés en clair.

---

## 5. Scripts de génération de rapports

### `generate_linkage_attack_report.py`

Génère un rapport HTML pour une attaque de linkage.

#### Rôle
- lire `summary.json` ;
- lire `targets.csv` ;
- éventuellement lire config et métriques d'anonymisation ;
- produire un rapport HTML synthétique.

---

### `generate_mia_attack_report.py`

Génère un rapport HTML pour une MIA.

#### Rôle
- lire `summary.json` ;
- lire `targets.csv` ;
- calculer les indicateurs de classification ;
- produire un rapport HTML.

---

## 6. Scripts de benchmark

### `run_benchmark.py`
Automatise plusieurs runs d'anonymisation.

### `run_linkage_benchmark.py`
Automatise des séries d'attaques de linkage.

### `run_mia_benchmark.py`
Automatise des séries d'attaques MIA.

Ces scripts sont utiles pour produire plusieurs expériences comparables, mais ils ne constituent pas le cœur logique des attaques elles-mêmes.

---

## Organisation logique

### Bloc préparation du dataset
- `prepare_dataset_with_record_id.py`

### Bloc anonymisation
- `run_ano.py`

### Bloc préparation linkage
- `make_auxiliary_base.py`

### Bloc attaque linkage
- `run_linkage_attack.py`
- `linkage_helpers.py`
- `privjedai_utils.py`

### Bloc préparation MIA
- `make_mia_targets.py`
- `make_mia_targets_post_ano.py`

### Bloc attaque MIA
- `run_mia_attack.py`

### Bloc utilitaires partagés
- `common.py`
- `attack_common.py`

### Bloc rapports
- `generate_linkage_attack_report.py`
- `generate_mia_attack_report.py`

### Bloc benchmarks
- `run_benchmark.py`
- `run_linkage_benchmark.py`
- `run_mia_benchmark.py`

---

## Résumé

La structure actuelle des scripts reflète une organisation en pipeline :

1. préparer le dataset ;
2. anonymiser ;
3. préparer les entrées des attaques ;
4. exécuter les attaques ;
5. générer des rapports.

La principale évolution récente du projet concerne surtout la MIA, qui est maintenant séparée en une phase pré-anonymisation et une phase post-anonymisation.
