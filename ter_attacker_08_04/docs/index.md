# Documentation TER

## Objectif de cette documentation

Cette documentation présente le fonctionnement global du projet TER, depuis la préparation du dataset jusqu'à l'évaluation des attaques après anonymisation.

L'objectif est de comprendre :

- comment le dataset est préparé ;
- comment l'anonymisation est exécutée ;
- comment la linkage attack et la membership inference attack (MIA) sont construites ;
- quels fichiers sont produits à chaque étape.

---

## Contexte du projet

Le projet étudie l'effet de l'anonymisation d'un dataset sur la protection de la vie privée.

À partir d'un dataset source, on produit une version anonymisée, puis on évalue ce qu'un attaquant peut encore apprendre à partir des données publiées.

Le travail se concentre principalement sur trois blocs :

- l'anonymisation ;
- la **linkage attack** ;
- la **membership inference attack (MIA)**.

Les scripts de benchmark existent aussi dans le projet, mais ils servent surtout à automatiser des séries d'expériences.

---

## Organisation de la documentation

### Vue générale du projet
Présente le pipeline global, les grandes étapes et la logique d'enchaînement entre préparation, anonymisation et attaques.

### Anonymisation
Explique comment un dataset source est transformé en dataset anonymisé, avec une version publique et une version d'évaluation.

### Linkage attack
Décrit comment une base auxiliaire attaquant est utilisée pour construire des classes d'équivalence, les réduire, puis inférer un attribut sensible.

### Membership Inference Attack (MIA)
Décrit comment des cibles IN et OUT sont construites, puis comment l'attaque décide si une cible appartenait ou non au dataset publié.

### Structure des scripts
Donne une vue d'ensemble du dossier `scripts/` et du rôle des principaux fichiers.

### Formats des fichiers d'entrée / sortie
Présente les principales familles de fichiers manipulées par le projet et leur place dans le pipeline.

---

## Pipeline global

Le projet suit globalement la logique suivante :

1. préparer un dataset avec un identifiant interne stable `record_id` ;
2. exécuter une anonymisation ;
3. produire un dataset anonymisé public et un dataset anonymisé d'évaluation ;
4. préparer les fichiers nécessaires à la linkage attack et/ou à la MIA ;
5. exécuter les attaques ;
6. sauvegarder les résultats détaillés et les rapports HTML dans `outputs/`.

---

## Point important sur `record_id`

Le projet utilise un identifiant interne stable, généralement `record_id`, pour relier proprement les différentes étapes.

Cet identifiant :

- est utile pour l'évaluation interne ;
- est conservé dans `anonymized_eval` ;
- peut être retiré du dataset public avec `--public-drop-columns record_id`.

Il ne doit donc pas être considéré comme une information publiée à l'attaquant.

---

## Point important sur la MIA

Dans l'état actuel du projet, la MIA se fait en deux temps :

1. `make_mia_targets.py` prépare un **published subset** et un **OUT holdout pool** avant anonymisation ;
2. `make_mia_targets_post_ano.py` construit ensuite les **vraies cibles finales** après anonymisation, en ne prenant les cibles IN que parmi les enregistrements qui ont réellement survécu dans `anonymized_eval`.

Cette séparation est importante pour garder une vérité terrain cohérente lorsque certaines lignes sont supprimées de l'export final.

---

## Point important sur la linkage attack

Dans l'état actuel du projet, la linkage attack suit une logique en deux étages :

- **stade 1** : filtrage avec les attributs que l'attaquant voit sous forme généralisée ou supprimée dans le dataset publié ;
- **stade 2** : raffinement avec les attributs encore visibles en clair, soit en exact, soit éventuellement avec `privJedAI` en mode fuzzy.

La documentation détaillée de cette logique se trouve dans la page dédiée à la linkage attack.
