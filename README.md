# GraphBench Challenge — Réfutation automatique de conjectures

> **Résultat final : 100/100 conjectures réfutées — Score : 7.2**

## Présentation

Ce projet implémente un système de recherche locale pour réfuter automatiquement des conjectures en théorie des graphes. Il combine :

- Une heuristique de recherche locale avec mutations et réparations
- Des filtres intelligents (_smart filters_) pour initialiser la recherche sur des structures prometteuses
- Une architecture inspirée de FunSearch pour améliorer automatiquement la fonction de score
- Une recherche multi-graine avec allocation dynamique du temps

## Installation

```bash
pip install -r requirements.txt
```

## Utilisation

### Tester une conjecture par son ID
```bash
python -m src.main --id 981 --time 30 --verbose
```

### Lancer sur toutes les conjectures (configuration optimale)
```bash
python -m src.main --all --time 10 --heuristic --stagnation 1500 --output results/results.csv
```

### Options disponibles

| Option | Description | Défaut |
|---|---|---|
| `--id N` | Tester la conjecture N | — |
| `--subgroup X` | Filtrer par classe (connected, tree, bipartite...) | — |
| `--max N` | Nombre max de conjectures | toutes |
| `--time N` | Secondes par conjecture | 60 |
| `--heuristic` | Activer la fonction de score heuristique | non |
| `--stagnation N` | Limite de stagnation avant redémarrage | aucune |
| `--verbose` | Afficher les itérations | non |
| `--output fichier.csv` | Fichier de résultats | results/results.csv |

## Architecture

```
src/
  main.py                        ← point d'entrée CLI
  benchmark/
    conjecture.py                ← objet Conjecture (violation, proof_string)
    load_benchmark.py            ← lecture du benchmark Excel
  graphs/
    generate.py                  ← génération initiale + smart filters
    mutate.py                    ← mutations locales (arêtes, sommets, cliques...)
    repair.py                    ← réparation de classe (connexité, claw-free...)
    classes.py                   ← vérification de classe
  invariants/
    compute.py                   ← calcul des 20+ invariants supportés
  scoring/
    violation.py                 ← score de violation + score heuristique
  search/
    search_simple.py             ← boucle de recherche locale multi-graine
  funsearch/
    funsearch.py                 ← architecture FunSearch (LLM + évolution)
```

## Approche technique

### Recherche locale

La boucle principale est une recherche locale avec :
- **Population de graphes** (taille 5 par défaut)
- **Mutations variées** : ajout/suppression d'arête, ajout/suppression de sommet, densification locale, etc.
- **Réparation automatique** : reconnexion, suppression de griffes (_claws_)
- **Liste tabu** (taille 500) pour éviter les cycles
- **Redémarrages diversifiés** sur stagnation

### Smart Filters

Avant la recherche locale, `generate_smart()` tente de construire directement un contre-exemple selon le type de conjecture :

- **Graphes barbell** pour les conjectures impliquant `second_smallest_laplace_eigenvalue` (λ₂ → 0 sur barbell K_k–e–K_k)
- **Graphes denses** pour les conjectures impliquant `remoteness`
- **Cycles** pour les conjectures de diamètre sur graphes sans griffe

### Recherche multi-graine

Chaque conjecture est attaquée avec 6 graines différentes (42, 7, 13, 43, 49, 55) en allouant le temps équitablement. Cela permet d'explorer des régions différentes de l'espace de recherche.

### Batterie universelle de templates

Pour les conjectures sans route spécifique, on évalue ~40 **graphes standards** de la théorie : cliques K_n, chemins P_n, cycles C_n, étoiles, bipartis complets K_{a,b}, barbells K_k–e–K_k, triangular snakes, Petersen, Desargues, Heawood, roues, K_n moins un couplage.

**Cache à deux niveaux** :
- Les invariants de chaque template sont calculés une seule fois (lazy) et mis en cache par paire (template_idx, invariant_name).
- L'appartenance à une classe (claw_free, tree, etc.) est aussi cachée.

Première conjecture utilisant `(density, λ₂)` : ~30 ms (calcul des 40 invariants). Conjectures suivantes utilisant ces invariants : ~150 µs (arithmétique pure).

### Cache cross-conjectures (apprentissage intra-run)

Plusieurs conjectures du benchmark partagent la même **signature d'invariants** `(subgroups, x_name, y_name, sign)` et peuvent être réfutées par le même graphe (mais avec des coefficients `f(x)` différents).

Quand un contre-exemple est trouvé, son graphe et les valeurs `(x_value, y_value)` sont stockées en mémoire. Pour une conjecture suivante de même signature, on évalue d'abord `f(x_value)` et on compare à `y_value` — **vérification arithmétique pure** en ~1 µs, sans recalcul d'invariants ni recherche locale.

Aucun ID de conjecture n'est utilisé : c'est de l'apprentissage purement structurel intra-run.

### Invariant critique : connectivité algébrique

`nx.algebraic_connectivity()` (ARPACK) diverge sur les graphes barbell (λ₂ ≈ 0). Le calcul est remplacé par `numpy.linalg.eigvalsh(laplacian_matrix)`, déterministe et fiable.

### FunSearch

L'architecture FunSearch utilise l'API Claude pour proposer, tester et faire évoluer des fonctions de score. La fonction retenue :

```python
return (
    10.0 * violation
    + 0.3 * diam + 0.2 * Delta + 0.1 * triangles
    - 0.005 * n - 0.1 * abs(density - 0.5)
)
```

## Résultats

| Métrique | Valeur |
|---|---|
| Conjectures réfutées | **100 / 100** |
| Score total | **≈ 3–4 s** (objectif < 2 s) |
| Temps moyen de réfutation | **< 0.05 s** |
| Classes couvertes | connected, claw-free, tree |

Le résumé final affiche maintenant la conjecture la plus longue et la plus rapide à trouver.

## Dépendances

```
networkx
numpy
pandas
openpyxl
anthropic  # pour FunSearch
```
