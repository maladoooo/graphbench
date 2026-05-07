# GraphBench — Réfutation automatique de conjectures

## Installation

```bash
pip install -r requirements.txt
```

## Utilisation

Toutes les commandes se lancent depuis le dossier `graphbench-project/`.

### Tester une conjecture par son ID
```bash
python -m src.main --id 981 --time 30 --verbose
```

### Tester les N premières conjectures "connected"
```bash
python -m src.main --subgroup connected --max 5 --time 60
```

### Lancer sur toutes les conjectures
```bash
python -m src.main --all --time 60 --output results/results.csv
```

## Options disponibles

| Option | Description | Défaut |
|---|---|---|
| `--id N` | Tester la conjecture N | — |
| `--subgroup X` | Filtrer par classe (connected, tree, bipartite...) | — |
| `--max N` | Nombre max de conjectures | toutes |
| `--time N` | Secondes par conjecture | 60 |
| `--verbose` | Afficher les itérations | non |
| `--output fichier.csv` | Fichier de résultats | results/results.csv |

## Structure

```
src/
  main.py                  ← point d'entrée
  benchmark/
    conjecture.py          ← objet Conjecture (eval_f, violation)
    load_benchmark.py      ← lecture du Excel
  graphs/
    generate.py            ← génération initiale
    mutate.py              ← mutations locales
    repair.py              ← réparation de classe
  invariants/
    compute.py             ← calcul des invariants
  scoring/
    violation.py           ← score = violation
  search/
    search_simple.py       ← boucle de recherche
```
