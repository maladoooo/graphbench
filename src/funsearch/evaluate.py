"""
Évalue une fonction de score heuristique sur un ensemble de conjectures.

Rôle dans FunSearch :
- Prend une fonction Python (sous forme de code string)
- L'exécute de manière sécurisée (sandbox minimal)
- Mesure : combien de conjectures réfutées, coût total
- Retourne un score de performance global
"""
from __future__ import annotations
import time
import traceback
from typing import List, Optional, Tuple

import networkx as nx

from ..benchmark.conjecture import Conjecture
from ..search.search_simple import search


# Fonction de base (seed minimal) — violation pure (point de départ)
SEED_FUNCTION = """def heuristic_score(G, invariants, conjecture):
    \"\"\"Fonction de base : score = violation pure.\"\"\"
    return conjecture.violation(invariants)
"""

# Fonction baseline améliorée (Partie 1 + bonus structurels, exemple du TP section 7.3)
# Sert de SEED ENRICHI : FunSearch part de cette base et l'améliore par évolution.
BASELINE_FUNCTION = """def heuristic_score(G, invariants, conjecture):
    \"\"\"Baseline enrichie : violation + bonus structurels.
    Inspirée de l'exemple TP section 7.3 (coefficients ajustés).\"\"\"
    violation = conjecture.violation(invariants)
    n = invariants.get("order", G.number_of_nodes())
    m = invariants.get("size", G.number_of_edges())
    diam = invariants.get("diameter", 0)
    Delta = invariants.get("maximum_degree", 0)
    triangles = invariants.get("triangle_number", 0)
    density = 0.0
    if n > 1:
        density = 2 * m / (n * (n - 1))
    return (
        10.0 * violation
        + 0.3 * diam
        + 0.2 * Delta
        + 0.1 * triangles
        - 0.005 * n
        - 0.1 * abs(density - 0.5)
    )
"""


def compile_heuristic(code: str):
    """
    Compile et retourne la fonction heuristic_score depuis son code Python.
    Retourne None si le code est invalide.
    """
    namespace = {}
    try:
        exec(compile(code, "<funsearch>", "exec"), namespace)
        fn = namespace.get("heuristic_score")
        if callable(fn):
            return fn
        return None
    except Exception as e:
        print(f"  [FunSearch] Erreur compilation : {e}")
        return None


def evaluate_heuristic(
    heuristic_fn,
    conjectures: List[Conjecture],
    time_limit: float = 15.0,
    verbose: bool = False,
) -> dict:
    """
    Évalue une fonction heuristique sur une liste de conjectures.

    Returns:
        {
            "found": int,           # nombre de conjectures réfutées
            "total_cost": float,    # coût total (ti si trouvé, 120 sinon)
            "avg_time": float,      # temps moyen pour les réfutées
        }
    """
    found = 0
    total_cost = 0.0
    times = []

    for conj in conjectures:
        # Wrapper qui injecte notre heuristique dans la recherche
        result = search(
            conj,
            time_limit=time_limit,
            use_heuristic=True,
            heuristic_fn=heuristic_fn,
            verbose=False,
            stagnation_limit=300,
        )
        total_cost += result.cost
        if result.found:
            found += 1
            times.append(result.time_s)
            if verbose:
                print(f"    ✅ #{conj.id} en {result.time_s:.2f}s")
        else:
            if verbose:
                print(f"    ❌ #{conj.id} (violation={result.best_violation:.4f})")

    return {
        "found": found,
        "total_cost": total_cost,
        "avg_time": sum(times) / len(times) if times else 0.0,
    }
