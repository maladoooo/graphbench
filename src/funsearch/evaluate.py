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


# Fonction de base (seed) — violation pure
SEED_FUNCTION = """def heuristic_score(G, invariants, conjecture):
    \"\"\"Fonction de base : score = violation pure.\"\"\"
    return conjecture.violation(invariants)
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
