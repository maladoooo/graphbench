"""
Fonctions de score pour guider la recherche.
Phase 1: score = violation pure.
Phase 2 (FunSearch): score = violation + bonus - penalty.

Le `heuristic_score` ci-dessous est la FONCTION DE FALLBACK (baseline) :
si une meilleure fonction a été produite par FunSearch (results/funsearch/best_heuristic.py),
on l'utilise à la place via `load_best_heuristic()`.
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Optional, Callable

import networkx as nx


def violation_score(invariants: Dict[str, float], conjecture) -> float:
    """
    Score de base = violation de la conjecture.
    violation > 0 ⟺ contre-exemple trouvé.
    """
    try:
        return conjecture.violation(invariants)
    except KeyError:
        return float("-inf")


def heuristic_score(G: nx.Graph, invariants: Dict[str, float], conjecture) -> float:
    """
    Fonction de score heuristique BASELINE (Phase 2 — exemple TP section 7.3 ajusté).
    C'est le fallback : si FunSearch a produit une meilleure fonction, elle est utilisée
    à la place via `load_best_heuristic()`.

    Paramètres récupérés mais non tous utilisés → prêts pour l'évolution.
    """
    violation = conjecture.violation(invariants)
    n = invariants.get("order", G.number_of_nodes())
    m = invariants.get("size", G.number_of_edges())
    diam = invariants.get("diameter", 0)
    delta = invariants.get("minimum_degree", 0)
    Delta = invariants.get("maximum_degree", 0)
    gamma = invariants.get("domination_number", 0)
    alpha = invariants.get("independence_number", 0)
    tau = invariants.get("vertex_cover_number", 0)
    triangles = invariants.get("triangle_number", 0)

    density = 0.0
    if n > 1:
        density = 2 * m / (n * (n - 1))

    # Score de base: violation pondérée + bonus structurels
    return (
        10.0 * violation
        + 0.3 * diam
        + 0.2 * Delta
        + 0.1 * triangles
        - 0.005 * n
        - 0.1 * abs(density - 0.5)
    )


# ─── Chargement de la fonction FunSearch (Phase 2) ────────────────────────────
_LOADED_BEST_FN: Optional[Callable] = None
_LOADED_BEST_PATH: Optional[Path] = None


def load_best_heuristic(path: Optional[Path] = None) -> Optional[Callable]:
    """
    Charge la fonction `heuristic_score` produite par FunSearch (best_heuristic.py)
    si elle existe ET qu'elle est syntaxiquement valide.
    Retourne None sinon (le caller utilise alors `heuristic_score` de ce module en fallback).
    """
    global _LOADED_BEST_FN, _LOADED_BEST_PATH
    if path is None:
        path = Path(__file__).parent.parent.parent / "results" / "funsearch" / "best_heuristic.py"
    if not path.is_file():
        return None
    if _LOADED_BEST_FN is not None and _LOADED_BEST_PATH == path:
        return _LOADED_BEST_FN
    try:
        code = path.read_text()
        namespace: Dict[str, Any] = {}
        exec(compile(code, str(path), "exec"), namespace)
        fn = namespace.get("heuristic_score")
        if callable(fn):
            _LOADED_BEST_FN = fn
            _LOADED_BEST_PATH = path
            return fn
    except Exception as e:
        print(f"  ⚠️  Impossible de charger {path}: {e}")
    return None
