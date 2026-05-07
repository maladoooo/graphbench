"""
Fonctions de score pour guider la recherche.
Phase 1: score = violation pure.
Phase 2 (FunSearch): score = violation + bonus - penalty.
"""
from __future__ import annotations
from typing import Dict, Any

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
    Fonction de score heuristique (Phase 2).
    À améliorer via FunSearch.
    
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
        - 0.05 * n
        - 0.2 * abs(density - 0.5)
    )
