"""
Utilitaires de vérification de classe de graphes.
"""
from __future__ import annotations
from typing import List

import networkx as nx
from itertools import combinations


def check_graph_class(G: nx.Graph, subgroups: List[str]) -> bool:
    """Retourne True si G appartient à toutes les classes spécifiées."""
    if G.number_of_nodes() == 0:
        return False
    if "connected" in subgroups and not nx.is_connected(G):
        return False
    if "tree" in subgroups and not nx.is_tree(G):
        return False
    if "bipartite" in subgroups and not nx.is_bipartite(G):
        return False
    if "planar" in subgroups and not nx.is_planar(G):
        return False
    if "claw_free" in subgroups and not is_claw_free(G):
        return False
    return True


def is_claw_free(G: nx.Graph) -> bool:
    """Vérifie que G ne contient pas K_{1,3} comme sous-graphe induit."""
    for v in G.nodes():
        neighbors = list(G.neighbors(v))
        if len(neighbors) < 3:
            continue
        for a, b, c in combinations(neighbors, 3):
            if not G.has_edge(a, b) and not G.has_edge(a, c) and not G.has_edge(b, c):
                return False
    return True
