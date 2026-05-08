"""
Calcul des invariants de graphes.
Seuls les invariants X et Y nécessaires à une conjecture sont calculés obligatoirement.
Les autres sont optionnels (pour le score avancé).
"""
from __future__ import annotations
from typing import Dict, Any, Set, Optional

import networkx as nx
import numpy as np


class InvariantNotImplementedError(NotImplementedError):
    """Levée quand un invariant demandé n'est pas implémenté."""
    def __init__(self, name: str):
        super().__init__(f"Invariant '{name}' non implémenté.")
        self.name = name


def compute_invariant(G: nx.Graph, name: str) -> float:
    """
    Calcule un invariant nommé pour le graphe G.
    Lève InvariantNotImplementedError si non supporté.
    """
    n = G.number_of_nodes()
    m = G.number_of_edges()

    if name == "order":
        return float(n)

    if name == "size":
        return float(m)

    if name == "density":
        if n <= 1:
            return 0.0
        return 2 * m / (n * (n - 1))

    if name == "diameter":
        if not nx.is_connected(G):
            return float("inf")
        return float(nx.diameter(G))

    if name == "radius":
        if not nx.is_connected(G):
            return float("inf")
        return float(nx.radius(G))

    if name == "minimum_degree":
        if n == 0:
            return 0.0
        return float(min(d for _, d in G.degree()))

    if name == "maximum_degree":
        if n == 0:
            return 0.0
        return float(max(d for _, d in G.degree()))

    if name == "average_degree":
        if n == 0:
            return 0.0
        return 2 * m / n

    if name == "clique_number":
        return float(nx.graph_clique_number(G) if hasattr(nx, "graph_clique_number")
                     else len(max(nx.find_cliques(G), key=len, default=[])))

    if name == "triangle_number":
        triangles = sum(nx.triangles(G).values()) // 3
        return float(triangles)

    if name == "domination_number":
        return float(_domination_number(G))

    if name == "total_domination_number":
        return float(_total_domination_number(G))

    if name == "independence_number":
        return float(_independence_number(G))

    if name == "vertex_cover_number":
        alpha = _independence_number(G)
        return float(n - alpha)

    if name == "independent_domination_number":
        return float(_independent_domination_number(G))

    if name == "matching_number":
        return float(len(nx.max_weight_matching(G, maxcardinality=True)))

    if name == "proximity":
        return float(_proximity(G))

    if name == "remoteness":
        return float(_remoteness(G))

    if name == "randic_index":
        return float(_randic_index(G))

    if name == "harmonic_index":
        return float(_harmonic_index(G))

    if name == "first_zagreb_index":
        return float(sum(d * d for _, d in G.degree()))

    if name == "second_zagreb_index":
        return float(sum(G.degree(u) * G.degree(v) for u, v in G.edges()))

    if name == "largest_eigenvalue":
        return float(_largest_eigenvalue(G))

    if name == "largest_distance_eigenvalue":
        return float(_largest_distance_eigenvalue(G))

    if name == "second_smallest_laplace_eigenvalue":
        return float(_algebraic_connectivity(G))

    raise InvariantNotImplementedError(name)


def compute_invariants(G: nx.Graph, names: Set[str]) -> Dict[str, float]:
    """
    Calcule un ensemble d'invariants pour G.
    Retourne un dict {nom: valeur}.
    Lève InvariantNotImplementedError si un invariant est inconnu.
    """
    return {name: compute_invariant(G, name) for name in names}


# ─── Implémentations internes ───────────────────────────────────────────────

def _domination_number(G: nx.Graph) -> int:
    """Approximation gloutonne du nombre de domination."""
    if G.number_of_nodes() == 0:
        return 0
    dominated = set()
    domset = set()
    nodes = sorted(G.nodes(), key=lambda v: G.degree(v), reverse=True)
    for v in nodes:
        if v not in dominated:
            domset.add(v)
            dominated.add(v)
            dominated.update(G.neighbors(v))
    return len(domset)


def _total_domination_number(G: nx.Graph) -> int:
    """Approximation gloutonne du nombre de domination totale."""
    if G.number_of_nodes() == 0:
        return 0
    n = G.number_of_nodes()
    if n <= 1:
        return n
    dominated = set()
    domset = set()
    nodes = sorted(G.nodes(), key=lambda v: G.degree(v), reverse=True)
    for v in nodes:
        if len(domset) >= n:
            break
        neighbors = set(G.neighbors(v))
        if neighbors - dominated:
            domset.add(v)
            dominated.update(neighbors)
        if dominated >= set(G.nodes()):
            break
    # Fallback si pas de domination totale possible (graphe avec noeuds isolés)
    return max(2, len(domset))


def _independence_number(G: nx.Graph) -> int:
    """Calcule le nombre d'indépendance via complémentaire + clique."""
    if G.number_of_nodes() == 0:
        return 0
    complement = nx.complement(G)
    cliques = list(nx.find_cliques(complement))
    if not cliques:
        return 1
    return len(max(cliques, key=len))


def _independent_domination_number(G: nx.Graph) -> int:
    """
    Nombre de domination indépendante = taille du plus petit ensemble indépendant dominant.
    Approximation: on cherche un ensemble indépendant maximal de petite taille.
    """
    if G.number_of_nodes() == 0:
        return 0
    # Ensemble indépendant maximal glouton (ordre degré décroissant)
    nodes = sorted(G.nodes(), key=lambda v: G.degree(v))
    independent = set()
    excluded = set()
    for v in nodes:
        if v not in excluded:
            independent.add(v)
            excluded.update(G.neighbors(v))
            excluded.add(v)
    return len(independent)


def _proximity(G: nx.Graph) -> float:
    """
    Proximity = MIN de la closeness centrality sur tous les sommets.
    closeness(v) = (n-1) / sum_des_distances_depuis_v
    C'est le sommet le MOINS central (le plus excentrique).
    """
    if not nx.is_connected(G) or G.number_of_nodes() <= 1:
        return 0.0
    closeness = nx.closeness_centrality(G)
    return float(min(closeness.values()))


def _remoteness(G: nx.Graph) -> float:
    """
    Remoteness = MAX de la closeness centrality sur tous les sommets.
    closeness(v) = (n-1) / sum_des_distances_depuis_v
    C'est le sommet le PLUS central (le plus proche de tous les autres).
    """
    if not nx.is_connected(G) or G.number_of_nodes() <= 1:
        return 0.0
    closeness = nx.closeness_centrality(G)
    return float(max(closeness.values()))


def _randic_index(G: nx.Graph) -> float:
    """Indice de Randić = somme sur les arêtes de 1/sqrt(d(u)*d(v))."""
    total = 0.0
    for u, v in G.edges():
        du, dv = G.degree(u), G.degree(v)
        if du > 0 and dv > 0:
            total += 1.0 / (du * dv) ** 0.5
    return total


def _harmonic_index(G: nx.Graph) -> float:
    """Indice harmonique = somme sur les arêtes de 2/(d(u)+d(v))."""
    total = 0.0
    for u, v in G.edges():
        du, dv = G.degree(u), G.degree(v)
        if du + dv > 0:
            total += 2.0 / (du + dv)
    return total


def _largest_eigenvalue(G: nx.Graph) -> float:
    """Plus grande valeur propre de la matrice d'adjacence."""
    if G.number_of_nodes() == 0:
        return 0.0
    A = nx.to_numpy_array(G)
    eigvals = np.linalg.eigvalsh(A)
    return float(eigvals[-1])


def _largest_distance_eigenvalue(G: nx.Graph) -> float:
    """Plus grande valeur propre de la matrice des distances."""
    if not nx.is_connected(G) or G.number_of_nodes() <= 1:
        return 0.0
    D = nx.floyd_warshall_numpy(G)
    eigvals = np.linalg.eigvalsh(D)
    return float(eigvals[-1])


def _algebraic_connectivity(G: nx.Graph) -> float:
    """Connectivité algébrique = 2ème plus petite valeur propre du Laplacien.
    Utilise numpy.linalg.eigvalsh (déterministe, fiable) plutôt que ARPACK
    qui peut diverger sur les graphes quasi-déconnectés (barbell, λ₂ ≈ 0).
    """
    import numpy as np
    n = G.number_of_nodes()
    if not nx.is_connected(G) or n <= 1:
        return 0.0
    # Pour les grands graphes, numpy eigvalsh est O(n³) — limiter à n ≤ 300
    if n > 300:
        try:
            return float(nx.algebraic_connectivity(G, tol=1e-6))
        except Exception:
            return 0.0
    L = nx.laplacian_matrix(G).toarray().astype(float)
    eigenvalues = np.linalg.eigvalsh(L)
    return float(sorted(eigenvalues)[1])
