"""
Calcul des invariants de graphes.
Seuls les invariants X et Y nécessaires à une conjecture sont calculés obligatoirement.
Les autres sont optionnels (pour le score avancé).
"""
from __future__ import annotations
from typing import Dict, Any, Set, Optional

import networkx as nx
import numpy as np


SUPPORTED_INVARIANTS: frozenset = frozenset({
    "order", "size", "density",
    "diameter", "radius",
    "minimum_degree", "maximum_degree", "average_degree",
    "clique_number", "triangle_number",
    "domination_number", "total_domination_number",
    "independence_number", "vertex_cover_number", "independent_domination_number",
    "matching_number",
    "proximity", "remoteness",
    "randic_index", "harmonic_index",
    "first_zagreb_index", "second_zagreb_index",
    "largest_eigenvalue", "largest_distance_eigenvalue",
    "second_smallest_laplace_eigenvalue",
})


class InvariantNotImplementedError(NotImplementedError):
    """Levée quand un invariant demandé n'est pas implémenté."""
    def __init__(self, name: str):
        super().__init__(
            f"Invariant '{name}' non implémenté. "
            f"Invariants supportés : {sorted(SUPPORTED_INVARIANTS)}"
        )
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

# ─── Helpers bitmask pour branch-and-bound exact ─────────────────────────────

def _to_bitmask_adj(G: nx.Graph):
    """Convertit un graphe NetworkX en (adj_list_of_bitmasks, n).
    Relabel les sommets en 0..n-1 si nécessaire."""
    n = G.number_of_nodes()
    nodes = list(G.nodes())
    if nodes != list(range(n)):
        G = nx.convert_node_labels_to_integers(G)
    adj = [0] * n
    for u, v in G.edges():
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    return tuple(adj), n


def _min_set_cover_size(masks, n: int):
    """Branch-and-bound : taille MIN d'un sous-ensemble de masks dont l'union couvre 0..n-1.
    Retourne None si impossible."""
    import math as _math
    all_mask = (1 << n) - 1
    union = 0
    for mask in masks:
        union |= mask
    if union != all_mask:
        return None

    candidates_for_v = tuple(
        tuple(i for i, mask in enumerate(masks) if mask & (1 << v))
        for v in range(n)
    )
    max_cover = max(mask.bit_count() for mask in masks) if masks else 1
    best = [n + 1]

    def dfs(covered, available, count):
        if count >= best[0]:
            return
        uncovered = all_mask & ~covered
        if not uncovered:
            best[0] = count
            return
        if count + _math.ceil(uncovered.bit_count() / max_cover) >= best[0]:
            return

        def cand_count(v):
            return sum(1 for idx in candidates_for_v[v]
                       if (available & (1 << idx)) and (masks[idx] & uncovered))

        chosen_v = min(
            (v for v in range(n) if (1 << v) & uncovered),
            key=cand_count,
        )
        candidates = [
            idx for idx in candidates_for_v[chosen_v]
            if (available & (1 << idx)) and (masks[idx] & uncovered)
        ]
        candidates.sort(key=lambda idx: (masks[idx] & uncovered).bit_count(), reverse=True)
        for idx in candidates:
            dfs(covered | masks[idx], available & ~(1 << idx), count + 1)

    dfs(0, (1 << len(masks)) - 1, 0)
    return best[0] if best[0] <= n else None


def _domination_number(G: nx.Graph) -> int:
    """Nombre de domination EXACT via branch-and-bound (min set cover sur voisinages fermés)."""
    if G.number_of_nodes() == 0:
        return 0
    adj, n = _to_bitmask_adj(G)
    closed = tuple(adj[v] | (1 << v) for v in range(n))
    result = _min_set_cover_size(closed, n)
    if result is None:
        # Théoriquement impossible (chaque sommet se domine lui-même)
        return n
    return result


def _total_domination_number(G: nx.Graph) -> int:
    """Nombre de domination totale EXACT via branch-and-bound (min set cover sur voisinages OUVERTS).
    LÈVE ValueError s'il n'existe pas de total dominating set (n=1 ou sommet isolé)."""
    if G.number_of_nodes() == 0:
        return 0
    adj, n = _to_bitmask_adj(G)
    # Voisinages ouverts (sans soi-même)
    result = _min_set_cover_size(adj, n)
    if result is None:
        # Pas de TDS — invariant non défini. On lève comme le fait le vérificateur strict.
        raise ValueError("no total dominating set exists (isolated vertex)")
    return result


def _independence_number(G: nx.Graph) -> int:
    """Nombre d'indépendance EXACT via complémentaire + max clique (déjà exact via find_cliques)."""
    if G.number_of_nodes() == 0:
        return 0
    complement = nx.complement(G)
    cliques = list(nx.find_cliques(complement))
    if not cliques:
        return 1
    return len(max(cliques, key=len))


def _independent_domination_number(G: nx.Graph) -> int:
    """
    Nombre de domination indépendante EXACT (i(G)) via branch-and-bound.
    = MIN sur les ensembles S qui sont à la fois INDÉPENDANTS et DOMINANTS.
    NB : ce N'EST PAS la taille d'un ensemble indépendant maximal !
    Pour K_{1,3} : i = 1 (le centre seul), pas 3 (les feuilles).
    """
    import math as _math
    if G.number_of_nodes() == 0:
        return 0
    adj, n = _to_bitmask_adj(G)
    closed = tuple(adj[v] | (1 << v) for v in range(n))
    all_mask = (1 << n) - 1
    candidates_for_v = tuple(
        tuple(i for i, mask in enumerate(closed) if mask & (1 << v))
        for v in range(n)
    )
    max_cover = max(mask.bit_count() for mask in closed) if closed else 1
    best = [n + 1]

    def dfs(dominated, allowed, count):
        if count >= best[0]:
            return
        uncovered = all_mask & ~dominated
        if not uncovered:
            best[0] = count
            return
        if count + _math.ceil(uncovered.bit_count() / max_cover) >= best[0]:
            return

        def cand_count(v):
            return sum(1 for idx in candidates_for_v[v]
                       if (allowed & (1 << idx)) and (closed[idx] & uncovered))

        chosen_v = min(
            (v for v in range(n) if (1 << v) & uncovered),
            key=cand_count,
        )
        candidates = [
            idx for idx in candidates_for_v[chosen_v]
            if (allowed & (1 << idx)) and (closed[idx] & uncovered)
        ]
        candidates.sort(key=lambda idx: (closed[idx] & uncovered).bit_count(), reverse=True)
        for idx in candidates:
            # Contrainte d'indépendance : on retire idx ET tous ses voisins de allowed
            forbidden = adj[idx] | (1 << idx)
            dfs(dominated | closed[idx], allowed & ~forbidden, count + 1)

    dfs(0, all_mask, 0)
    return best[0] if best[0] <= n else n


def _proximity(G: nx.Graph) -> float:
    """
    Proximity = MIN de la closeness centrality sur tous les sommets.
    closeness(v) = (n-1) / sum_des_distances_depuis_v
    LÈVE ValueError si n <= 1 ou disconnected (proximity non définie).
    """
    if not nx.is_connected(G) or G.number_of_nodes() <= 1:
        raise ValueError("proximity undefined (n <= 1 or disconnected)")
    closeness = nx.closeness_centrality(G)
    return float(min(closeness.values()))


def _remoteness(G: nx.Graph) -> float:
    """
    Remoteness = MAX de la closeness centrality sur tous les sommets.
    closeness(v) = (n-1) / sum_des_distances_depuis_v
    LÈVE ValueError si n <= 1 ou disconnected (remoteness non définie).
    """
    if not nx.is_connected(G) or G.number_of_nodes() <= 1:
        raise ValueError("remoteness undefined (n <= 1 or disconnected)")
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
