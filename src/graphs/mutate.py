"""
Mutations locales sur les graphes.
Chaque mutation retourne un NOUVEAU graphe (copie modifiée).
"""
from __future__ import annotations
import random
from typing import Optional, List

import networkx as nx


def mutate(
    G: nx.Graph,
    subgroups: List[str],
    rng: Optional[random.Random] = None,
) -> nx.Graph:
    """
    Applique une mutation aléatoire à G.
    Le graphe résultant peut temporairement violer les contraintes de classe
    (il faudra appeler repair() ensuite).

    Args:
        G: graphe d'entrée
        subgroups: classes du graphe (guide le choix de mutation)
        rng: générateur aléatoire

    Returns:
        Un nouveau graphe muté (copie).
    """
    if rng is None:
        rng = random.Random()

    H = G.copy()
    n = H.number_of_nodes()

    if "tree" in subgroups:
        mutations = [_add_leaf, _remove_leaf, _rewire_tree]
    elif "bipartite" in subgroups:
        mutations = [_add_edge_bipartite, _remove_edge, _add_node_bipartite]
    else:
        mutations = [
            _add_edge, _remove_edge, _rewire_edge,
            _add_node_with_edges, _subdivide_edge,
        ]
        if n > 4:
            mutations.append(_remove_node)

    mutation_fn = rng.choice(mutations)
    try:
        H = mutation_fn(H, rng)
    except Exception:
        # En cas d'échec (ex: pas d'arête à supprimer), on retourne une copie non modifiée
        H = G.copy()

    return H


# ─── Mutations génériques ───────────────────────────────────────────────────

def _add_edge(G: nx.Graph, rng: random.Random) -> nx.Graph:
    nodes = list(G.nodes())
    if len(nodes) < 2:
        return G
    for _ in range(20):
        u, v = rng.sample(nodes, 2)
        if not G.has_edge(u, v):
            G.add_edge(u, v)
            return G
    return G


def _remove_edge(G: nx.Graph, rng: random.Random) -> nx.Graph:
    edges = list(G.edges())
    if not edges:
        return G
    u, v = rng.choice(edges)
    G.remove_edge(u, v)
    return G


def _rewire_edge(G: nx.Graph, rng: random.Random) -> nx.Graph:
    """Retire (u,v) et ajoute (u,w) avec w ≠ u,v."""
    edges = list(G.edges())
    if not edges:
        return G
    nodes = list(G.nodes())
    u, v = rng.choice(edges)
    candidates = [w for w in nodes if w != u and w != v and not G.has_edge(u, w)]
    if not candidates:
        return G
    w = rng.choice(candidates)
    G.remove_edge(u, v)
    G.add_edge(u, w)
    return G


def _add_node_with_edges(G: nx.Graph, rng: random.Random) -> nx.Graph:
    """Ajoute un nouveau sommet relié à 1-3 sommets existants."""
    if G.number_of_nodes() == 0:
        G.add_node(0)
        return G
    new_id = max(G.nodes()) + 1
    G.add_node(new_id)
    nodes = list(G.nodes())
    nodes.remove(new_id)
    k = rng.randint(1, min(3, len(nodes)))
    neighbors = rng.sample(nodes, k)
    for v in neighbors:
        G.add_edge(new_id, v)
    return G


def _remove_node(G: nx.Graph, rng: random.Random) -> nx.Graph:
    """Supprime un sommet aléatoire (pas le dernier)."""
    if G.number_of_nodes() <= 2:
        return G
    v = rng.choice(list(G.nodes()))
    G.remove_node(v)
    G = nx.convert_node_labels_to_integers(G)
    return G


def _subdivide_edge(G: nx.Graph, rng: random.Random) -> nx.Graph:
    """Subdivise une arête (u,v) en ajoutant un sommet intermédiaire."""
    edges = list(G.edges())
    if not edges:
        return G
    u, v = rng.choice(edges)
    new_id = max(G.nodes()) + 1
    G.remove_edge(u, v)
    G.add_node(new_id)
    G.add_edge(u, new_id)
    G.add_edge(new_id, v)
    return G


# ─── Mutations spécifiques aux arbres ───────────────────────────────────────

def _add_leaf(G: nx.Graph, rng: random.Random) -> nx.Graph:
    """Ajoute une feuille (sommet de degré 1) à un sommet aléatoire."""
    if G.number_of_nodes() == 0:
        G.add_node(0)
        return G
    parent = rng.choice(list(G.nodes()))
    new_id = max(G.nodes()) + 1
    G.add_node(new_id)
    G.add_edge(parent, new_id)
    return G


def _remove_leaf(G: nx.Graph, rng: random.Random) -> nx.Graph:
    """Supprime une feuille aléatoire (sommet de degré 1)."""
    leaves = [v for v in G.nodes() if G.degree(v) == 1]
    if not leaves:
        return G
    leaf = rng.choice(leaves)
    G.remove_node(leaf)
    G = nx.convert_node_labels_to_integers(G)
    return G


def _rewire_tree(G: nx.Graph, rng: random.Random) -> nx.Graph:
    """
    Déplace une feuille vers un autre parent.
    Maintient n-1 arêtes mais peut créer un cycle → repair corrigera.
    """
    leaves = [v for v in G.nodes() if G.degree(v) == 1]
    if not leaves:
        return G
    leaf = rng.choice(leaves)
    old_parent = list(G.neighbors(leaf))[0]
    candidates = [v for v in G.nodes() if v != leaf and v != old_parent]
    if not candidates:
        return G
    new_parent = rng.choice(candidates)
    G.remove_edge(leaf, old_parent)
    G.add_edge(leaf, new_parent)
    return G


# ─── Mutations spécifiques aux bipartis ─────────────────────────────────────

def _add_edge_bipartite(G: nx.Graph, rng: random.Random) -> nx.Graph:
    """Ajoute une arête entre les deux parties du graphe biparti."""
    if not nx.is_bipartite(G):
        return _add_edge(G, rng)
    top, bottom = nx.bipartite.sets(G)
    top, bottom = list(top), list(bottom)
    if not top or not bottom:
        return G
    for _ in range(20):
        u = rng.choice(top)
        v = rng.choice(bottom)
        if not G.has_edge(u, v):
            G.add_edge(u, v)
            return G
    return G


def _add_node_bipartite(G: nx.Graph, rng: random.Random) -> nx.Graph:
    """Ajoute un noeud à la plus petite partie du biparti."""
    if not nx.is_bipartite(G):
        return _add_node_with_edges(G, rng)
    top, bottom = nx.bipartite.sets(G)
    top, bottom = list(top), list(bottom)
    new_id = max(G.nodes(), default=-1) + 1
    # Ajouter dans la plus petite partie
    if len(top) <= len(bottom):
        part = bottom  # on connecte vers l'autre partie
        G.add_node(new_id, bipartite=0)
    else:
        part = top
        G.add_node(new_id, bipartite=1)
    k = rng.randint(1, min(3, len(part)))
    for v in rng.sample(part, k):
        G.add_edge(new_id, v)
    return G
