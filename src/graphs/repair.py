"""
Réparation des graphes après mutation pour qu'ils respectent la classe demandée.
"""
from __future__ import annotations
import random
from typing import List, Optional, Tuple

import networkx as nx


def repair(
    G: nx.Graph,
    subgroups: List[str],
    rng: Optional[random.Random] = None,
) -> Tuple[nx.Graph, bool]:
    """
    Répare G pour qu'il respecte les classes dans subgroups.

    Args:
        G: graphe potentiellement invalide
        subgroups: classes à respecter
        rng: générateur aléatoire

    Returns:
        (graphe réparé, succès) – succès=False si réparation impossible.
    """
    if rng is None:
        rng = random.Random()

    H = G.copy()

    try:
        if "tree" in subgroups:
            H = _repair_tree(H, rng)
        elif "bipartite" in subgroups:
            H = _repair_bipartite(H, rng)
        elif "planar" in subgroups:
            ok, H = _repair_planar(H, rng)
            if not ok:
                return G, False
        elif "claw_free" in subgroups:
            ok, H = _repair_claw_free(H, rng)
            if not ok:
                return G, False
            if "connected" in subgroups and not nx.is_connected(H):
                H = _connect_components(H, rng)
        elif "connected" in subgroups:
            if not nx.is_connected(H):
                H = _connect_components(H, rng)

        # Vérification finale
        if not _check_class(H, subgroups):
            return G, False

        return H, True

    except Exception:
        return G, False


def _check_class(G: nx.Graph, subgroups: List[str]) -> bool:
    """Vérifie que G appartient à toutes les classes spécifiées."""
    if G.number_of_nodes() == 0:
        return False
    if "connected" in subgroups and not nx.is_connected(G):
        return False
    if "tree" in subgroups:
        if not nx.is_tree(G):
            return False
    if "bipartite" in subgroups and not nx.is_bipartite(G):
        return False
    if "planar" in subgroups and not nx.is_planar(G):
        return False
    if "claw_free" in subgroups and not _is_claw_free(G):
        return False
    return True


def _connect_components(G: nx.Graph, rng: random.Random) -> nx.Graph:
    """Ajoute des arêtes pour rendre G connexe."""
    components = list(nx.connected_components(G))
    while len(components) > 1:
        c1, c2 = rng.sample(components, 2)
        u = rng.choice(list(c1))
        v = rng.choice(list(c2))
        G.add_edge(u, v)
        components = list(nx.connected_components(G))
    return G


def _repair_tree(G: nx.Graph, rng: random.Random) -> nx.Graph:
    """
    Répare G pour qu'il soit un arbre:
    - Supprime les cycles (en retirant des arêtes)
    - Reconnecte les composantes
    """
    # Supprimer des arêtes pour casser les cycles
    while not nx.is_forest(G):
        cycles = nx.cycle_basis(G)
        if not cycles:
            break
        cycle = cycles[0]
        # Retirer une arête du cycle
        u, v = cycle[0], cycle[1]
        if G.has_edge(u, v):
            G.remove_edge(u, v)

    # Reconnecter les composantes
    components = list(nx.connected_components(G))
    while len(components) > 1:
        c1, c2 = rng.sample(components, 2)
        u = rng.choice(list(c1))
        v = rng.choice(list(c2))
        G.add_edge(u, v)
        components = list(nx.connected_components(G))

    return G


def _repair_bipartite(G: nx.Graph, rng: random.Random) -> nx.Graph:
    """
    Répare G pour qu'il soit biparti connexe.
    Si G n'est plus biparti, supprime les arêtes coupant la bipartition.
    """
    if nx.is_bipartite(G):
        if not nx.is_connected(G):
            G = _connect_bipartite(G, rng)
        return G

    # Essayer de trouver une 2-coloration valide en supprimant des arêtes impaires
    # Stratégie: reconstruire la bipartition depuis 0
    G = _rebuild_bipartite(G, rng)
    if not nx.is_connected(G):
        G = _connect_bipartite(G, rng)
    return G


def _rebuild_bipartite(G: nx.Graph, rng: random.Random) -> nx.Graph:
    """Reconstruit un graphe biparti en gardant un maximum d'arêtes."""
    nodes = list(G.nodes())
    rng.shuffle(nodes)
    # BFS coloring
    color = {}
    H = nx.Graph()
    H.add_nodes_from(G.nodes())
    queue = [nodes[0]]
    color[nodes[0]] = 0
    while queue:
        v = queue.pop(0)
        for u in G.neighbors(v):
            if u not in color:
                color[u] = 1 - color[v]
                queue.append(u)
            elif color[u] != 1 - color[v]:
                # Arête impaire: on la saute
                continue
            else:
                if color[v] != color[u]:
                    H.add_edge(v, u)
    # Colorier les sommets non atteints
    for v in nodes:
        if v not in color:
            color[v] = 0
    # Copier les attributs bipartite
    for v in H.nodes():
        H.nodes[v]["bipartite"] = color.get(v, 0)
    return H


def _connect_bipartite(G: nx.Graph, rng: random.Random) -> nx.Graph:
    """Rend G connexe en respectant la bipartition."""
    if not nx.is_bipartite(G):
        return _connect_components(G, rng)
    top, bottom = nx.bipartite.sets(G)
    top, bottom = list(top), list(bottom)
    components = list(nx.connected_components(G))
    while len(components) > 1:
        c1, c2 = rng.sample(components, 2)
        # Choisir un sommet de chaque composante dans des parties différentes
        nodes_c1 = list(c1)
        nodes_c2 = list(c2)
        u = rng.choice(nodes_c1)
        # v doit être dans la partie opposée à u
        u_part = "top" if u in top else "bottom"
        candidates_v = [v for v in nodes_c2 if (v in bottom if u_part == "top" else v in top)]
        if not candidates_v:
            candidates_v = nodes_c2
        v = rng.choice(candidates_v)
        G.add_edge(u, v)
        components = list(nx.connected_components(G))
    return G


def _repair_planar(G: nx.Graph, rng: random.Random):
    """
    Répare G pour qu'il soit planaire.
    Supprime des arêtes jusqu'à planarité.
    """
    is_planar, _ = nx.check_planarity(G)
    if is_planar:
        if "connected" in []:
            pass
        return True, G

    # Supprimer des arêtes aléatoires jusqu'à obtenir la planarité
    edges = list(G.edges())
    rng.shuffle(edges)
    for u, v in edges:
        if G.has_edge(u, v):
            G.remove_edge(u, v)
            is_planar, _ = nx.check_planarity(G)
            if is_planar:
                break

    return nx.check_planarity(G)[0], G


def _is_claw_free(G: nx.Graph) -> bool:
    """Vérifie qu'aucun sommet n'a 3 voisins mutuellement non-adjacents (griffe K_{1,3})."""
    for v in G.nodes():
        neighbors = list(G.neighbors(v))
        if len(neighbors) < 3:
            continue
        # Chercher 3 voisins indépendants
        from itertools import combinations
        for a, b, c in combinations(neighbors, 3):
            if not G.has_edge(a, b) and not G.has_edge(a, c) and not G.has_edge(b, c):
                return False
    return True


def _repair_claw_free(G: nx.Graph, rng: random.Random):
    """
    Répare G pour qu'il soit sans griffe.
    Stratégie: détecter les griffes et ajouter des arêtes entre les branches.
    """
    from itertools import combinations
    max_iter = 100
    for _ in range(max_iter):
        claw_found = False
        for v in list(G.nodes()):
            neighbors = list(G.neighbors(v))
            if len(neighbors) < 3:
                continue
            for a, b, c in combinations(neighbors, 3):
                if not G.has_edge(a, b) and not G.has_edge(a, c) and not G.has_edge(b, c):
                    # Ajouter une arête au hasard entre deux branches
                    pair = rng.choice([(a, b), (a, c), (b, c)])
                    G.add_edge(*pair)
                    claw_found = True
                    break
            if claw_found:
                break
        if not claw_found:
            return True, G

    return _is_claw_free(G), G
