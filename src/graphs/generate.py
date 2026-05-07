"""
Générateurs de graphes initiaux respectant une classe donnée.

Deux modes :
- generate_initial_graph()  : génération aléatoire classique (Phase 1)
- generate_smart()          : génération guidée par la conjecture (Filter, Phase 2)
"""
from __future__ import annotations
import random
from typing import List, Optional, TYPE_CHECKING

import networkx as nx

if TYPE_CHECKING:
    from ..benchmark.conjecture import Conjecture


def generate_smart(
    conjecture: "Conjecture",
    seed: Optional[int] = None,
) -> Optional[nx.Graph]:
    """
    Génère un graphe de départ INTELLIGENT basé sur la structure de la conjecture.
    Retourne None si aucune stratégie spécifique n'est connue (fallback sur aléatoire).

    Idée : analyser les invariants X et Y pour deviner la forme optimale du graphe.
    Exemple : si Y=clique_number et X=average_degree → clique + feuilles est idéal.
    """
    rng = random.Random(seed)
    x, y = conjecture.x_name, conjecture.y_name

    # --- Stratégie 1 : clique_number vs average_degree ---
    # Pour maximiser clique/avg_degree : grosse clique + beaucoup de feuilles
    if "clique_number" in (x, y) and "average_degree" in (x, y):
        return _best_clique_plus_leaves(conjecture, rng)

    # --- Stratégie 2a : clique_number vs minimum_degree ---
    # Pour violer : min_degree > f(clique_number)
    # Structure : Kn moins quelques arêtes non-adjacentes (réduit clique sans baisser min_degree)
    if "clique_number" in (x, y) and "minimum_degree" in (x, y):
        return _best_dense_minus_matching(conjecture, rng)

    # --- Stratégie 2b : clique_number vs maximum_degree ---
    if "clique_number" in (x, y) and "maximum_degree" in (x, y):
        return _best_clique_plus_leaves(conjecture, rng)

    # --- Stratégie 3 : density vs proximity ---
    # Pour violer : density haute + proximity basse
    # Structure : clique Kk + chemin de longueur b depuis le sommet 0
    if "density" in (x, y) and "proximity" in (x, y):
        return _best_clique_plus_path(conjecture, rng)

    # --- Stratégie 4 : second_smallest_laplace_eigenvalue vs n'importe quoi ---
    # Deux cliques reliées par un pont : lambda2 petite, avg_degree/independence_number contrôlables
    if "second_smallest_laplace_eigenvalue" in (x, y):
        return _best_two_cliques(conjecture, rng)

    # --- Stratégie 4b : remoteness dans un arbre ---
    # Arbre en "balai" : deux hubs reliés par un long chemin + feuilles
    if "remoteness" in (x, y) and "tree" in conjecture.subgroups:
        return _best_broom_tree(conjecture, rng)

    # --- Stratégie 5 : independence_number vs total_domination_number ---
    if "independence_number" in (x, y) and "total_domination_number" in (x, y):
        return _star_graph(rng)

    # --- Stratégie 6 : triangle_number vs maximum_degree ---
    # Pour violer : graphes d-réguliers avec beaucoup de triangles et d petit
    if "triangle_number" in (x, y) and "maximum_degree" in (x, y):
        return _best_regular_graph(conjecture, rng)

    # --- Stratégie 6b : triangle_number vs autre chose ---
    if "triangle_number" in (x, y):
        return _best_clique_plus_leaves(conjecture, rng)

    # Pas de stratégie connue
    return None


def _eval(G: nx.Graph, conjecture) -> Optional[float]:
    """Calcule la violation d'un graphe pour une conjecture. Retourne None si impossible."""
    inv = _quick_invariants(G, conjecture.x_name, conjecture.y_name)
    if inv is None:
        return None
    try:
        return conjecture.violation(inv)
    except Exception:
        return None


def _best_clique_plus_leaves(conjecture, rng: random.Random) -> nx.Graph:
    best_viol = float("-inf")
    best_G = None
    for k in range(3, 9):
        for f in range(0, 30):
            G = nx.complete_graph(k)
            for i in range(f):
                G.add_edge(0, k + i)
            viol = _eval(G, conjecture)
            if viol is None:
                continue
            if viol > best_viol:
                best_viol = viol
                best_G = G.copy()
                if viol > 0:
                    return best_G
    return best_G


def _quick_invariants(G: nx.Graph, x_name: str, y_name: str) -> Optional[dict]:
    """Calcule rapidement uniquement X et Y sans importer compute.py (évite import circulaire)."""
    n = G.number_of_nodes()
    m = G.number_of_edges()
    result = {}

    for name in (x_name, y_name):
        if name == "average_degree":
            result[name] = 2 * m / n if n > 0 else 0.0
        elif name == "clique_number":
            cliques = list(nx.find_cliques(G))
            result[name] = float(len(max(cliques, key=len))) if cliques else 1.0
        elif name == "minimum_degree":
            result[name] = float(min(d for _, d in G.degree())) if n > 0 else 0.0
        elif name == "maximum_degree":
            result[name] = float(max(d for _, d in G.degree())) if n > 0 else 0.0
        elif name == "triangle_number":
            result[name] = float(sum(nx.triangles(G).values()) // 3)
        elif name == "order":
            result[name] = float(n)
        elif name == "size":
            result[name] = float(m)
        elif name == "density":
            result[name] = 2 * m / (n * (n - 1)) if n > 1 else 0.0
        elif name == "proximity":
            if nx.is_connected(G) and n > 1:
                result[name] = float(min(nx.closeness_centrality(G).values()))
            else:
                result[name] = 0.0
        elif name == "remoteness":
            if nx.is_connected(G) and n > 1:
                result[name] = float(max(nx.closeness_centrality(G).values()))
            else:
                result[name] = 0.0
        elif name == "second_smallest_laplace_eigenvalue":
            if nx.is_connected(G) and n > 1:
                result[name] = float(nx.algebraic_connectivity(G))
            else:
                result[name] = 0.0
        elif name == "independence_number":
            complement = nx.complement(G)
            cliques = list(nx.find_cliques(complement))
            result[name] = float(len(max(cliques, key=len))) if cliques else 1.0
        elif name == "domination_number":
            dominated = set()
            domset = set()
            for v in sorted(G.nodes(), key=lambda v: G.degree(v), reverse=True):
                if v not in dominated:
                    domset.add(v)
                    dominated.add(v)
                    dominated.update(G.neighbors(v))
            result[name] = float(len(domset))
        elif name == "diameter":
            if nx.is_connected(G) and n > 1:
                result[name] = float(nx.diameter(G))
            else:
                result[name] = float("inf")
        elif name == "average_degree":
            result[name] = 2 * m / n if n > 0 else 0.0
        else:
            return None  # invariant non géré ici → fallback

    return result


def _best_clique_plus_path(conjecture, rng: random.Random) -> nx.Graph:
    best_viol = float("-inf")
    best_G = None
    for k in range(4, 13):
        for b in range(1, 6):
            G = nx.complete_graph(k)
            prev = 0
            for i in range(b):
                new_node = k + i
                G.add_edge(prev, new_node)
                prev = new_node
            viol = _eval(G, conjecture)
            if viol is None:
                continue
            if viol > best_viol:
                best_viol = viol
                best_G = G.copy()
                if viol > 0:
                    return best_G
    return best_G


def _best_two_cliques(conjecture, rng: random.Random) -> nx.Graph:
    best_viol = float("-inf")
    best_G = None
    for k in range(3, 15):
        G = nx.complete_graph(k)
        G2 = nx.relabel_nodes(nx.complete_graph(k), {v: v + k for v in range(k)})
        G = nx.compose(G, G2)
        G.add_edge(0, k)
        viol = _eval(G, conjecture)
        if viol is None:
            continue
        if viol > best_viol:
            best_viol = viol
            best_G = G.copy()
            if viol > 0:
                return best_G
    return best_G


def _best_broom_tree(conjecture, rng: random.Random) -> nx.Graph:
    best_viol = float("-inf")
    best_G = None
    for path_len in range(2, 9):
        for f in range(1, 16):
            G = nx.path_graph(path_len + 1)
            hub_a, hub_b = 0, path_len
            for i in range(f):
                G.add_edge(hub_a, path_len + 1 + i)
                G.add_edge(hub_b, path_len + 1 + f + i)
            viol = _eval(G, conjecture)
            if viol is None:
                continue
            if viol > best_viol:
                best_viol = viol
                best_G = G.copy()
                if viol > 0:
                    return best_G
    return best_G


def _star_graph(rng: random.Random) -> nx.Graph:
    """Étoile K_{1,n} : bon point de départ pour domination/independence."""
    n = rng.randint(8, 15)
    return nx.star_graph(n)


def _best_regular_graph(conjecture, rng: random.Random) -> nx.Graph:
    """
    Cherche le meilleur graphe d-régulier pour triangle_number vs maximum_degree.
    Les graphes d-réguliers avec d petit peuvent avoir plus de triangles que Kd+1.
    Ex: graphe cubique (d=3) avec 30+ sommets peut avoir >4 triangles (> f(3)=4).
    """
    best_viol = float("-inf")
    best_G = None
    for d in range(3, 7):
        for n in [10, 14, 20, 30, 36, 50, 70]:
            if n * d % 2 != 0 or d >= n:
                continue
            for _ in range(8):
                try:
                    G = nx.random_regular_graph(d, n, seed=rng.randint(0, 2**31))
                    viol = _eval(G, conjecture)
                    if viol is None:
                        continue
                    if viol > best_viol:
                        best_viol = viol
                        best_G = G.copy()
                        if viol > 0:
                            return best_G
                except Exception:
                    continue
    return best_G


def _best_dense_minus_matching(conjecture, rng: random.Random) -> nx.Graph:
    """
    Kn moins un couplage (matching) pour clique_number vs minimum_degree.
    Enlever des arêtes non-adjacentes réduit la clique sans trop baisser le min_degree.
    Ex: K11 moins 2 arêtes disjointes → clique=9, min_degree=9 → violation ~0.095.
    """
    best_viol = float("-inf")
    best_G = None
    for n in range(8, 15):
        Kn = nx.complete_graph(n)
        edges = list(Kn.edges())
        # Essayer de supprimer 0..n//2 arêtes formant un couplage
        for k_remove in range(0, min(n // 2 + 1, 6)):
            matching_edges = []
            used_nodes: set = set()
            for u, v in rng.sample(edges, len(edges)):
                if u not in used_nodes and v not in used_nodes:
                    matching_edges.append((u, v))
                    used_nodes.add(u)
                    used_nodes.add(v)
                if len(matching_edges) == k_remove:
                    break
            G = Kn.copy()
            G.remove_edges_from(matching_edges)
            if not nx.is_connected(G):
                continue
            viol = _eval(G, conjecture)
            if viol is None:
                continue
            if viol > best_viol:
                best_viol = viol
                best_G = G.copy()
                if viol > 0:
                    return best_G
    return best_G


def generate_initial_graph(
    subgroups: List[str],
    n: int = 8,
    seed: Optional[int] = None,
) -> nx.Graph:
    """
    Génère un graphe initial respectant les classes spécifiées.

    Args:
        subgroups: liste de classes à respecter (ex: ['connected', 'tree'])
        n: nombre de sommets initial
        seed: graine pour la reproductibilité

    Returns:
        Un graphe NetworkX respectant les classes.
    """
    rng = random.Random(seed)

    if "tree" in subgroups:
        return _generate_tree(n, rng)
    if "bipartite" in subgroups:
        return _generate_bipartite(n, rng)
    if "planar" in subgroups:
        return _generate_planar(n, rng)
    if "claw_free" in subgroups and "connected" in subgroups:
        return _generate_claw_free_connected(n, rng)
    if "connected" in subgroups:
        return _generate_connected(n, rng)

    # Fallback: graphe aléatoire quelconque
    G = nx.gnm_random_graph(n, n + rng.randint(0, n), seed=rng.randint(0, 2**31))
    return G


def _generate_connected(n: int, rng: random.Random) -> nx.Graph:
    """Arbre aléatoire + arêtes supplémentaires → graphe connexe."""
    G = nx.random_labeled_tree(n, seed=rng.randint(0, 2**31))
    # Ajout de quelques arêtes supplémentaires
    extra = rng.randint(0, max(0, n // 2))
    nodes = list(G.nodes())
    for _ in range(extra):
        u, v = rng.sample(nodes, 2)
        G.add_edge(u, v)
    return G


def _generate_tree(n: int, rng: random.Random) -> nx.Graph:
    """Arbre aléatoire pur."""
    return nx.random_labeled_tree(n, seed=rng.randint(0, 2**31))


def _generate_bipartite(n: int, rng: random.Random) -> nx.Graph:
    """Graphe biparti connexe aléatoire."""
    n1 = max(1, n // 2)
    n2 = n - n1
    p = rng.uniform(0.3, 0.7)
    G = nx.bipartite.random_graph(n1, n2, p, seed=rng.randint(0, 2**31))
    # Assurer la connexité
    if not nx.is_connected(G):
        G = _connect_graph(G, rng)
    return G


def _generate_planar(n: int, rng: random.Random) -> nx.Graph:
    """
    Graphe planaire connexe via triangulation incrémentale.
    On part d'un triangle et on ajoute des sommets dans les faces.
    """
    if n < 3:
        G = nx.path_graph(n)
        return G
    G = nx.cycle_graph(3)
    node_id = 3
    while node_id < n:
        # Choisir une arête existante et ajouter un sommet relié à ses deux extrémités
        edge = rng.choice(list(G.edges()))
        G.add_node(node_id)
        G.add_edge(node_id, edge[0])
        G.add_edge(node_id, edge[1])
        node_id += 1
    return G


def _generate_claw_free_connected(n: int, rng: random.Random) -> nx.Graph:
    """
    Graphe sans griffe connexe.
    Stratégie: complémentaire d'un graphe triangle-free (Turán),
    ou plus simplement: graphe de ligne d'un graphe aléatoire.
    Le graphe de ligne L(H) est toujours sans griffe.
    """
    # Le graphe de ligne d'un graphe quelconque est sans griffe
    # On génère H tel que L(H) ait environ n sommets
    # |V(L(H))| = |E(H)|, donc on veut |E(H)| ≈ n
    # Un graphe à m arêtes peut être obtenu avec ~sqrt(2m) sommets en clique
    import math
    m_target = n
    nh = max(3, int(math.ceil((-1 + math.sqrt(1 + 8 * m_target)) / 2)) + 1)
    H = nx.gnm_random_graph(nh, m_target, seed=rng.randint(0, 2**31))
    if H.number_of_edges() == 0:
        H.add_edge(0, 1)
    G = nx.line_graph(H)
    # Relabeler les noeuds en entiers
    G = nx.convert_node_labels_to_integers(G)
    if not nx.is_connected(G):
        G = _connect_graph(G, rng)
    return G


def _connect_graph(G: nx.Graph, rng: random.Random) -> nx.Graph:
    """Rend un graphe connexe en ajoutant des arêtes entre composantes."""
    components = list(nx.connected_components(G))
    while len(components) > 1:
        # Relier deux composantes aléatoires
        c1, c2 = rng.sample(components, 2)
        u = rng.choice(list(c1))
        v = rng.choice(list(c2))
        G.add_edge(u, v)
        components = list(nx.connected_components(G))
    return G
