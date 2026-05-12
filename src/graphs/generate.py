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

    # --- Stratégie 3b : density vs remoteness ---
    # Même structure que density+proximity (clique+chemin)
    if "density" in (x, y) and "remoteness" in (x, y):
        return _best_clique_plus_path(conjecture, rng)

    # --- Stratégie 4 : second_smallest_laplace_eigenvalue vs n'importe quoi ---
    # Pour claw_free : barbell (deux cliques reliées par un pont) → alg_conn très petite
    # Pour autres graphes : deux cliques reliées par un pont
    if "second_smallest_laplace_eigenvalue" in (x, y):
        if "claw_free" in conjecture.subgroups:
            # Si diameter est aussi impliqué ET sign="<=" (barbell a diameter=3 trop petit):
            # On essaie d'abord des cycles (grand diamètre + petite alg_conn).
            # NB: pour sign=">=", le barbell (small diam) est correct → pas de cycle filter.
            if "diameter" in (x, y) and conjecture.sign == "<=":
                result = _best_cycle_for_lambda_diameter(conjecture, rng)
                if result is not None:
                    return result
            return _best_barbell_claw_free(conjecture, rng)
        return _best_two_cliques(conjecture, rng)

    # --- Stratégie 4b : independence + radius dans graphe claw_free → triangular snake ---
    # Triangular snake : k triangles en chaîne → claw_free, radius = k//2, independence = k
    if "independence_number" in (x, y) and "radius" in (x, y) and "claw_free" in conjecture.subgroups:
        return _best_triangular_snake_claw_free(conjecture, rng)

    # --- Stratégie 4c : radius OU diameter dans graphe claw_free → cycles ---
    # Cycles C_k : naturellement claw_free, radius = k//2, diameter = k//2
    if ("radius" in (x, y) or "diameter" in (x, y)) and "claw_free" in conjecture.subgroups:
        return _best_cycle_claw_free(conjecture, rng)

    # --- Stratégie 4c : remoteness dans un arbre ---
    # Arbre en "balai" : deux hubs reliés par un long chemin + feuilles
    if "remoteness" in (x, y) and "tree" in conjecture.subgroups:
        return _best_broom_tree(conjecture, rng)

    # --- Stratégie 4d : remoteness vs matching/vertex_cover (graphe general) ---
    # Pour violer : graphe dense + grande remoteness → chercher des graphes de n=10..20 nœuds
    if "remoteness" in (x, y) and any(v in (x, y) for v in [
        "matching_number", "vertex_cover_number", "independence_number"
    ]):
        return _best_dense_for_remoteness(conjecture, rng)

    # --- Stratégie 4e : proximity + total_domination ---
    # Essaie dense (K_n : proximity=1, total_dom=2) et sparse (paths)
    if "proximity" in (x, y) and "total_domination_number" in (x, y):
        return _best_for_proximity_total_dom(conjecture, rng)

    # --- Stratégie 5 : independence_number vs total_domination_number ---
    if "independence_number" in (x, y) and "total_domination_number" in (x, y):
        return _star_graph(rng)

    # --- Stratégie 5b : matching_number vs total_domination_number ---
    # Essaie des graphes de densité variée
    if "matching_number" in (x, y) and "total_domination_number" in (x, y):
        return _best_sparse_for_matching_dom(conjecture, rng)

    # --- Stratégie 6 : triangle_number vs maximum_degree ---
    # Pour violer : graphes d-réguliers avec beaucoup de triangles et d petit
    if "triangle_number" in (x, y) and "maximum_degree" in (x, y):
        return _best_regular_graph(conjecture, rng)

    # --- Stratégie 6b : triangle_number vs autre chose ---
    if "triangle_number" in (x, y):
        return _best_clique_plus_leaves(conjecture, rng)

    # --- Dernier recours : batterie universelle de templates ---
    # Tente ~40 graphes standards (K_n, P_n, C_n, étoiles, bipartis, barbells,
    # triangular snakes, Petersen). Les invariants sont mis en cache au premier
    # appel : les conjectures suivantes ne font que de l'arithmétique pure.
    return _try_universal_templates(conjecture)


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
                import numpy as _np
                _L = nx.laplacian_matrix(G).toarray().astype(float)
                _ev = _np.linalg.eigvalsh(_L)
                result[name] = float(sorted(_ev)[1])
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
        elif name == "radius":
            if nx.is_connected(G) and n > 1:
                result[name] = float(nx.radius(G))
            else:
                result[name] = float("inf")
        elif name == "total_domination_number":
            # Greedy approximation (non-optimal mais rapide)
            if n <= 1:
                result[name] = 0.0
            else:
                dominated = set()
                domset = []
                nodes_sorted = sorted(G.nodes(), key=lambda v: G.degree(v), reverse=True)
                for v in nodes_sorted:
                    if v not in dominated:
                        # total domination: v doit avoir un voisin dans domset
                        # On cherche un voisin non couvert
                        nb = set(G.neighbors(v))
                        if nb - dominated:
                            domset.append(v)
                            dominated.update(nb)
                # Fallback: greedy classique
                if not dominated.issuperset(set(G.nodes())):
                    dominated2, domset2 = set(), set()
                    for v in nodes_sorted:
                        if v not in dominated2:
                            domset2.add(v)
                            dominated2.update(G.neighbors(v))
                    result[name] = float(len(domset2))
                else:
                    result[name] = float(len(domset))
        elif name == "independent_domination_number":
            # Greedy: independent set maximal = dominating set indépendant
            independent = set()
            dominated = set()
            for v in sorted(G.nodes(), key=lambda v: G.degree(v)):
                if v not in dominated and v not in independent:
                    independent.add(v)
                    dominated.update(G.neighbors(v))
            result[name] = float(len(independent))
        elif name == "vertex_cover_number":
            # König: pour graphes bipartis, sinon approx greedy
            matching = nx.max_weight_matching(G, maxcardinality=True)
            result[name] = float(len(matching))
        elif name == "matching_number":
            matching = nx.max_weight_matching(G, maxcardinality=True)
            result[name] = float(len(matching))
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


def _best_barbell_claw_free(conjecture, rng: random.Random) -> nx.Graph:
    """
    Barbell graph = deux cliques K_k reliées par une arête.
    Claw_free (cliques complètes), algebraic connectivity ≈ 2/k (très petite pour k grand).
    Idéal pour les conjectures impliquant second_smallest_laplace_eigenvalue sur claw_free.
    Utilise compute_invariants directement pour supporter tous les invariants.
    """
    from ..invariants.compute import compute_invariants as _compute_inv
    best_viol = float("-inf")
    best_G = None
    # largest_distance_eigenvalue est O(n³) sur la matrice des distances → limiter n
    involves_dist = "largest_distance_eigenvalue" in (conjecture.x_name, conjecture.y_name)
    k_max = 12 if involves_dist else 25
    for k in range(4, k_max):
        G = nx.complete_graph(k)
        H = nx.relabel_nodes(nx.complete_graph(k), {v: v + k for v in range(k)})
        G = nx.compose(G, H)
        G.add_edge(0, k)  # pont entre les deux cliques
        try:
            inv = _compute_inv(G, {conjecture.x_name, conjecture.y_name})
            viol = conjecture.violation(inv)
        except Exception:
            viol = None
        if viol is None:
            continue
        if viol > best_viol:
            best_viol = viol
            best_G = G.copy()
            if viol > 1e-9:  # seuil epsilon pour éviter les cas limites flottants
                return best_G
    return best_G


def _best_cycle_claw_free(conjecture, rng: random.Random) -> nx.Graph:
    """
    Cycles C_k : toujours claw_free et connexes.
    Pour les conjectures avec radius : radius(C_k) = k//2.
    Teste C_4 à C_20 pour trouver le meilleur contre-exemple.
    Utilise compute_invariants directement pour supporter tous les invariants.
    """
    from ..invariants.compute import compute_invariants as _compute_inv
    best_viol = float("-inf")
    best_G = None
    for n in range(4, 25):
        G = nx.cycle_graph(n)
        try:
            inv = _compute_inv(G, {conjecture.x_name, conjecture.y_name})
            viol = conjecture.violation(inv)
        except Exception:
            viol = None
        if viol is None:
            continue
        if viol > best_viol:
            best_viol = viol
            best_G = G.copy()
            if viol > 0:
                return best_G
    return best_G


def _best_dense_for_remoteness(conjecture, rng: random.Random) -> Optional[nx.Graph]:
    """
    Pour les conjectures remoteness vs matching/vertex_cover/independence.
    Teste des graphes aléatoires denses de n=10..20 nœuds.
    La remoteness est maximisée par des graphes "lollipop" ou quasi-complets.
    """
    from ..invariants.compute import compute_invariants as _compute_inv
    best_viol = float("-inf")
    best_G = None
    for n in range(12, 19):
        for trial in range(2):
            # Essayer différents types : aléatoires, lollipop, etc.
            if trial == 0:
                # Graphe complet moins quelques arêtes
                G = nx.complete_graph(n)
                edges = list(G.edges())
                for _ in range(n // 3):
                    if edges:
                        e = rng.choice(edges)
                        G.remove_edge(*e)
                        edges.remove(e)
                        if not nx.is_connected(G):
                            G.add_edge(*e)
                            edges.append(e)
            else:
                # Graphe aléatoire avec densité modérée
                p = 0.5
                seed_trial = rng.randint(0, 2**31)
                G = nx.gnp_random_graph(n, p, seed=seed_trial)
                if not nx.is_connected(G) or G.number_of_edges() == 0:
                    continue
            try:
                inv = _compute_inv(G, {conjecture.x_name, conjecture.y_name})
                viol = conjecture.violation(inv)
            except Exception:
                continue
            if viol > best_viol:
                best_viol = viol
                best_G = G.copy()
                if viol > 1e-9:
                    return best_G
    return best_G if best_viol > float("-inf") else None


def _best_cycle_for_lambda_diameter(conjecture, rng: random.Random) -> Optional[nx.Graph]:
    """
    Cycles C_n pour les conjectures diameter vs second_smallest_laplace_eigenvalue (claw_free).
    Les cycles ont grand diamètre (n//2) et petite connectivité algébrique (≈4π²/n²).
    Teste C_4 à C_60. Si aucun n'a violation > 0, retourne None (fallback barbell).
    """
    from ..invariants.compute import compute_invariants as _compute_inv
    best_viol = float("-inf")
    best_G = None
    for n in range(4, 61):
        G = nx.cycle_graph(n)
        try:
            inv = _compute_inv(G, {conjecture.x_name, conjecture.y_name})
            viol = conjecture.violation(inv)
        except Exception:
            continue
        if viol > best_viol:
            best_viol = viol
            best_G = G.copy()
            if viol > 0:
                return best_G
    # Retourner seulement si on a trouvé une violation > 0; sinon None (fallback barbell)
    return best_G if best_viol > 0 else None


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


# ─── Batterie de templates universels (avec cache d'invariants) ──────────────
# Liste construite à la demande pour éviter le coût de génération à l'import.
_TEMPLATE_GRAPHS: Optional[List[nx.Graph]] = None
_TEMPLATE_INV: dict = {}        # (template_idx, inv_name) -> float
_TEMPLATE_CLASS: dict = {}      # (template_idx, frozenset(subgroups)) -> bool


def _build_template_graphs() -> List[nx.Graph]:
    """Construit la liste des graphes templates (graphes standards de la théorie)."""
    graphs: List[nx.Graph] = []
    # Cliques K_n
    for n in [4, 5, 6, 8, 10, 12, 15, 18]:
        graphs.append(nx.complete_graph(n))
    # Chemins P_n
    for n in [4, 6, 8, 10, 12, 15, 20, 25]:
        graphs.append(nx.path_graph(n))
    # Cycles C_n
    for n in [4, 5, 6, 7, 8, 10, 12, 15, 20]:
        graphs.append(nx.cycle_graph(n))
    # Étoiles K_{1,n}
    for n in [3, 5, 8, 12]:
        graphs.append(nx.star_graph(n))
    # Bipartis complets K_{a,b}
    for a, b in [(2, 3), (2, 5), (3, 4), (3, 6), (4, 5), (4, 7), (5, 8)]:
        graphs.append(nx.complete_bipartite_graph(a, b))
    # Barbells K_k–e–K_k
    for k in [3, 4, 5, 6, 7, 8]:
        G = nx.complete_graph(k)
        H = nx.relabel_nodes(nx.complete_graph(k), {v: v + k for v in range(k)})
        G = nx.compose(G, H)
        G.add_edge(0, k)
        graphs.append(G)
    # Triangular snakes (k triangles en chaîne)
    for k in [3, 4, 5, 6, 7]:
        G = nx.Graph()
        for i in range(k):
            G.add_edge(i, i + 1)
            G.add_edge(i, k + 1 + i)
            G.add_edge(i + 1, k + 1 + i)
        graphs.append(G)
    # Graphes nommés
    try:
        graphs.append(nx.petersen_graph())
    except Exception:
        pass
    try:
        graphs.append(nx.desargues_graph())
    except Exception:
        pass
    try:
        graphs.append(nx.heawood_graph())
    except Exception:
        pass
    # Roues W_n
    for n in [4, 5, 7, 10]:
        graphs.append(nx.wheel_graph(n))
    # K_n moins un couplage parfait (~cocktail party)
    for n in [6, 8, 10]:
        G = nx.complete_graph(n)
        for i in range(0, n - 1, 2):
            if G.has_edge(i, i + 1):
                G.remove_edge(i, i + 1)
        if nx.is_connected(G):
            graphs.append(G)
    return graphs


def _get_template_graphs() -> List[nx.Graph]:
    global _TEMPLATE_GRAPHS
    if _TEMPLATE_GRAPHS is None:
        _TEMPLATE_GRAPHS = _build_template_graphs()
    return _TEMPLATE_GRAPHS


def _template_in_class(idx: int, subgroups: List[str]) -> bool:
    """Cache booléen : ce template respecte-t-il la classe demandée ?"""
    key = (idx, frozenset(subgroups))
    cached = _TEMPLATE_CLASS.get(key)
    if cached is not None:
        return cached
    from .classes import check_graph_class
    G = _get_template_graphs()[idx]
    ok = check_graph_class(G, subgroups)
    _TEMPLATE_CLASS[key] = ok
    return ok


def _template_invariant(idx: int, name: str) -> Optional[float]:
    """Cache de valeurs d'invariants par template (calcul une fois par paire)."""
    key = (idx, name)
    if key in _TEMPLATE_INV:
        return _TEMPLATE_INV[key]
    from ..invariants.compute import compute_invariant, InvariantNotImplementedError
    G = _get_template_graphs()[idx]
    try:
        val = float(compute_invariant(G, name))
        # Filtrer les valeurs problématiques (infini, NaN)
        if val != val or val == float("inf") or val == float("-inf"):
            val = None
    except (InvariantNotImplementedError, Exception):
        val = None
    _TEMPLATE_INV[key] = val
    return val


def _try_universal_templates(conjecture) -> Optional[nx.Graph]:
    """
    Batterie de graphes templates avec cache d'invariants.
    Premier appel pour (x_name, y_name) : ~30 ms (calcul invariants sur ~40 templates).
    Appels suivants : ~150 µs (pure arithmétique via cache).
    """
    templates = _get_template_graphs()
    sg = conjecture.subgroups
    x_name = conjecture.x_name
    y_name = conjecture.y_name

    best_viol = float("-inf")
    best_G = None
    for idx in range(len(templates)):
        # Filtre de classe (mis en cache)
        if not _template_in_class(idx, sg):
            continue
        x_val = _template_invariant(idx, x_name)
        y_val = _template_invariant(idx, y_name)
        if x_val is None or y_val is None:
            continue
        # Vérification arithmétique de la violation
        try:
            fx_val = conjecture.eval_f(x_val)
        except Exception:
            continue
        if conjecture.sign == "<=":
            viol = y_val - fx_val
        elif conjecture.sign == ">=":
            viol = fx_val - y_val
        else:
            continue
        if viol > best_viol:
            best_viol = viol
            best_G = templates[idx]
            if viol > 1e-9:
                return best_G
    return best_G if best_viol > float("-inf") else None


def _best_triangular_snake_claw_free(conjecture, rng: random.Random) -> Optional[nx.Graph]:
    """
    Triangular snake = chaîne de k triangles partageant une arête consécutive.
    Propriétés : claw_free ✓, radius = k//2, independence = k (les k sommets "wing").
    Idéal pour independence_number vs radius sur claw_free.

    Structure (k=6, 13 sommets) :
      v0-v1-v2-v3-v4-v5-v6 (chemin)
      wi adjacent à vi et v(i+1) pour i=0..5
    """
    from ..invariants.compute import compute_invariants as _compute_inv
    best_viol = float("-inf")
    best_G = None
    for k in range(3, 20):
        # Construire le triangular snake avec k triangles
        G = nx.Graph()
        # Sommets : v_0..v_k (chemin) + w_0..w_{k-1} (un par triangle)
        path_nodes = list(range(k + 1))
        wing_nodes = list(range(k + 1, 2 * k + 1))
        for i in range(k):
            G.add_edge(path_nodes[i], path_nodes[i + 1])   # arête du chemin
            G.add_edge(path_nodes[i], wing_nodes[i])        # v_i – w_i
            G.add_edge(path_nodes[i + 1], wing_nodes[i])    # v_{i+1} – w_i
        try:
            inv = _compute_inv(G, {conjecture.x_name, conjecture.y_name})
            viol = conjecture.violation(inv)
        except Exception:
            continue
        if viol > best_viol:
            best_viol = viol
            best_G = G.copy()
            if viol > 0:
                return best_G
    return best_G if best_viol > float("-inf") else None


def _best_for_proximity_total_dom(conjecture, rng: random.Random) -> Optional[nx.Graph]:
    """
    Pour proximity + total_domination_number.
    Essaie des graphes denses (K_n : proximity=1, total_dom=2) ET des chemins
    (P_n : proximity basse, total_dom≈n/3).
    """
    best_viol = float("-inf")
    best_G = None
    # Graphes denses : K_n
    for n in range(4, 20):
        G = nx.complete_graph(n)
        viol = _eval(G, conjecture)
        if viol is not None and viol > best_viol:
            best_viol = viol
            best_G = G.copy()
            if viol > 0:
                return best_G
    # Chemins : P_n (faible proximity, fort total_dom approx)
    for n in range(6, 30):
        G = nx.path_graph(n)
        viol = _eval(G, conjecture)
        if viol is not None and viol > best_viol:
            best_viol = viol
            best_G = G.copy()
            if viol > 0:
                return best_G
    return best_G if best_viol > float("-inf") else None


def _best_sparse_for_matching_dom(conjecture, rng: random.Random) -> Optional[nx.Graph]:
    """
    Pour matching_number + total_domination_number.
    Essaie des graphes connexes aléatoires de densité variée (n=10..20, p=0.2..0.4).
    """
    from ..invariants.compute import compute_invariants as _compute_inv
    best_viol = float("-inf")
    best_G = None
    for n in range(10, 22):
        for trial in range(4):
            p = 0.20 + trial * 0.06
            G = nx.gnp_random_graph(n, p, seed=rng.randint(0, 2**31))
            if not nx.is_connected(G):
                # Connecter les composantes
                comps = list(nx.connected_components(G))
                for ci in range(len(comps) - 1):
                    u = next(iter(comps[ci]))
                    v = next(iter(comps[ci + 1]))
                    G.add_edge(u, v)
            if G.number_of_edges() == 0:
                continue
            try:
                inv = _compute_inv(G, {conjecture.x_name, conjecture.y_name})
                viol = conjecture.violation(inv)
            except Exception:
                continue
            if viol > best_viol:
                best_viol = viol
                best_G = G.copy()
                if viol > 1e-9:
                    return best_G
    return best_G if best_viol > float("-inf") else None


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
