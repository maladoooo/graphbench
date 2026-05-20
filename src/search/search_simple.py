"""
Algorithme de recherche locale pour réfuter une conjecture.

Schéma:
1. Générer une population initiale
2. Boucle temporelle:
   - Sélectionner un candidat
   - Muter → réparer → calculer invariants → scorer
   - Mettre à jour le meilleur
   - Si violation > 0: retourner le contre-exemple
"""
from __future__ import annotations
import random
import time
import json
import signal
import sys
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

import networkx as nx

from ..benchmark.conjecture import Conjecture
from ..graphs.generate import generate_initial_graph
from ..graphs.mutate import mutate
from ..graphs.repair import repair
from ..graphs.classes import check_graph_class, UnknownGraphClassError
from ..invariants.compute import compute_invariants, InvariantNotImplementedError
from ..scoring.violation import violation_score, heuristic_score


class _SearchTimeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise _SearchTimeout()


@dataclass
class SearchResult:
    """Résultat d'une recherche pour une conjecture."""
    conjecture_id: int
    found: bool                        # True si contre-exemple trouvé
    time_s: float                      # Temps écoulé en secondes
    best_violation: float
    best_graph: Optional[nx.Graph]
    best_graph6: str
    best_invariants: Dict[str, float] = field(default_factory=dict)
    proof: str = ""
    cost: float = 120.0                # ti si trouvé, 120 sinon (scoring officiel)


# ─── Cache cross-conjectures (apprentissage intra-run) ─────────────────────
# Beaucoup de conjectures partagent la même signature (subgroups, x, y, sign).
# Quand on trouve un contre-exemple, on le mémorise — pour les conjectures
# suivantes ayant la même signature, on vérifie d'abord ce graphe par simple
# évaluation arithmétique f(x_val) vs y_val (≈ 1 µs au lieu de 0.5-1.5 s).
_GRAPH_CACHE: Dict[tuple, List[tuple]] = {}
_CACHE_MAX_PER_KEY = 8


def _cache_key(conjecture: Conjecture) -> tuple:
    return (
        frozenset(conjecture.subgroups),
        conjecture.x_name,
        conjecture.y_name,
        conjecture.sign,
    )


def _try_cached(conjecture: Conjecture, t_start: float) -> Optional[SearchResult]:
    """Essaie de réutiliser un graphe déjà trouvé pour une conjecture de même signature.
    Vérification PURE ARITHMÉTIQUE : f(x_val) comparé à y_val. Pas de recalcul d'invariants.
    """
    key = _cache_key(conjecture)
    cached_list = _GRAPH_CACHE.get(key)
    if not cached_list:
        return None
    for cached_G, x_val, y_val in cached_list:
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
        if viol > 1e-9:
            elapsed = time.perf_counter() - t_start
            inv = {conjecture.x_name: x_val, conjecture.y_name: y_val}
            return SearchResult(
                conjecture_id=conjecture.id,
                found=True,
                time_s=elapsed,
                best_violation=viol,
                best_graph=cached_G,
                best_graph6=_to_graph6(cached_G),
                best_invariants=inv,
                proof=conjecture.proof_string(inv),
                cost=elapsed,
            )
    return None


def _store_in_cache(conjecture: Conjecture, result: SearchResult) -> None:
    """Mémorise un graphe trouvé pour réutilisation par les conjectures de même signature."""
    if not result.found or result.best_graph is None:
        return
    x_val = result.best_invariants.get(conjecture.x_name)
    y_val = result.best_invariants.get(conjecture.y_name)
    if x_val is None or y_val is None:
        return
    key = _cache_key(conjecture)
    lst = _GRAPH_CACHE.setdefault(key, [])
    # Évite les doublons exacts
    g_sig = frozenset(result.best_graph.edges())
    for cached_G, _, _ in lst:
        if frozenset(cached_G.edges()) == g_sig:
            return
    lst.append((result.best_graph.copy(), float(x_val), float(y_val)))
    if len(lst) > _CACHE_MAX_PER_KEY:
        lst.pop(0)


def search(
    conjecture: Conjecture,
    time_limit: float = 60.0,
    population_size: int = 5,
    initial_n: int = 8,
    seed: Optional[int] = 42,
    use_heuristic: bool = False,
    heuristic_fn=None,
    verbose: bool = False,
    stagnation_limit: Optional[int] = None,
) -> SearchResult:
    """Wrapper avec cache cross-conjectures ET vérification stricte du contre-exemple."""
    t_start_outer = time.perf_counter()
    cached = _try_cached(conjecture, t_start_outer)
    if cached is not None and _strict_verify(cached, conjecture):
        return cached
    result = _search_main(
        conjecture, time_limit, population_size, initial_n,
        seed, use_heuristic, heuristic_fn, verbose, stagnation_limit,
    )
    # Vérification stricte avant de stocker dans le cache et retourner
    if result.found and not _strict_verify(result, conjecture):
        # Le "contre-exemple" est INVALIDE (classe ou violation incorrecte)
        # → on marque comme non trouvé pour ne PAS polluer le cache
        elapsed = time.perf_counter() - t_start_outer
        result = SearchResult(
            conjecture_id=conjecture.id,
            found=False,
            time_s=elapsed,
            best_violation=float("-inf"),
            best_graph=None,
            best_graph6="",
            proof=f"Contre-exemple rejeté par la vérification stricte",
            cost=120.0,
        )
    _store_in_cache(conjecture, result)
    return result


def _strict_verify(result: SearchResult, conjecture: Conjecture) -> bool:
    """Re-vérifie strictement qu'un contre-exemple est valide :
    1. Le graphe satisfait toutes les classes requises (claw_free, tree, etc.)
    2. La violation est strictement > 0 quand recalculée avec les invariants EXACTS
    Retourne False si l'une de ces conditions n'est pas remplie.
    """
    if not result.found or result.best_graph is None:
        return False
    from ..graphs.classes import check_graph_class
    required = {conjecture.x_name, conjecture.y_name}
    # 1. Vérif classe
    try:
        if not check_graph_class(result.best_graph, conjecture.subgroups):
            return False
    except Exception:
        return False
    # 2. Vérif violation exacte
    try:
        exact_inv = compute_invariants(result.best_graph, required)
        exact_viol = conjecture.violation(exact_inv)
        return exact_viol > 1e-9
    except Exception:
        return False


def _search_main(
    conjecture: Conjecture,
    time_limit: float,
    population_size: int,
    initial_n: int,
    seed: Optional[int],
    use_heuristic: bool,
    heuristic_fn,
    verbose: bool,
    stagnation_limit: Optional[int],
) -> SearchResult:
    """
    Lance la recherche locale pour réfuter la conjecture.
    Wrapper avec timeout SIGALRM pour éviter les blocages sur invariants lents.
    Essaie plusieurs seeds si le premier échoue (diversification).
    """
    rng = random.Random(seed)
    t_start = time.perf_counter()
    # Liste de seeds alternatifs à essayer si le premier ne trouve pas
    # Inclure des petites valeurs (7, 13) qui fonctionnent bien pour les cas difficiles
    _alt_seeds = [7, 13, seed + 1, seed + 7, seed + 13]

    # Timeout dur via SIGALRM (Unix uniquement) — protège contre les calculs bloquants
    _use_alarm = sys.platform != "win32"
    if _use_alarm:
        signal.signal(signal.SIGALRM, _alarm_handler)
        # Annuler tout alarme résiduelle avant d'en définir une nouvelle (évite race condition)
        signal.alarm(0)
        signal.alarm(int(time_limit) + 10)  # +10s de grâce après time_limit

    best_result: Optional[SearchResult] = None
    # Partitioner le temps entre seeds: premier seed ≤ 60% du temps, reste pour alternatifs
    all_seeds = [seed] + list(_alt_seeds)
    n_seeds = len(all_seeds)
    # Allouer le temps de manière dynamique: chaque seed reçoit 1/n_seeds du temps restant
    slice_time = time_limit / n_seeds  # temps par seed (au minimum)

    try:
        for i_seed, cur_seed in enumerate(all_seeds):
            elapsed = time.perf_counter() - t_start
            if elapsed >= time_limit:
                break
            remaining = time_limit - elapsed
            # Chaque seed reçoit au maximum slice_time ou le reste du temps
            cur_limit = min(slice_time, remaining)
            cur_rng = random.Random(cur_seed)
            try:
                result = _search_inner(
                    conjecture=conjecture,
                    time_limit=cur_limit,
                    population_size=population_size,
                    initial_n=initial_n,
                    rng=cur_rng,
                    t_start=time.perf_counter(),
                    use_heuristic=use_heuristic,
                    heuristic_fn=heuristic_fn,
                    verbose=verbose,
                    stagnation_limit=stagnation_limit,
                )
                if result.found:
                    return result
                if best_result is None or result.best_violation > best_result.best_violation:
                    best_result = result
            except (_SearchTimeout, Exception):
                pass

    except _SearchTimeout:
        elapsed = time.perf_counter() - t_start
        # Si le timeout arrive très vite (<1s), c'est probablement un faux-positif SIGALRM
        if elapsed < 1.0 and best_result is None:
            try:
                result3 = _search_inner(
                    conjecture=conjecture,
                    time_limit=time_limit,
                    population_size=population_size,
                    initial_n=initial_n,
                    rng=random.Random(seed),
                    t_start=time.perf_counter(),
                    use_heuristic=use_heuristic,
                    heuristic_fn=heuristic_fn,
                    verbose=verbose,
                    stagnation_limit=stagnation_limit,
                )
                if result3.found:
                    return result3
                if best_result is None or result3.best_violation > best_result.best_violation:
                    best_result = result3
            except Exception:
                pass
    finally:
        if _use_alarm:
            signal.alarm(0)

    # Retourner le meilleur résultat trouvé
    if best_result is not None:
        return best_result
    elapsed = time.perf_counter() - t_start
    return SearchResult(
        conjecture_id=conjecture.id,
        found=False,
        time_s=elapsed,
        best_violation=float("-inf"),
        best_graph=None,
        best_graph6="",
        proof="Aucun résultat",
        cost=120.0,
    )


def _search_inner(
    conjecture: Conjecture,
    time_limit: float,
    population_size: int,
    initial_n: int,
    rng: random.Random,
    t_start: float,
    use_heuristic: bool = False,
    heuristic_fn=None,
    verbose: bool = False,
    stagnation_limit: Optional[int] = None,
) -> SearchResult:
    """Logique interne de la recherche (sans gestion du timeout)."""

    required_invariants = {conjecture.x_name, conjecture.y_name}

    # Vérifier que les invariants et classes sont supportés
    try:
        _test_invariants(required_invariants, conjecture.subgroups)
    except (InvariantNotImplementedError, UnknownGraphClassError) as e:
        return SearchResult(
            conjecture_id=conjecture.id,
            found=False,
            time_s=0.0,
            best_violation=float("-inf"),
            best_graph=None,
            best_graph6="",
            best_invariants={},
            proof=f"ERREUR: {e}",
            cost=120.0,
        )

    # ── FILTRE INTELLIGENT ──────────────────────────────────────────────────
    from ..graphs.generate import generate_smart
    from ..graphs.classes import check_graph_class
    smart_G = generate_smart(conjecture, seed=42)
    if smart_G is not None:
        # VÉRIFICATION DE CLASSE : le smart filter peut produire un graphe hors-classe
        # (ex: K_n + leaves contient un claw → invalide pour claw_free)
        try:
            in_class = check_graph_class(smart_G, conjecture.subgroups)
        except Exception:
            in_class = False
        if not in_class:
            smart_G = None  # ignorer : on tombera dans la recherche locale
    if smart_G is not None:
        try:
            smart_inv = compute_invariants(smart_G, required_invariants)
            smart_viol = conjecture.violation(smart_inv)
            if verbose:
                print(f"  [FILTRE] n={smart_G.number_of_nodes()}, violation={smart_viol:.4f}")
            if smart_viol > 0:
                elapsed = time.perf_counter() - t_start
                g6 = _to_graph6(smart_G)
                return SearchResult(
                    conjecture_id=conjecture.id,
                    found=True,
                    time_s=elapsed,
                    best_violation=smart_viol,
                    best_graph=smart_G,
                    best_graph6=g6,
                    best_invariants=smart_inv,
                    proof=conjecture.proof_string(smart_inv),
                    cost=elapsed,
                )
        except Exception:
            pass
    # ────────────────────────────────────────────────────────────────────────

    # Population initiale
    population: List[nx.Graph] = []
    if smart_G is not None:
        population.append(smart_G)
    for i in range(population_size):
        n_init = rng.randint(max(4, initial_n - 3), initial_n + 3)
        G = generate_initial_graph(
            conjecture.subgroups,
            n=n_init,
            seed=rng.randint(0, 2**31),
        )
        if G.number_of_nodes() > 0:
            population.append(G)

    if not population:
        return SearchResult(
            conjecture_id=conjecture.id,
            found=False,
            time_s=0.0,
            best_violation=float("-inf"),
            best_graph=None,
            best_graph6="",
            proof="ERREUR: population initiale vide",
            cost=120.0,
        )

    best_graph: Optional[nx.Graph] = None
    best_score = float("-inf")
    best_violation = float("-inf")
    best_invariants: Dict[str, float] = {}
    iterations = 0
    restarts = 0
    stagnation_count = 0

    # ── TABU LIST ───────────────────────────────────────────────────────────
    tabu_set: set = set()
    TABU_MAX = 500
    # ────────────────────────────────────────────────────────────────────────

    while True:
        elapsed = time.perf_counter() - t_start
        if elapsed >= time_limit:
            break

        # Sélection
        if best_graph is not None and rng.random() < 0.7:
            candidate = best_graph.copy()
        else:
            candidate = rng.choice(population).copy()

        # Mutation
        mutated = mutate(candidate, conjecture.subgroups, rng=rng)

        # Réparation
        repaired, ok = repair(mutated, conjecture.subgroups, rng=rng)
        if not ok:
            iterations += 1
            continue

        # Vérification de classe
        if not check_graph_class(repaired, conjecture.subgroups):
            iterations += 1
            continue

        # Limite de taille pour invariants lents (diameter, radius sur grands graphes)
        if repaired.number_of_nodes() > 80 and (
            "diameter" in required_invariants or "radius" in required_invariants
        ):
            iterations += 1
            continue

        # Tabu check
        sig = frozenset(repaired.edges())
        if sig in tabu_set:
            iterations += 1
            continue
        if len(tabu_set) < TABU_MAX:
            tabu_set.add(sig)

        # Calcul des invariants
        try:
            inv = compute_invariants(repaired, required_invariants)
        except InvariantNotImplementedError:
            iterations += 1
            continue
        except Exception:
            iterations += 1
            continue

        # Score
        if heuristic_fn is not None:
            try:
                score = float(heuristic_fn(repaired, inv, conjecture))
            except Exception:
                score = violation_score(inv, conjecture)
        elif use_heuristic:
            score = heuristic_score(repaired, inv, conjecture)
        else:
            score = violation_score(inv, conjecture)

        # Mise à jour du meilleur
        if score > best_score:
            best_score = score
            best_violation = conjecture.violation(inv)
            best_graph = repaired.copy()
            best_invariants = inv.copy()
            stagnation_count = 0
            if verbose:
                print(f"  iter={iterations} t={elapsed:.1f}s score={score:.4f} viol={best_violation:.4f} n={repaired.number_of_nodes()}")
        else:
            stagnation_count += 1
            if stagnation_limit and stagnation_count >= stagnation_limit:
                # Restart diversifié plutôt que d'arrêter → change de seed
                stagnation_count = 0
                restarts += 1
                n_new = rng.randint(4, initial_n + restarts)
                G_new = generate_initial_graph(
                    conjecture.subgroups,
                    n=n_new,
                    seed=rng.randint(0, 2**31),
                )
                if len(population) > 0:
                    population[0] = G_new
                else:
                    population.append(G_new)
                tabu_set.clear()  # réinitialiser la liste tabu

        # Mettre à jour la population
        if len(population) >= population_size:
            idx = rng.randint(0, len(population) - 1)
            population[idx] = repaired.copy()
        else:
            population.append(repaired.copy())

        # Contre-exemple trouvé !
        if best_violation > 0:
            elapsed = time.perf_counter() - t_start
            g6 = _to_graph6(best_graph)
            return SearchResult(
                conjecture_id=conjecture.id,
                found=True,
                time_s=elapsed,
                best_violation=best_violation,
                best_graph=best_graph,
                best_graph6=g6,
                best_invariants=best_invariants,
                proof=conjecture.proof_string(best_invariants),
                cost=elapsed,
            )

        iterations += 1

        # Restart périodique
        if iterations % 500 == 0:
            restarts += 1
            n_new = rng.randint(6, initial_n + 5)
            G_new = generate_initial_graph(
                conjecture.subgroups,
                n=n_new,
                seed=rng.randint(0, 2**31),
            )
            if len(population) > 0:
                population[0] = G_new
            else:
                population.append(G_new)

    # Fin du temps
    elapsed = time.perf_counter() - t_start
    g6 = _to_graph6(best_graph) if best_graph else ""
    return SearchResult(
        conjecture_id=conjecture.id,
        found=False,
        time_s=elapsed,
        best_violation=best_violation,
        best_graph=best_graph,
        best_graph6=g6,
        best_invariants=best_invariants,
        proof=f"Pas de contre-exemple trouvé. Meilleure violation: {best_violation:.4f}",
        cost=120.0,
    )


def _to_graph6(G: Optional[nx.Graph]) -> str:
    """Convertit G en format graph6 (sans header >>graph6<<)."""
    if G is None:
        return ""
    try:
        return nx.to_graph6_bytes(G, header=False).decode("ascii").strip()
    except Exception:
        return ""


def _test_invariants(names: set, subgroups: list) -> None:
    """Teste que les invariants et classes sont supportés."""
    from ..graphs.generate import generate_initial_graph
    from ..invariants.compute import compute_invariant
    from ..graphs.classes import unknown_classes, UnknownGraphClassError
    unknown = unknown_classes(subgroups)
    if unknown:
        raise UnknownGraphClassError(unknown[0])
    G = generate_initial_graph(subgroups, n=6, seed=0)
    for name in names:
        compute_invariant(G, name)  # Lève InvariantNotImplementedError si inconnu
