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
import io
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

import networkx as nx

from ..benchmark.conjecture import Conjecture
from ..graphs.generate import generate_initial_graph
from ..graphs.mutate import mutate
from ..graphs.repair import repair
from ..graphs.classes import check_graph_class
from ..invariants.compute import compute_invariants, InvariantNotImplementedError
from ..scoring.violation import violation_score, heuristic_score


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


def search(
    conjecture: Conjecture,
    time_limit: float = 60.0,
    population_size: int = 5,
    initial_n: int = 8,
    seed: Optional[int] = 42,
    use_heuristic: bool = False,
    verbose: bool = False,
    stagnation_limit: Optional[int] = None,
) -> SearchResult:
    """
    Lance la recherche locale pour réfuter la conjecture.

    Args:
        conjecture: la conjecture à réfuter
        time_limit: limite de temps en secondes
        population_size: taille de la population
        initial_n: taille des graphes initiaux
        seed: graine pour la reproductibilité
        use_heuristic: si True, utilise heuristic_score au lieu de violation_score
        verbose: si True, affiche des logs

    Returns:
        SearchResult avec le meilleur graphe trouvé.
    """
    rng = random.Random(seed)
    t_start = time.perf_counter()

    required_invariants = {conjecture.x_name, conjecture.y_name}

    # Vérifier que les invariants sont implémentés
    try:
        _test_invariants(required_invariants, conjecture.subgroups)
    except InvariantNotImplementedError as e:
        return SearchResult(
            conjecture_id=conjecture.id,
            found=False,
            time_s=0.0,
            best_violation=float("-inf"),
            best_graph=None,
            best_graph6="",
            proof=f"ERREUR: {e}",
            cost=120.0,
        )

    # ── FILTRE INTELLIGENT ──────────────────────────────────────────────────
    # Génère un graphe structurellement optimal AVANT toute mutation.
    # Si c'est déjà un contre-exemple, on retourne immédiatement.
    from ..graphs.generate import generate_smart
    smart_G = generate_smart(conjecture, seed=seed)
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
            pass  # filtre échoué → on continue normalement
    # ────────────────────────────────────────────────────────────────────────

    # Population initiale (graphe intelligent en tête si disponible)
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
    # On mémorise les signatures (frozenset d'arêtes) des graphes déjà visités.
    # Si on retombe sur un graphe déjà vu, on le saute.
    tabu_set: set = set()
    TABU_MAX = 500   # taille max de la liste tabu (évite la surcharge mémoire)
    # ────────────────────────────────────────────────────────────────────────

    while True:
        elapsed = time.perf_counter() - t_start
        if elapsed >= time_limit:
            break

        # Sélection: alterner entre meilleur et aléatoire
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

        # ── TABU CHECK ──────────────────────────────────────────────────────
        # Signature du graphe = ensemble de ses arêtes (indépendant du nommage)
        sig = frozenset(repaired.edges())
        if sig in tabu_set:
            iterations += 1
            continue   # déjà visité → on saute
        if len(tabu_set) < TABU_MAX:
            tabu_set.add(sig)
        # ────────────────────────────────────────────────────────────────────

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
        if use_heuristic:
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
                break

        # Mettre à jour la population (remplacement du pire)
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

        # Restart périodique pour éviter les minima locaux
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

    # Fin du temps: retourner le meilleur candidat
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
    """Convertit G en format graph6."""
    if G is None:
        return ""
    try:
        buf = io.BytesIO()
        nx.readwrite.write_graph6(G, buf)
        buf.seek(0)
        return buf.read().decode("ascii").strip()
    except Exception:
        return ""


def _test_invariants(names: set, subgroups: list) -> None:
    """Teste que les invariants sont calculables sur un graphe minimal."""
    from ..graphs.generate import generate_initial_graph
    from ..invariants.compute import compute_invariant
    G = generate_initial_graph(subgroups, n=6, seed=0)
    for name in names:
        compute_invariant(G, name)  # Lève InvariantNotImplementedError si inconnu
