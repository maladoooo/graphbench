"""
Point d'entrée principal du projet GraphBench.

Usage:
    # Phase 0: tester sur une conjecture (par ID)
    python -m src.main --id 981 --time 60 --verbose

    # Phase 1: tester sur les conjectures 'connected'
    python -m src.main --subgroup connected --max 5 --time 60

    # Phase 2: batch complet
    python -m src.main --all --time 60 --output results/results.csv

    # Reprendre un batch : réutiliser les JSON déjà présents dans results/
    python -m src.main --all --time 60 --resume
"""
from __future__ import annotations
import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import List, Optional

from .benchmark.load_benchmark import load_benchmark, load_conjecture_by_id
from .funsearch.funsearch import FunSearch
from .benchmark.conjecture import Conjecture
from .search.search_simple import search, SearchResult
from .graphs.classes import check_graph_class
from .invariants.compute import InvariantNotImplementedError

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def run_single(
    conjecture: Conjecture,
    time_limit: float,
    verbose: bool,
    use_heuristic: bool = False,
) -> SearchResult:
    """Lance la recherche sur une seule conjecture et affiche le résultat."""
    print(f"\n{'='*60}")
    print(f"Conjecture #{conjecture.id}: {conjecture.description[:80]}...")
    print(f"Classe: {conjecture.subgroups}")
    print(f"X={conjecture.x_name}, Y={conjecture.y_name}, sign={conjecture.sign}")
    print(f"Limite de temps: {time_limit}s")

    result = search(
        conjecture,
        time_limit=time_limit,
        verbose=verbose,
        use_heuristic=use_heuristic,
    )

    print(f"\n--- Résultat ---")
    if result.found:
        print(f"✅ CONTRE-EXEMPLE TROUVÉ en {result.time_s:.2f}s !")
        print(f"graph6: {result.best_graph6}")
        print(result.proof)
        print(f"Invariants: {result.best_invariants}")
    else:
        print(f"❌ Pas de contre-exemple (temps: {result.time_s:.2f}s)")
        print(f"Meilleure violation: {result.best_violation:.4f}")
        if result.best_graph6:
            print(f"Meilleur graphe (graph6): {result.best_graph6}")

    return result


def _load_cached_result(path: Path, expected_id: int) -> Optional[SearchResult]:
    """Charge un SearchResult depuis un JSON existant si le fichier est valide."""
    if not path.is_file():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        cid = int(data["conjecture_id"])
        if cid != expected_id:
            return None
        inv = data.get("invariants") or {}
        return SearchResult(
            conjecture_id=cid,
            found=bool(data["found"]),
            time_s=float(data["time_s"]),
            best_violation=float(data["best_violation"]),
            best_graph=None,
            best_graph6=data.get("best_graph6") or "",
            best_invariants={k: float(v) for k, v in inv.items()},
            proof=data.get("proof") or "",
            cost=float(data.get("cost", 120.0)),
        )
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None


def run_batch(
    conjectures: List[Conjecture],
    time_limit: float,
    output_csv: Path,
    verbose: bool,
    use_heuristic: bool = False,
    stagnation_limit: Optional[int] = None,
    resume: bool = False,
) -> None:
    """Lance la recherche sur toutes les conjectures et sauvegarde les résultats."""
    results = []
    total_cost = 0.0
    found_count = 0
    cache_hits = 0

    for i, conj in enumerate(conjectures, 1):
        print(f"\n[{i}/{len(conjectures)}] Conjecture #{conj.id} ({conj.subgroups})")
        json_path = RESULTS_DIR / f"conjecture_{conj.id}.json"
        if resume:
            cached = _load_cached_result(json_path, conj.id)
            if cached is not None:
                result = cached
                cache_hits += 1
                results.append(result)
                total_cost += result.cost
                status = "✅ TROUVÉ" if result.found else "❌ échec"
                print(f"  ⏭️  Cache (--resume) | {status} | violation={result.best_violation:.4f} | t={result.time_s:.2f}s | coût={result.cost:.1f}")
                if result.found:
                    found_count += 1
                    print(f"  graph6: {result.best_graph6}")
                continue

        result = search(conj, time_limit=time_limit, verbose=verbose,
                        use_heuristic=use_heuristic, stagnation_limit=stagnation_limit)
        results.append(result)
        total_cost += result.cost

        status = "✅ TROUVÉ" if result.found else "❌ échec"
        print(f"  {status} | violation={result.best_violation:.4f} | t={result.time_s:.2f}s | coût={result.cost:.1f}")
        if result.found:
            found_count += 1
            print(f"  graph6: {result.best_graph6}")

        _save_result_json(result, conj, json_path)

    # Sauvegarde CSV global
    _save_results_csv(results, conjectures, output_csv)

    print(f"\n{'='*60}")
    print(f"RÉSULTATS FINAUX")
    print(f"  Conjectures réfutées: {found_count}/{len(conjectures)}")
    if resume and cache_hits:
        print(f"  Reprise cache: {cache_hits} conjecture(s) non recalculée(s)")
    print(f"  Score total: {total_cost:.1f}")
    print(f"  Résultats CSV: {output_csv}")


def _save_results_csv(
    results: List[SearchResult],
    conjectures: List[Conjecture],
    path: Path,
) -> None:
    """Sauvegarde les résultats dans un CSV."""
    conj_by_id = {c.id: c for c in conjectures}

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "conjecture_id", "subgroup", "x_name", "y_name", "sign",
            "found", "time_s", "best_violation", "best_graph6",
            "x_value", "y_value", "fx_value", "cost",
        ])
        writer.writeheader()
        for r in results:
            conj = conj_by_id.get(r.conjecture_id)
            x_val = r.best_invariants.get(conj.x_name, "") if conj else ""
            y_val = r.best_invariants.get(conj.y_name, "") if conj else ""
            fx_val = conj.eval_f(float(x_val)) if conj and x_val != "" else ""
            writer.writerow({
                "conjecture_id": r.conjecture_id,
                "subgroup": str(conj.subgroups) if conj else "",
                "x_name": conj.x_name if conj else "",
                "y_name": conj.y_name if conj else "",
                "sign": conj.sign if conj else "",
                "found": int(r.found),
                "time_s": round(r.time_s, 3),
                "best_violation": round(r.best_violation, 6),
                "best_graph6": r.best_graph6,
                "x_value": x_val,
                "y_value": y_val,
                "fx_value": fx_val,
                "cost": round(r.cost, 3),
            })


def _save_result_json(result: SearchResult, conjecture: Conjecture, path: Path) -> None:
    """Sauvegarde le résultat détaillé en JSON."""
    data = {
        "conjecture_id": result.conjecture_id,
        "description": conjecture.description,
        "subgroups": conjecture.subgroups,
        "x_name": conjecture.x_name,
        "y_name": conjecture.y_name,
        "sign": conjecture.sign,
        "found": result.found,
        "time_s": round(result.time_s, 3),
        "best_violation": round(result.best_violation, 6),
        "best_graph6": result.best_graph6,
        "invariants": {k: round(v, 6) for k, v in result.best_invariants.items()},
        "proof": result.proof,
        "cost": round(result.cost, 3),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GraphBench: réfutation automatique de conjectures en théorie des graphes"
    )
    parser.add_argument("--id", type=int, nargs="+", help="ID(s) de conjectures à tester")
    parser.add_argument("--subgroup", type=str, help="Filtrer par sous-groupe (ex: connected, tree)")
    parser.add_argument("--all", action="store_true", help="Tester toutes les conjectures")
    parser.add_argument("--max", type=int, default=None, help="Nombre maximum de conjectures à tester")
    parser.add_argument("--time", type=float, default=60.0, help="Limite de temps par conjecture (secondes)")
    parser.add_argument("--verbose", action="store_true", help="Afficher les logs de recherche")
    parser.add_argument("--heuristic", action="store_true", help="Utiliser le score heuristique (Phase 2)")
    parser.add_argument("--output", type=str, default="results/results.csv", help="Fichier CSV de sortie")
    parser.add_argument("--stagnation", type=int, default=None, help="Arrêt anticipé si pas d'amélioration après N itérations (ex: 2000)")
    parser.add_argument("--resume", action="store_true", help="En mode batch : réutiliser results/conjecture_<id>.json si présent au lieu de recalculer")
    parser.add_argument("--funsearch", action="store_true", help="Lancer la Phase 2 FunSearch (évolution LLM du score)")
    parser.add_argument("--funsearch-iter", type=int, default=5, help="Nombre d'itérations FunSearch (défaut: 5)")
    parser.add_argument("--funsearch-time", type=float, default=15.0, help="Temps par conjecture lors de l'évaluation FunSearch (défaut: 15s)")
    parser.add_argument("--api-key", type=str, default=None, help="Clé API Anthropic (ou variable ANTHROPIC_API_KEY)")

    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(exist_ok=True)

    if args.id:
        if len(args.id) == 1:
            conj = load_conjecture_by_id(args.id[0])
            run_single(conj, time_limit=args.time, verbose=args.verbose, use_heuristic=args.heuristic)
        else:
            conjectures = load_benchmark(ids=args.id)
            run_batch(
                conjectures,
                time_limit=args.time,
                output_csv=output_path,
                verbose=args.verbose,
                use_heuristic=args.heuristic,
                stagnation_limit=args.stagnation,
                resume=args.resume,
            )

    elif args.funsearch:
        # Phase 2 : FunSearch — doit être AVANT le batch normal
        subgroup_filter = [args.subgroup] if args.subgroup else None
        conjectures = load_benchmark(subgroup_filter=subgroup_filter)
        if args.max:
            conjectures = conjectures[:args.max]
        print(f"FunSearch sur {len(conjectures)} conjectures | {args.funsearch_iter} itérations")
        fs = FunSearch(
            conjectures=conjectures,
            n_iterations=args.funsearch_iter,
            eval_time_limit=args.funsearch_time,
            output_dir=output_path.parent / "funsearch",
            api_key=args.api_key,
            verbose=args.verbose,
        )
        fs.run()

    elif args.all or args.subgroup or args.max:
        # Phase 1 : batch normal
        subgroup_filter = [args.subgroup] if args.subgroup else None
        conjectures = load_benchmark(subgroup_filter=subgroup_filter)
        if args.max:
            conjectures = conjectures[:args.max]
        print(f"Chargement de {len(conjectures)} conjectures.")
        run_batch(
            conjectures,
            time_limit=args.time,
            output_csv=output_path,
            verbose=args.verbose,
            use_heuristic=args.heuristic,
            stagnation_limit=args.stagnation,
            resume=args.resume,
        )

    else:
        parser.print_help()
        print("\nExemples:")
        print("  python -m src.main --id 981 --time 30 --verbose")
        print("  python -m src.main --subgroup connected --max 5 --time 60")
        print("  python -m src.main --all --time 60 --output results/results.csv")
        print("  python -m src.main --subgroup connected --max 10 --time 60 --resume")


if __name__ == "__main__":
    main()
