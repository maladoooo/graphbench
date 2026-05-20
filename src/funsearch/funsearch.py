"""
Boucle principale FunSearch.

Architecture (inspirée de DeepMind FunSearch, 2023) :

    1. Seed : on part d'une fonction de score minimale (violation seule)
    2. Évaluation : on teste cette fonction sur un sous-ensemble de conjectures
    3. Génération : le LLM propose une variante améliorée
    4. Sélection : on garde la meilleure fonction
    5. Répétition : le LLM voit les meilleures fonctions → s'améliore

Pourquoi ça marche ?
- En Phase 1, score = violation nous dit COMBIEN on est loin du contre-exemple
- En Phase 2, le LLM ajoute des bonus/pénalités qui guident la recherche
  AVANT même d'atteindre violation > 0
  Ex: "graphe avec grand diamètre → probablement bon pour cette conjecture"
"""
from __future__ import annotations
import json
import time
from pathlib import Path
from typing import List, Optional

from ..benchmark.conjecture import Conjecture
from ..benchmark.load_benchmark import load_benchmark
from .evaluate import SEED_FUNCTION, BASELINE_FUNCTION, compile_heuristic, evaluate_heuristic
from .llm_client import build_prompt, build_crossover_prompt, call_llm, extract_function_code


class FunSearch:
    """
    Gère la boucle évolutive FunSearch.

    Attributs:
        population : liste des meilleures fonctions trouvées
                     chaque entrée = {"code": str, "score": float, "found": int}
        best_fn    : la meilleure fonction compilée actuellement
    """

    def __init__(
        self,
        conjectures: List[Conjecture],
        n_iterations: int = 5,
        eval_time_limit: float = 5.0,
        max_eval_conjectures: int = 10,  # max conjectures évaluées par itération
        output_dir: Path = Path("results/funsearch"),
        api_key: Optional[str] = None,
        verbose: bool = True,
    ):
        self.conjectures = conjectures
        self.n_iterations = n_iterations
        self.eval_time_limit = eval_time_limit
        self.max_eval_conjectures = max_eval_conjectures
        self.output_dir = output_dir
        self.api_key = api_key
        self.verbose = verbose
        self.population: List[dict] = []
        self.best_fn = None
        output_dir.mkdir(parents=True, exist_ok=True)

    def run(self):
        """Lance la boucle FunSearch complète."""
        print(f"\n{'='*60}")
        print("🔬 FUNSEARCH — Évolution automatique de la fonction de score")
        print(f"   {len(self.conjectures)} conjectures | {self.n_iterations} itérations max")
        print(f"{'='*60}\n")

        # --- Étape 1 : identifier les conjectures "difficiles" via le filtre rapide ---
        # On vérifie juste si le filtre intelligent trouve immédiatement un CE (sans mutation).
        # Celles résolues en <0.1s par le filtre sont exclues — inutile d'optimiser leur score.
        hard_conjectures = self._select_hard_conjectures_fast()
        print(f"📌 Conjectures difficiles : {len(hard_conjectures)}/{len(self.conjectures)}")
        if not hard_conjectures:
            print("  ✅ Toutes les conjectures sont triviales. FunSearch inutile.")
            return

        # Limiter le nombre de conjectures évaluées pour rester rapide
        import random
        random.seed(42)
        if len(hard_conjectures) > self.max_eval_conjectures:
            self._eval_conjectures = random.sample(hard_conjectures, self.max_eval_conjectures)
            print(f"📌 Échantillon pour évaluation : {self.max_eval_conjectures} conjectures")
        else:
            self._eval_conjectures = hard_conjectures
        print(f"   IDs : {[c.id for c in self._eval_conjectures]}")

        # --- Étape 2 : évaluer les 2 seeds (violation pure + baseline TP) ---
        # On démarre avec une POPULATION INITIALE de 2 fonctions, pas 1.
        # Cela donne au LLM 2 références pour ses variantes au lieu d'une seule.
        print("\n📌 Génération 0 — Seeds initiaux")
        seed_result = self._evaluate_and_register(SEED_FUNCTION, label="G0-seed-pure")
        if seed_result is None:
            print("  ❌ Erreur sur la fonction de base. Abandon.")
            return
        baseline_result = self._evaluate_and_register(BASELINE_FUNCTION, label="G0-baseline-TP")
        if baseline_result is None:
            print("  ⚠️  Baseline TP non évaluée (continue avec seed pure).")

        # Si déjà 100% résolues → inutile d'appeler le LLM
        best_initial = self.population[0]
        if best_initial["found"] == len(hard_conjectures):
            print("  ✅ 100% résolues par un seed. FunSearch convergé d'emblée.")
            self._save_best()
            return

        # --- Étape 3 : boucle évolutive avec early stopping ---
        no_improve_count = 0
        MAX_NO_IMPROVE = 3   # arrêt si 3 itérations consécutives sans amélioration
        seen_codes = {SEED_FUNCTION}  # évite de réévaluer la même fonction

        for iteration in range(1, self.n_iterations + 1):
            print(f"\n🔄 Génération {iteration}/{self.n_iterations}")

            # Early stopping si stagnation
            if no_improve_count >= MAX_NO_IMPROVE:
                print(f"  ⏹  Arrêt anticipé : {no_improve_count} générations sans amélioration.")
                break

            best_before = self.population[0]["found"] if self.population else 0
            score_before = self.population[0]["score"] if self.population else float("inf")
            improved_this_gen = False

            # === Opérateur 1 : MUTATION (LLM voit les meilleures et propose une variante) ===
            mutation_code = self._propose_with_retry(seen_codes, mode="mutation")
            if mutation_code is not None:
                seen_codes.add(mutation_code)
                print(f"  💡 [mutation] fonction proposée ({len(mutation_code)} chars)")
                r = self._evaluate_and_register(mutation_code, label=f"G{iteration}-mut")
                if r is not None and (r["found"] > best_before or
                                       (r["found"] == best_before and r["score"] < score_before)):
                    improved_this_gen = True

            # === Opérateur 2 : CROSSOVER (combine les 2 meilleures fonctions) ===
            if len(self.population) >= 2:
                crossover_code = self._propose_with_retry(seen_codes, mode="crossover")
                if crossover_code is not None:
                    seen_codes.add(crossover_code)
                    print(f"  🧬 [crossover] fonction proposée ({len(crossover_code)} chars)")
                    r = self._evaluate_and_register(crossover_code, label=f"G{iteration}-cross")
                    best_after = self.population[0]["found"]
                    score_after = self.population[0]["score"]
                    if r is not None and (best_after > best_before or
                                           (best_after == best_before and score_after < score_before)):
                        improved_this_gen = True

            # Bilan génération
            if improved_this_gen:
                new_best = self.population[0]
                print(f"  🎉 GÉNÉRATION {iteration} : amélioration → {new_best['found']} réfutées, coût={new_best['score']:.1f}")
                no_improve_count = 0
            else:
                no_improve_count += 1
                print(f"  ~ Génération {iteration} : pas d'amélioration ({no_improve_count}/{MAX_NO_IMPROVE})")

            # 100% résolues → fin
            if self.population[0]["found"] == len(self._eval_conjectures):
                print("  ✅ 100% des conjectures-test résolues !")
                break

        # --- Résultat final ---
        self._print_summary()
        self._save_best()

    def _evaluate_and_register(self, code: str, label: str) -> Optional[dict]:
        """Compile, évalue et enregistre une fonction dans la population."""
        fn = compile_heuristic(code)
        if fn is None:
            return None

        print(f"  ⏱  Évaluation en cours...")
        t0 = time.perf_counter()
        metrics = evaluate_heuristic(
            fn,
            self._eval_conjectures,
            time_limit=self.eval_time_limit,
            verbose=self.verbose,
        )
        elapsed = time.perf_counter() - t0

        entry = {
            "label": label,
            "code": code,
            "found": metrics["found"],
            "score": metrics["total_cost"],   # plus bas = meilleur
            "avg_time": metrics["avg_time"],
            "eval_elapsed": round(elapsed, 1),
        }

        self.population.append(entry)
        # Trier : plus de conjectures trouvées ET coût plus faible = meilleur
        self.population.sort(key=lambda x: (-(x["found"]), x["score"]))

        # Mettre à jour la meilleure fonction compilée
        best_code = self.population[0]["code"]
        self.best_fn = compile_heuristic(best_code)

        print(f"  📊 Résultat [{label}] : {metrics['found']}/{len(self.conjectures)} trouvées | coût={metrics['total_cost']:.1f} | temps éval={elapsed:.1f}s")
        return entry

    def _select_hard_conjectures_fast(self) -> List[Conjecture]:
        """
        Sélection rapide : on teste uniquement le filtre intelligent (sans mutation).
        Si le filtre trouve un CE immédiatement → conjecture triviale, on l'exclut.
        Coût : quelques ms par conjecture au lieu de 2s.
        """
        from ..graphs.generate import generate_smart
        from ..invariants.compute import compute_invariants
        hard = []
        for c in self.conjectures:
            try:
                G = generate_smart(c, seed=42)
                if G is None:
                    hard.append(c)  # pas de filtre → difficile
                    continue
                inv = compute_invariants(G, {c.x_name, c.y_name})
                viol = c.violation(inv)
                if viol <= 0:
                    hard.append(c)  # filtre ne suffit pas → difficile
            except Exception:
                hard.append(c)
        return hard

    def _ask_llm(self, mode: str = "mutation") -> Optional[str]:
        """Demande au LLM de générer une nouvelle fonction de score.
        mode='mutation' : variante des meilleures fonctions
        mode='crossover' : enfant qui combine les 2 meilleures (parents)
        """
        conj_descriptions = [
            f"#{c.id}: {c.y_name} {c.sign} f({c.x_name}) sur {c.subgroups}"
            for c in self._eval_conjectures[:8]
        ]

        if mode == "crossover" and len(self.population) >= 2:
            prompt = build_crossover_prompt(
                parent_a=self.population[0],
                parent_b=self.population[1],
                conjecture_descriptions=conj_descriptions,
            )
        else:
            prompt = build_prompt(
                best_functions=self.population[:3],
                conjecture_descriptions=conj_descriptions,
            )

        raw = call_llm(prompt, api_key=self.api_key)
        if raw is None:
            return None

        code = extract_function_code(raw)
        if code is None:
            print(f"  ⚠️  Impossible d'extraire la fonction du output LLM")
            if self.verbose:
                print(f"  Output LLM brut:\n{raw[:300]}")
        return code

    def _propose_with_retry(self, seen_codes: set, mode: str = "mutation") -> Optional[str]:
        """Demande au LLM avec retry sur duplicats / syntaxe invalide."""
        for _attempt in range(2):
            candidate = self._ask_llm(mode=mode)
            if candidate is None:
                return None
            if candidate in seen_codes:
                continue
            if compile_heuristic(candidate) is not None:
                return candidate
        return None

    def _print_summary(self):
        """Affiche la timeline d'évolution + classement final des fonctions."""
        n_eval = len(self._eval_conjectures)
        print(f"\n{'='*65}")
        print(f"📈 TIMELINE D'ÉVOLUTION FUNSEARCH — {n_eval} conjectures-test")
        print(f"{'='*65}")
        # Recalculer la trace dans l'ordre d'insertion (pas trié par score)
        # NB: self.population est trié par perf, donc on garde un journal séparé
        # Ici on affiche tous les essais avec leur label de génération
        history_by_label = sorted(self.population, key=lambda e: e["label"])
        running_best = -1
        running_best_score = float("inf")
        for entry in history_by_label:
            improved = ""
            if entry["found"] > running_best or (entry["found"] == running_best and entry["score"] < running_best_score):
                running_best = entry["found"]
                running_best_score = entry["score"]
                improved = "  ⬆️ NEW BEST"
            print(
                f"  [{entry['label']:18s}] réfutées={entry['found']:>2}/{n_eval} | "
                f"coût={entry['score']:>7.1f} | t_moy={entry.get('avg_time', 0):.3f}s{improved}"
            )

        print(f"\n{'='*65}")
        print("🏆 CLASSEMENT FINAL (top 5)")
        print(f"{'='*65}")
        for i, entry in enumerate(self.population[:5], 1):
            avg = entry.get("avg_time", 0.0)
            print(
                f"  #{i} [{entry['label']:18s}] "
                f"réfutées={entry['found']}/{n_eval} | "
                f"coût={entry['score']:.3f} | t_moy={avg:.3f}s"
            )
        if self.population:
            best = self.population[0]
            print(f"\n  ★ Meilleure fonction retenue : [{best['label']}]")
            print(f"  ★ Score total                : {best['score']:.3f}")
            print(f"  ★ Conjectures réfutées       : {best['found']}/{n_eval}")
            print(f"  ★ Temps moyen (trouvées)     : {best.get('avg_time', 0.0):.3f}s")
            print(f"\n  → Sauvegardée dans : {self.output_dir}/best_heuristic.py")
            print(f"  → Utilisable avec : python -m src.main --all --time 10 --heuristic")

    def _save_best(self):
        """Sauvegarde la meilleure fonction et les résultats."""
        if not self.population:
            return

        best = self.population[0]

        # Sauvegarder le code de la meilleure fonction
        code_path = self.output_dir / "best_heuristic.py"
        with open(code_path, "w") as f:
            f.write("# Meilleure fonction de score trouvée par FunSearch\n")
            f.write(f"# Résultats : {best['found']}/{len(self.conjectures)} conjectures | coût={best['score']:.1f}\n\n")
            f.write(best["code"])
        print(f"\n💾 Meilleure fonction sauvegardée : {code_path}")

        # Sauvegarder l'historique complet
        history_path = self.output_dir / "funsearch_history.json"
        with open(history_path, "w") as f:
            json.dump(self.population, f, indent=2, ensure_ascii=False)
        print(f"📄 Historique complet : {history_path}")
