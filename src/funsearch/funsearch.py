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
from .evaluate import SEED_FUNCTION, compile_heuristic, evaluate_heuristic
from .llm_client import build_prompt, call_llm, extract_function_code


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

        # --- Étape 2 : évaluer la fonction de base (seed) ---
        print("\n📌 Itération 0 : fonction de base (violation seule)")
        seed_result = self._evaluate_and_register(SEED_FUNCTION, label="seed")
        if seed_result is None:
            print("  ❌ Erreur sur la fonction de base. Abandon.")
            return

        # Si déjà 100% résolues → inutile d'appeler le LLM
        if seed_result["found"] == len(hard_conjectures):
            print("  ✅ 100% résolues avec la fonction de base. FunSearch inutile.")
            self._save_best()
            return

        # --- Étape 3 : boucle évolutive avec early stopping ---
        no_improve_count = 0
        MAX_NO_IMPROVE = 3   # arrêt si 3 itérations consécutives sans amélioration
        seen_codes = {SEED_FUNCTION}  # évite de réévaluer la même fonction

        for iteration in range(1, self.n_iterations + 1):
            print(f"\n🔄 Itération {iteration}/{self.n_iterations}")

            # Early stopping si stagnation
            if no_improve_count >= MAX_NO_IMPROVE:
                print(f"  ⏹  Arrêt anticipé : {no_improve_count} itérations sans amélioration.")
                break

            # 1. Demander au LLM une nouvelle fonction (avec retry sur erreur)
            new_code = None
            for _attempt in range(2):  # max 2 tentatives
                candidate = self._ask_llm()
                if candidate is None:
                    break
                if candidate in seen_codes:
                    print("  ⚠️  Fonction identique à une déjà testée, retry...")
                    continue
                # Vérifier la syntaxe avant d'évaluer
                from .evaluate import compile_heuristic
                if compile_heuristic(candidate) is not None:
                    new_code = candidate
                    break
                print(f"  ⚠️  Syntaxe invalide, retry ({_attempt+1}/2)...")

            if new_code is None:
                print("  ⚠️  LLM n'a pas pu générer de fonction valide.")
                no_improve_count += 1
                continue

            seen_codes.add(new_code)

            print(f"  💡 Nouvelle fonction générée ({len(new_code)} chars)")

            # 2. Évaluer
            best_before = self.population[0]["found"] if self.population else 0
            result = self._evaluate_and_register(new_code, label=f"iter{iteration}")
            if result is None:
                print("  ⚠️  Fonction invalide.")
                no_improve_count += 1
                continue

            # 3. Détecter l'amélioration
            if result["found"] > best_before:
                print(f"  🎉 Amélioration ! {best_before} → {result['found']} conjectures résolues")
                no_improve_count = 0
            else:
                no_improve_count += 1
                print(f"  ~ Pas d'amélioration ({no_improve_count}/{MAX_NO_IMPROVE})")

            # 4. Si 100% résolues, inutile de continuer
            if result["found"] == len(hard_conjectures):
                print("  ✅ 100% des conjectures difficiles résolues !")
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

    def _ask_llm(self) -> Optional[str]:
        """Demande au LLM de générer une nouvelle fonction de score."""
        # Construire le contexte : les 3 meilleures fonctions actuelles
        conj_descriptions = [
            f"#{c.id}: {c.y_name} {c.sign} f({c.x_name}) sur {c.subgroups}"
            for c in self.conjectures[:8]
        ]

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

    def _print_summary(self):
        """Affiche le classement final des fonctions."""
        print(f"\n{'='*60}")
        print("🏆 CLASSEMENT FINAL DES FONCTIONS")
        print(f"{'='*60}")
        n_eval = len(self._eval_conjectures)
        for i, entry in enumerate(self.population[:5], 1):
            print(f"  #{i} [{entry['label']}] : {entry['found']}/{n_eval} trouvées | coût={entry['score']:.1f}")

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
