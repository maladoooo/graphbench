"""
Chargement du benchmark depuis benchmark.xlsx.
Chaque ligne = une conjecture.
"""
from __future__ import annotations
import ast
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

from .conjecture import Conjecture, parse_coefficients, _parse_fraction
from ..invariants.compute import SUPPORTED_INVARIANTS
from ..graphs.classes import SUPPORTED_CLASSES

BENCHMARK_PATH = Path(__file__).parent.parent.parent / "benchmark" / "benchmark.xlsx"


def load_benchmark(
    path: Path = BENCHMARK_PATH,
    subgroup_filter: Optional[List[str]] = None,
    ids: Optional[List[int]] = None,
) -> List[Conjecture]:
    """
    Charge toutes les conjectures du benchmark.

    Args:
        path: Chemin vers benchmark.xlsx
        subgroup_filter: Si fourni, ne charge que les conjectures dont
                         le subgroup contient TOUS les éléments de la liste.
                         Ex: ['connected'] → uniquement les connexes.
        ids: Si fourni, charge uniquement les conjectures avec ces IDs.

    Returns:
        Liste de Conjecture triée par ID.
    """
    df = pd.read_excel(path, dtype=str)
    df.columns = [c.strip() for c in df.columns]

    conjectures = []
    for _, row in df.iterrows():
        try:
            cid = int(row["Conjecture ID"])
        except (ValueError, KeyError):
            continue

        if ids is not None and cid not in ids:
            continue

        # Parse subgroups (format: "['connected', 'tree']")
        try:
            subgroups = ast.literal_eval(row["Subgroup"])
        except Exception:
            subgroups = [str(row["Subgroup"]).strip()]

        if subgroup_filter is not None:
            if not all(sg in subgroups for sg in subgroup_filter):
                continue

        coefficients = parse_coefficients(str(row.get("Coefficients", "[]")))
        intercept_raw = str(row.get("Intercept", "0")).strip()
        intercept = _parse_fraction(intercept_raw) if intercept_raw else 0.0

        try:
            degree = int(row.get("Degree", 1))
        except (ValueError, TypeError):
            degree = len(coefficients)

        known_ce = str(row.get("Counter example (g6)", "")).strip()
        if known_ce.lower() in ("nan", "none", ""):
            known_ce = ""

        c = Conjecture(
            id=cid,
            description=str(row.get("Conjecture", "")),
            subgroups=subgroups,
            x_name=str(row.get("X", "")).strip(),
            y_name=str(row.get("Y", "")).strip(),
            sign=str(row.get("Sign", "<=")).strip(),
            coefficients=coefficients,
            intercept=intercept,
            degree=degree,
            known_counterexample=known_ce,
        )
        conjectures.append(c)

    conjectures.sort(key=lambda c: c.id)
    return conjectures


def validate_benchmark(conjectures: List[Conjecture]) -> Tuple[List[Conjecture], List[str]]:
    """
    Valide une liste de conjectures et filtre celles qu'on ne peut pas traiter.

    Vérifie pour chaque conjecture :
    - Les invariants X et Y sont implémentés
    - Toutes les classes de graphes sont supportées

    Retourne :
        valid   : conjectures traitables
        warnings: messages d'avertissement pour les conjectures ignorées
    """
    valid = []
    warnings = []

    for c in conjectures:
        issues = []

        for inv in (c.x_name, c.y_name):
            if inv not in SUPPORTED_INVARIANTS:
                issues.append(f"invariant inconnu '{inv}'")

        for cls in c.subgroups:
            if cls not in SUPPORTED_CLASSES:
                issues.append(f"classe inconnue '{cls}'")

        if issues:
            msg = f"  ⚠️  Conjecture #{c.id} ignorée — {', '.join(issues)}"
            warnings.append(msg)
        else:
            valid.append(c)

    return valid, warnings


def load_conjecture_by_id(cid: int, path: Path = BENCHMARK_PATH) -> Conjecture:
    """Charge une seule conjecture par son ID. Lève ValueError si introuvable."""
    results = load_benchmark(path=path, ids=[cid])
    if not results:
        raise ValueError(f"Conjecture #{cid} introuvable dans {path}")
    return results[0]
