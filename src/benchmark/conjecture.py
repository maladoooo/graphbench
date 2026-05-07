"""
Représentation d'une conjecture du benchmark GraphBench.
Une conjecture est de la forme : Y(G) <= f(X(G)) ou Y(G) >= f(X(G))
où f est un polynôme donné par ses coefficients, son intercept et son degré.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from fractions import Fraction
from typing import List


@dataclass
class Conjecture:
    """Décrit une conjecture issue du benchmark."""
    id: int
    description: str
    subgroups: List[str]          # ex: ['connected'], ['connected', 'tree']
    x_name: str                   # nom de l'invariant X
    y_name: str                   # nom de l'invariant Y
    sign: str                     # '<=' ou '>='
    coefficients: List[float]     # coefficients du polynôme [a1, a2, ...] (degré 1..n)
    intercept: float              # terme constant
    degree: int                   # degré du polynôme
    known_counterexample: str = ""  # graph6 connu (interdit d'utiliser dans la recherche)

    def eval_f(self, x: float) -> float:
        """Évalue f(x) = intercept + a1*x + a2*x^2 + ... + an*x^n."""
        result = self.intercept
        for i, coef in enumerate(self.coefficients, start=1):
            result += coef * (x ** i)
        return result

    def violation(self, invariants: dict) -> float:
        """
        Calcule la violation de la conjecture pour un graphe donné ses invariants.
        violation > 0 ⟺ contre-exemple trouvé.
        
        Lève KeyError si X ou Y ne sont pas dans invariants.
        """
        x_val = invariants[self.x_name]
        y_val = invariants[self.y_name]
        fx = self.eval_f(x_val)

        if self.sign == "<=":
            # conjecture: Y <= f(X)  →  violation = Y - f(X)
            return y_val - fx
        else:
            # conjecture: Y >= f(X)  →  violation = f(X) - Y
            return fx - y_val

    def proof_string(self, invariants: dict) -> str:
        """Retourne une preuve textuelle que la conjecture est violée."""
        x_val = invariants[self.x_name]
        y_val = invariants[self.y_name]
        fx = self.eval_f(x_val)
        viol = self.violation(invariants)
        lines = [
            f"Conjecture #{self.id}: {self.description}",
            f"  X = {self.x_name}(G) = {x_val}",
            f"  Y = {self.y_name}(G) = {y_val}",
            f"  f(X) = {fx:.6f}",
            f"  Violation = {viol:.6f} > 0  ✓ (contre-exemple valide)",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"Conjecture(id={self.id}, {self.y_name} {self.sign} f({self.x_name}), classes={self.subgroups})"


def _parse_fraction(s: str) -> float:
    """Convertit une fraction ou un float en float. Ex: '1/6' → 0.1667."""
    s = s.strip()
    try:
        return float(Fraction(s))
    except Exception:
        return float(s)


def parse_coefficients(raw: str) -> List[float]:
    """
    Parse la colonne Coefficients du benchmark.
    Format attendu: "['1/6', '-1/3']" ou "['3/5']"
    """
    import ast
    try:
        lst = ast.literal_eval(raw)
        return [_parse_fraction(str(c)) for c in lst]
    except Exception:
        return []
