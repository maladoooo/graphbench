# Meilleure fonction de score trouvée par FunSearch
# Résultats : 10/100 conjectures | coût=0.0

def heuristic_score(G, invariants, conjecture):
    """Baseline enrichie : violation + bonus structurels.
    Inspirée de l'exemple TP section 7.3 (coefficients ajustés)."""
    violation = conjecture.violation(invariants)
    n = invariants.get("order", G.number_of_nodes())
    m = invariants.get("size", G.number_of_edges())
    diam = invariants.get("diameter", 0)
    Delta = invariants.get("maximum_degree", 0)
    triangles = invariants.get("triangle_number", 0)
    density = 0.0
    if n > 1:
        density = 2 * m / (n * (n - 1))
    return (
        10.0 * violation
        + 0.3 * diam
        + 0.2 * Delta
        + 0.1 * triangles
        - 0.005 * n
        - 0.1 * abs(density - 0.5)
    )
