# Meilleure fonction de score trouvée par FunSearch
# Résultats : 6/15 conjectures | coût=120.3

def heuristic_score(G, invariants, conjecture):
    violation = conjecture.violation(invariants)
    # Calcul du bonus pour la régularité
    regularity_bonus = 0.5 * (5 - G.number_of_nodes()) / 5
    # Calcul du bonus pour la densité (trop dense)
    density_bonus = -0.2 * min(1, invariants.get("density", 0) - 0.6)
    # Calcul du bonus pour la densité (trop creuse)
    density_bonus += -0.5 * min(1, (1 - invariants.get("density", 0)) - 0.7)
    # Calcul du bonus pour le diamètre
    if G.number_of_nodes() > 10:
        diameter_bonus = 0.1 * G.diameter() / G.number_of_nodes()
    else:
        diameter_bonus = 0
    # Calcul de la pénalité pour un grand nombre de sommets
    node_penalty = -0.1 * G.number_of_nodes() / 100
    # Calcul de la pénalité pour un graphe trop dense
    if invariants.get("density", 0) > 0.8:
        density_penalty = -0.3
    else:
        density_penalty = 0
    # Calcul de la pénalité pour un graphe trop creux
    if invariants.get("density", 0) < 0.2:
        density_penalty += -0.2
    else:
        density_penalty += 0
    # Calcul du score total
    bonus = regularity_bonus + diameter_bonus + density_bonus + density_penalty / 2 + node_penalty / 2
    return 10.0 * violation + bonus