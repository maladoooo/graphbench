"""
Interface avec l'API Anthropic (Claude) pour générer des fonctions de score.

Rôle dans FunSearch :
- Reçoit les meilleures fonctions de score actuelles + leurs performances
- Génère une nouvelle variante améliorée
- Retourne le code Python de la nouvelle fonction
"""
from __future__ import annotations
import re
from typing import List, Optional


def build_prompt(
    best_functions: List[dict],
    conjecture_descriptions: List[str],
) -> str:
    """
    Construit le prompt envoyé au LLM.

    Args:
        best_functions: liste de {"code": str, "score": float, "found": int}
                        triée du meilleur au moins bon
        conjecture_descriptions: descriptions textuelles des conjectures testées

    Returns:
        Prompt complet à envoyer au LLM.
    """
    conj_text = "\n".join(f"  - {d}" for d in conjecture_descriptions[:5])

    functions_text = ""
    for i, f in enumerate(best_functions[:2], 1):  # max 2 fonctions pour limiter les tokens
        code_truncated = f["code"][:600]  # tronquer si trop long
        if len(f["code"]) > 600:
            code_truncated += "\n    # ... (tronqué)"
        functions_text += f"\n--- Fonction #{i} (trouvé={f['found']}, coût={f['score']:.1f}) ---\n"
        functions_text += code_truncated + "\n"

    prompt = f"""Tu es un expert en théorie des graphes et en optimisation combinatoire.

Contexte : on cherche des contre-exemples à des conjectures mathématiques sur les graphes.
Pour chaque conjecture "Y(G) <= f(X(G))", on veut trouver un graphe G tel que Y(G) > f(X(G)).

On utilise une recherche locale guidée par une fonction de score.
Plus le score est élevé, plus le graphe est considéré prometteur.
Un score > 0 signifie qu'on a trouvé un contre-exemple.

Conjectures testées :
{conj_text}

Voici les meilleures fonctions de score trouvées jusqu'ici :
{functions_text}

Ta mission : proposer une NOUVELLE fonction `heuristic_score` qui guide mieux la recherche.
Tu peux ajouter des bonus pour des structures prometteuses (grand diamètre, régularité, densité...)
et des pénalités pour des structures mauvaises (trop de sommets, graphe trop dense ou trop creux...).

CONTRAINTES STRICTES (à respecter impérativement) :
1. La fonction doit s'appeler exactement `heuristic_score`
2. Signature exacte : `def heuristic_score(G, invariants, conjecture):`
3. Elle doit retourner un float
4. Zéro import à l'intérieur de la fonction
5. Zéro exception : utiliser .get("clé", valeur_par_défaut) pour tous les accès à invariants
6. La violation DOIT être le terme dominant : `violation = conjecture.violation(invariants)` puis `return 10.0 * violation + bonus`
7. Le code doit être du Python syntaxiquement valide et complet (pas de `...`, pas de commentaires incomplets)

Retourne UNIQUEMENT le code Python de la fonction, sans explication, sans markdown, sans ```python.
"""
    return prompt


def build_crossover_prompt(
    parent_a: dict,
    parent_b: dict,
    conjecture_descriptions: List[str],
) -> str:
    """
    Construit un prompt de CROSSOVER : on demande au LLM de COMBINER deux fonctions
    successful pour en produire une troisième héritant des forces de chacune.

    C'est l'opérateur "crossover" classique d'un GA, appliqué à du code Python via LLM.
    """
    conj_text = "\n".join(f"  - {d}" for d in conjecture_descriptions[:5])
    code_a = parent_a["code"][:500]
    code_b = parent_b["code"][:500]

    prompt = f"""Tu es un expert en évolution génétique de code Python pour la théorie des graphes.

Contexte : on cherche des contre-exemples à des conjectures via recherche locale.
La fonction `heuristic_score(G, invariants, conjecture)` guide la recherche.
Score plus élevé = graphe plus prometteur.

Conjectures testées :
{conj_text}

Voici DEUX fonctions parentes qui ont bien marché :

--- PARENT A (trouvé={parent_a['found']}, coût={parent_a['score']:.1f}) ---
{code_a}

--- PARENT B (trouvé={parent_b['found']}, coût={parent_b['score']:.1f}) ---
{code_b}

Ta mission : produire un ENFANT qui COMBINE les meilleures idées des deux parents.
- Garde les termes (bonus/pénalités) qui semblent les plus efficaces
- Combine de façon astucieuse les poids
- Tu peux légèrement modifier les coefficients pour innover

CONTRAINTES STRICTES :
1. Signature exacte : `def heuristic_score(G, invariants, conjecture):`
2. Retourne un float
3. Zéro import, zéro exception (utilise .get("clé", 0))
4. La violation DOIT être le terme dominant : `return 10.0 * violation + bonus - penalty`
5. NE JAMAIS appeler de méthodes NetworkX (G.diameter(), G.radius() qui peuvent crasher) — utilise invariants.get(...)

Retourne UNIQUEMENT le code Python, sans markdown, sans explication.
"""
    return prompt


def call_llm(prompt: str, api_key: Optional[str] = None) -> Optional[str]:
    """
    Appelle l'API Groq pour générer une nouvelle fonction de score.

    Returns:
        Code Python de la fonction, ou None si échec.
    """
    try:
        from groq import Groq
        import os
        from pathlib import Path

        # Charger .env si présent
        env_path = Path(__file__).parent.parent.parent / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.startswith("GROQ_API_KEY="):
                    os.environ.setdefault("GROQ_API_KEY", line.split("=", 1)[1].strip())

        key = api_key or os.environ.get("GROQ_API_KEY")
        if not key:
            print("  [FunSearch] GROQ_API_KEY non définie")
            return None

        client = Groq(api_key=key)
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",  # modèle rapide et gratuit
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content

    except Exception as e:
        print(f"  [FunSearch] Erreur API : {e}")
        return None


def extract_function_code(llm_output: str) -> Optional[str]:
    """
    Extrait le code Python de la fonction depuis la réponse du LLM.
    Gère les cas où le LLM met du markdown (```python ... ```)
    """
    if not llm_output:
        return None

    # Supprimer les blocs markdown
    llm_output = re.sub(r"```python\s*", "", llm_output)
    llm_output = re.sub(r"```\s*", "", llm_output)

    # Extraire la fonction
    match = re.search(
        r"(def heuristic_score\s*\(.*?\).*?)(?=\ndef |\Z)",
        llm_output,
        re.DOTALL,
    )
    if match:
        return match.group(1).strip()

    # Fallback : tout le texte si ça commence par def
    stripped = llm_output.strip()
    if stripped.startswith("def heuristic_score"):
        return stripped

    return None
