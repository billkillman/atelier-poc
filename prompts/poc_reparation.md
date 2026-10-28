Tu es ingénieur ML senior. Le notebook de preuve de concept ci-dessous n'est pas livrable : $motif Corrige-le.

# Plan retenu
$plan

# Faits vérifiés par le moteur
$verifications

# Sandbox
$sandbox

# Tes cellules actuelles (JSON)
$cellules

# Ce qui s'est passé
$compte_rendu

# Consignes de correction
- Commence par la cause racine, pas le symptôme : écris-la dans `diagnostic` (2 à 4 phrases), puis rends le notebook **complet** corrigé (toutes les cellules, pas seulement celle en cause).
- Si un identifiant, un format de données ou une API ne se comporte pas comme prévu, adapte-toi aux faits (autre colonne, autre split, autre méthode de chargement) en restant fidèle au plan ; n'invente pas de contournement fragile.
- Délai dépassé ou mémoire saturée : réduis les tailles de la phase `sandbox` (jamais la phase `complete`), allège le substitut, traite par lots.
- Ne supprime jamais la comparaison, une phase, une approche ou l'enregistrement des résultats pour faire passer l'exécution. Ne masque pas une erreur par un `try/except` qui l'avale.
- Problèmes de méthode relevés par la revue : corrige-les tous (fuite de données, référence handicapée, métrique inadaptée…).

$regles
