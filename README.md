# Atelier de preuves de concept IA

Chaque commande déposée dans `commandes/<jour>/` décrit une nouvelle approche d'IA (publiée récemment) et la manière de la comparer à une approche de référence. L'atelier :

1. fait écrire le notebook de comparaison par un modèle de langage ;
2. l'exécute de bout en bout dans un sandbox CPU sans secret, à échelle réduite : sur un jeu de données public, puis sur un jeu bancaire factice contre un faux service à la convention OpenAI (pour valider la phase prévue dans un environnement bancaire) ; les erreurs sont renvoyées au modèle pour correction ;
3. l'exécute à l'échelle complète sur le jeu public : sur GPU (Google Colab) pour tout entraînement PyTorch ou TensorFlow, ou quand la taille des modèles ou la durée le justifie, sur CPU sinon ;
4. en fait relire la méthode (fuite de données, équité de la comparaison, validité statistique) ;
5. publie chaque POC validé dans son propre dépôt public, nommé `AAAAMMJJ_mots_cles`.

Index des POC traités : [`index.json`](index.json).

Chaque dépôt de POC contient le notebook exécuté (`poc.ipynb`), son mode d'emploi (`README.md`), les résultats (`resultats.json`) et les dépendances exactes (`requirements.txt`). Le notebook a deux phases : `PHASE = "publique"` (jeu public) et `PHASE = "banque"` (jeu bancaire et services LLM/STT internes, par variables d'environnement, avec garde-fous : aucun service public, aucune donnée brute affichée, fonctionnement hors ligne possible).

Notebooks écrits par un modèle de langage : à relire avant toute décision.

Le code de ce dépôt est synchronisé automatiquement à chaque commande ; ne pas le modifier ici.
