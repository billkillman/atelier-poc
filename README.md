# Atelier d'expériences IA

Chaque commande déposée dans `commandes/<jour>/` décrit une approche d'IA publiée récemment et la manière de la comparer à une approche de référence. L'atelier :

1. fait écrire le notebook de comparaison par un modèle de langage, selon des règles de structure et de lisibilité vérifiées avant exécution ;
2. l'exécute de bout en bout dans un sandbox CPU sans secret, à échelle réduite : sur un jeu de données public, puis sur un jeu factice contre un faux service à la convention OpenAI (pour valider la phase prévue sur des données internes) ; les erreurs sont renvoyées au modèle pour correction ;
3. l'exécute à l'échelle complète sur le jeu public : sur GPU (Google Colab) pour tout entraînement PyTorch ou TensorFlow, ou quand la taille des modèles ou la durée le justifie, sur CPU sinon ;
4. en fait relire la méthode (fuite de données, équité de la comparaison, validité statistique) ;
5. publie chaque expérience validée dans son propre dépôt public, nommé `AAAAMMJJ_mots_cles`.

Index des expériences traitées : [`index.json`](index.json).

Chaque dépôt publié contient le notebook exécuté (`experiment.ipynb`), les fonctions communes (`helpers.py`), un README illustré, les résultats (`results/`), les figures (`figures/`), les dépendances exactes (`requirements.txt`) et de quoi rejouer l'expérience sur des données internes (`private.env.example`, `download_models.py`). Le notebook a deux phases : `PHASE = "public"` (jeu public) et `PHASE = "private"` (données internes et services LLM/STT internes, par variables d'environnement, avec garde-fous : aucun service public, aucune donnée brute affichée, fonctionnement hors ligne possible).

Le code de ce dépôt est synchronisé automatiquement à chaque commande ; ne pas le modifier ici.
