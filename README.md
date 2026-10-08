# Projet 7 — Hachage universel et résistance aux collisions

Démonstrateur interactif (Streamlit + Plotly) pour le module *Algorithmique, structures informatiques et cryptologie* (L3).

**L'idée en une phrase :** une table de hachage dont la fonction est fixée à l'avance peut toujours être piégée par des clés bien choisies (attaque HashDoS) ; si la fonction est tirée au hasard dans une famille 2-universelle, aucun jeu de clés n'est mauvais en moyenne.

## Interface

Barre latérale : taille `m`, nombre de clés `n`, profil des clés (uniforme, séquentiel, attaque DoS `kᵢ = i·m + c`, attaque adaptative avec `(a, b)` divulgués) et bouton **Rééchantillonner (a, b)**.

| Onglet | Contenu |
|---|---|
| ⚔️ Stress test | `k mod m` et Carter-Wegman côte à côte sur les mêmes clés : α, `L_max`, cases vides, collisions, profil de charge des alvéoles, rapports de coût A/B, `L_max` au fil des tirages |
| 🎯 Bornes Monte-Carlo | Fréquence de `h(x) = h(y)` sur `N` tirages de `(a, b)` contre `1/m` (jauge, convergence, statut PASS/FAIL à 3σ) ; histogramme des longueurs de chaînes contre Poisson(α) |
| 📈 Complexité | Comparaisons et temps par recherche en fonction de `n` (pire cas Θ(n) contre O(1)), pentes mesurées, coût de construction Θ(n²) sous attaque |

![Stress test](docs/onglet1.png)

## Organisation du code

| Fichier | Rôle |
|---|---|
| `app.py` | L'interface Streamlit (aucune logique algorithmique) |
| `moteur.py` | Le moteur en Python pur, typé : table à chaînage instrumentée, fonctions de hachage, jeux de données, attaques, expériences |
| `tests/test_moteur.py` | Les tests du moteur |
| `requirements.txt` | Les dépendances pour Streamlit Cloud |

## Lancer en local

Python 3.10 ou plus récent.

```
python -m venv .venv
source .venv/bin/activate          # Windows : .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py               # ouvre http://localhost:8501
python -m unittest                 # les tests
```

## Publier sur Streamlit Community Cloud

1. Sur [share.streamlit.io](https://share.streamlit.io), se connecter avec GitHub puis *Create app → Deploy a public app from GitHub*.
2. Choisir ce dépôt, la branche `main` et le fichier principal `app.py` ; dans *Advanced settings*, choisir Python 3.12.
3. *Deploy*. Chaque `git push` sur `main` redéploie l'application automatiquement.

Conseil pour la soutenance : activer « Tirages reproductibles » dans la barre latérale pour rejouer exactement la même démonstration.

## Points à savoir expliquer

- **La garantie porte sur une espérance, pas sur chaque tirage.** Sur des clés en progression arithmétique, la loi du coût selon `(a, b)` est à queue lourde : la plupart des tirages font mieux que `1 + α/2`, de rares tirages bien pire. Le remède : re-tirer la fonction.
- **L'adversaire gagne contre toute fonction qu'il connaît**, même universelle (bouton « Fuite de (a, b) »). La protection vient du secret du tirage.
- **Sur des clés favorables, `k mod m` peut faire mieux** (clés consécutives) : le hachage universel ne vise pas le meilleur cas, il supprime le pire.
