# Projet 7 — Hachage universel et résistance aux collisions

Démonstrateur interactif (Streamlit + Plotly) pour le module *Algorithmique, structures informatiques et cryptologie* (L3 Informatique, Université de Caen Normandie).

**En ligne :** https://hachage-universel-projet7.streamlit.app

**L'idée en une phrase :** une table de hachage dont la fonction est fixée à l'avance peut toujours être piégée par des clés bien choisies (attaque HashDoS) ; si la fonction est tirée au hasard, en secret, dans une famille universelle, aucun jeu de clés n'est mauvais en moyenne.

## Ce que montre l'application

Deux tables de hachage à chaînage reçoivent **exactement les mêmes clés** ; seule la fonction change.

- **Table A** : `h(k) = k mod m`, fonction fixe et connue de tous.
- **Table B** : `h(k) = ((a·k + b) mod p) mod m`, avec `p = 2⁶¹ − 1` et `(a, b)` tirés au hasard au démarrage.

Les quatre onglets suivent l'ordre de la soutenance :

| Onglet | Ce qu'on montre | Message |
|---|---|---|
| 1 · L'attaque | Clés piégées `kᵢ = i·m + c` (ou aléatoires, ou consécutives) : nombre de clés par case, plus longue liste, collisions, rapports de coût A / B | A s'effondre en une seule liste, B reste équilibrée |
| 2 · Pr[collision] ≤ 1/m | Monte-Carlo : on tire `(a, b)` N fois pour deux clés fixées et on compare la fréquence de collision à 1/m (écart en σ, courbe de convergence) | La définition d'une famille universelle se vérifie par l'expérience |
| 3 · Coût quand n grandit | Comparaisons par recherche en fonction de n, à α constant, contre les courbes théoriques | Θ(n) pour A attaquée, O(1) pour B |
| 4 · Si (a, b) fuit | L'attaquant connaît `(a, b)` et forge ses clés ; un nouveau tirage répare la table | La garantie repose sur le secret du tirage : universel ≠ cryptographique |

La barre latérale règle `m` et `n`, fait un **nouveau tirage de (a, b)** et propose le **déroulé de la démo**. Les détails techniques (probabilité exacte, comparaison à Poisson, temps mesurés, coût de construction, données brutes) sont rangés dans les volets « Pour aller plus loin ».

## Pour la soutenance

L'option **Tirages reproductibles** est activée par défaut (graine 2026) : la démonstration redonne toujours les mêmes chiffres. Avec m = 64 et n = 640 :

- table A : 640 clés dans une case, **204 480 collisions**, 98,4 % de cases vides ;
- table B : au plus 11 clés par case, **2 892 collisions** (borne théorique 3 195) ;
- recherche **× 58** plus rapide, construction **× 71** ;
- Monte-Carlo (x = 7, y = 71, N = 20 000) : **0,0162** contre 1/64 ≈ 0,0156, écart **+0,66 σ**.

Ouvrir l'application une dizaine de minutes avant de passer : une application Streamlit gratuite se met en veille et met un peu de temps à redémarrer.

## Thème clair et sombre

L'application suit le thème du système. Pour le changer : menu **⋮ → Settings → Theme** (Light, Dark ou System). Les couleurs sont définies dans `.streamlit/config.toml` ; la table A est toujours en orange, la table B en bleu-vert. Polices : IBM Plex (Google Fonts).

## Organisation du code

| Fichier | Rôle |
|---|---|
| `app.py` | L'interface Streamlit (aucune logique algorithmique) |
| `moteur.py` | Le moteur en Python pur, typé : table à chaînage instrumentée, fonctions de hachage, jeux de clés, attaques, expériences |
| `tests/test_moteur.py` | Les tests du moteur |
| `.streamlit/config.toml` | Thèmes clair et sombre, polices |
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
2. Choisir ce dépôt, la branche `main` et le fichier principal `app.py`.
3. *Deploy*. Chaque `git push` sur `main` redéploie l'application automatiquement.

## Points à savoir expliquer

- **La garantie porte sur une moyenne sur le tirage de (a, b), pas sur chaque tirage**, et elle vaut pour des clés choisies *avant* ce tirage. Un tirage malchanceux reste possible : le remède est de re-tirer la fonction.
- **L'adversaire gagne contre toute fonction qu'il connaît**, même universelle (onglet 4). La protection vient du secret du tirage.
- **Sur des clés favorables, `k mod m` peut faire mieux** (clés consécutives) : le hachage universel ne vise pas le meilleur cas, il supprime le pire.
- **Les temps mesurés sont bruités** (Python sur un serveur partagé) ; les comparaisons comptées sont la mesure fiable.
