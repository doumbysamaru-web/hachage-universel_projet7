# Projet 7 — Hachage universel et résistance aux collisions

Tableau de bord interactif (Streamlit + Plotly) pour le module *Algorithmique, structures informatiques et cryptologie* (L3 Informatique, Université de Caen Normandie).

**En ligne :** https://hachage-universel-projet7.streamlit.app

Deux tables de hachage à chaînage reçoivent **exactement les mêmes clés** ; seule la fonction change.

- **Table A** : `h(k) = k mod m`, fonction fixe et connue de tous.
- **Table B** : `h(k) = ((a·k + b) mod p) mod m`, avec `p = 2⁶¹ − 1` et `(a, b)` tirés au hasard et gardés secrets.

L'application ne contient que des réglages, des métriques et des graphiques : les explications sont données à l'oral.

## Les deux vues

| Vue | Réglages | Ce qu'on lit |
|---|---|---|
| **Attaque** | Scénario : clés aléatoires · attaque sur `k mod m` · fuite de `(a, b)` ; même échelle | Liste max, collisions, comparaisons et temps mesuré par recherche (A et B), répartition des clés par case |
| **Pr[collision] ≤ 1/m** | Clés x et y, nombre de tirages, relancer | Fréquence mesurée, 1/m, écart en σ, convergence vers 1/m |

La barre latérale règle `m` et `n`, fait un **nouveau tirage de (a, b)**, affiche le secret et active les **tirages reproductibles**.

## Chiffres de la soutenance

Tirages reproductibles activés (graine 2026), m = 64, n = 640, attaque sur `k mod m` :

- A : liste max **640**, **204 480** collisions ; B : liste max **11**, **2 892** collisions (borne 3 195) ;
- comparaisons par recherche : 320,5 contre 5,5, soit **× 58** ; collisions **× 71** ;
- Monte-Carlo (x = 7, y = 71, 20 000 tirages) : **0,0162** contre 1/64 ≈ 0,0156, écart **+0,66 σ**.

Le temps par recherche est mesuré en direct (300 recherches réelles) : il varie légèrement d'un affichage à l'autre.

Ouvrir l'application une dizaine de minutes avant de passer : une application Streamlit gratuite se met en veille et met un peu de temps à redémarrer.

## Thème clair et sombre

L'application suit le thème du système ; menu **⋮ → Settings → Theme** pour le changer. Table A en orange, table B en bleu-vert. Couleurs et polices dans `.streamlit/config.toml`.

## Organisation du code

| Fichier | Rôle |
|---|---|
| `app.py` | Le tableau de bord Streamlit (aucune logique algorithmique) |
| `moteur.py` | Le moteur en Python pur : table à chaînage instrumentée, fonctions de hachage, jeux de clés, attaques, expériences |
| `tests/test_moteur.py` | Les tests du moteur |
| `tests/test_app.py` | Le test de lancement du tableau de bord (chiffres de la soutenance, scénario de fuite) |
| `.streamlit/config.toml` | Thèmes clair et sombre, polices, barre d'outils |
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

Chaque modification de la branche `main` redéploie l'application automatiquement.
