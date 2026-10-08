"""Projet 7 — Hachage universel et résistance aux collisions.

Démonstrateur interactif Streamlit (L3 « Algorithmique, structures
informatiques et cryptologie »).

Lancer en local :
    pip install -r requirements.txt
    streamlit run app.py

Organisation :
    moteur.py   moteur algorithmique en Python pur (table, fonctions de
                hachage, jeux de données, expériences) — testé à part ;
    app.py      ce fichier : l'interface uniquement (4 onglets).
"""

from __future__ import annotations

import math
import random
import time
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import stats as sps

import moteur as M

# ==========================================================================
# Configuration générale & palette
# ==========================================================================

st.set_page_config(
    page_title="Hachage universel — Projet 7",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Couleurs d'état des alvéoles (toujours accompagnées d'un libellé en légende).
GOOD = "#0ca30c"       # charge idéale
SERIOUS = "#ec835a"    # charge modérée
CRITICAL = "#d03b3b"   # chaîne critique
HIGHLIGHT = "#4a3aa7"  # alvéole visée par la dernière recherche

# Couleurs catégorielles (ordre fixe, une couleur = une entité, jamais un rang).
SERIES = {
    "k mod m · clés uniforme": "#2a78d6",
    "k mod m · clés adverse": "#eb6834",
    "Carter-Wegman · clés uniforme": "#1baf7a",
    "Carter-Wegman · clés adverse": "#eda100",
}
DET_COLOR = "#eb6834"   # fonction déterministe
UNI_COLOR = "#2a78d6"   # fonction universelle
THEORY_COLOR = "#7c7b77"

MAX_KEY = M.P_MERSENNE_61 - 1
#: Les widgets numériques de Streamlit sont limités aux entiers « sûrs » de JavaScript.
MAX_INPUT_KEY = 2**53 - 1


# ==========================================================================
# Petits utilitaires d'affichage
# ==========================================================================

def fr(x: float | int, decimals: int = 2) -> str:
    """Format français : espace fine pour les milliers, virgule décimale."""
    if isinstance(x, (int, np.integer)):
        return f"{int(x):,}".replace(",", "\u202f")
    return f"{x:,.{decimals}f}".replace(",", "\u202f").replace(".", ",")


def get_rng() -> random.Random:
    """Générateur partagé par la session (reproductible si une graine est fixée)."""
    seed: Optional[int] = st.session_state.get("seed_value")
    signature = ("seeded", seed) if seed is not None else ("system", None)
    if st.session_state.get("rng_signature") != signature:
        st.session_state.rng = random.Random(seed) if seed is not None else random.SystemRandom()
        st.session_state.rng_signature = signature
    return st.session_state.rng


def bucket_status(length: int, alpha: float) -> str:
    """Classe une alvéole : idéale (≤ α), modérée, critique (> 3α).

    Comme une chaîne contient un nombre entier de clés, le seuil « idéal »
    est ⌈α⌉ (au moins 1) : avec α = 0,5 une alvéole à 1 clé reste idéale.
    """
    ideal = max(1, math.ceil(alpha - 1e-12))
    if length <= ideal:
        return "idéale"
    if length <= 3 * max(alpha, 1.0):
        return "modérée"
    return "critique"


STATUS_STYLE = {
    "idéale": (GOOD, "Charge idéale (≤ ⌈α⌉)"),
    "modérée": (SERIOUS, "Charge modérée"),
    "critique": (CRITICAL, "Chaîne critique (> 3α)"),
}


def bucket_figure(table: M.HashTableChaining, title: str,
                  highlight: Optional[int] = None,
                  y_max: Optional[float] = None,
                  height: int = 380) -> go.Figure:
    """Diagramme en colonnes : une colonne par alvéole, hauteur = longueur."""
    lengths = table.chain_lengths()
    alpha = table.n / table.m
    fig = go.Figure()
    for status, (color, label) in STATUS_STYLE.items():
        idx = [i for i, L in enumerate(lengths)
               if bucket_status(L, alpha) == status and i != highlight]
        if not idx:
            continue
        hover = []
        for i in idx:
            chain = list(table.buckets[i])
            preview = ", ".join(map(str, chain[:6])) + (" …" if len(chain) > 6 else "")
            hover.append(preview or "∅")
        fig.add_trace(go.Bar(
            x=idx, y=[lengths[i] for i in idx], name=label,
            marker=dict(color=color, line=dict(width=0)),
            customdata=hover,
            hovertemplate=("Alvéole %{x}<br>Longueur %{y}<br>"
                           "Clés : %{customdata}<extra></extra>"),
        ))
    if highlight is not None and 0 <= highlight < table.m:
        chain = list(table.buckets[highlight])
        fig.add_trace(go.Bar(
            x=[highlight], y=[max(lengths[highlight], 0.15)],
            name="Alvéole de la dernière recherche",
            marker=dict(color=HIGHLIGHT, line=dict(width=0)),
            customdata=[", ".join(map(str, chain[:6])) or "∅"],
            hovertemplate="Alvéole %{x} (recherchée)<br>Longueur %{y}<br>"
                          "Clés : %{customdata}<extra></extra>",
        ))
    if table.n:
        fig.add_hline(y=alpha, line=dict(color=THEORY_COLOR, dash="dot", width=1.5),
                      annotation_text=f"α = {fr(alpha)}", annotation_position="top left")
    fig.update_layout(
        title=dict(text=title, font=dict(size=15)),
        barmode="overlay", bargap=0.15 if table.m <= 128 else 0.0,
        height=height, margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=-0.32, x=0),
        xaxis=dict(title="Alvéole (bucket)", range=[-0.6, table.m - 0.4]),
        yaxis=dict(title="Nombre de clés", rangemode="tozero",
                   range=[0, y_max * 1.05] if y_max else None),
        hovermode="closest",
    )
    return fig


def show(fig: go.Figure, key: Optional[str] = None) -> None:
    """Affiche une figure Plotly (séparateurs décimaux à la française)."""
    fig.update_layout(separators=",\u202f")
    st.plotly_chart(fig, config={"displaylogo": False}, key=key)


def chain_markdown(table: M.HashTableChaining, key: int, limit: int = 25) -> str:
    """Représente la chaîne parcourue : [k1] → [k2] → … → ∅."""
    bucket = table.hash_fn(key)
    chain = list(table.buckets[bucket])
    parts: list[str] = []
    for other in chain[:limit]:
        parts.append(f"**`[{other}]`**" if other == key else f"`[{other}]`")
    if len(chain) > limit:
        parts.append(f"… (+{len(chain) - limit})")
    parts.append("∅")
    return f"T[{bucket}] → " + " → ".join(parts)


def explain_latex(h: M.HashFunction, key: int) -> str:
    """Calcul de h(k), en LaTeX, en masquant les grands a, b, p."""
    return h.explain(key)


def show_hash_parameters(h: M.HashFunction) -> None:
    """Affiche la formule de la fonction et ses paramètres tirés."""
    st.latex(h.formula())
    if isinstance(h, M.CarterWegmanHash):
        st.caption(f"Paramètres tirés : a = {h.a} · b = {h.b} · p = {h.p}"
                   + (" (= 2⁶¹ − 1, premier de Mersenne)" if h.p == M.P_MERSENNE_61 else ""))


# ==========================================================================
# Barre latérale
# ==========================================================================

with st.sidebar:
    st.title("🔐 Projet 7")
    st.markdown("**Hachage universel et résistance aux collisions**")
    st.divider()
    reproducible = st.toggle("Tirages reproductibles (graine fixée)", value=False,
                             help="Pratique pour répéter exactement une démonstration.")
    if reproducible:
        st.session_state.seed_value = int(st.number_input("Graine", 0, 10**9, 2026))
    else:
        st.session_state.seed_value = None
    st.divider()
    st.markdown(
        "**Rappel**  \n"
        "Famille 2-universelle : pour tout $x \\neq y$,  \n"
        "$\\Pr_{h \\in \\mathcal{H}}[h(x)=h(y)] \\le \\dfrac{1}{m}$.  \n\n"
        "Conséquence : coût moyen d'une recherche $\\le 1 + \\alpha/2$ "
        "**pour tout jeu de clés**, l'espérance portant sur le tirage de $h$."
    )
    st.divider()
    st.caption("L3 · Algorithmique, structures informatiques et cryptologie")

st.title("Hachage universel et résistance aux collisions")
st.caption("Table de hachage à chaînage · fonctions déterministes vs famille "
           "2-universelle de Carter-Wegman · attaque HashDoS")

tab1, tab2, tab3, tab4 = st.tabs([
    "① Visualiseur de la table",
    "② Laboratoire : attaque DoS",
    "③ Théorèmes & benchmarks",
    "④ Fiche soutenance",
])


# ==========================================================================
# Onglet 1 — Visualiseur interactif
# ==========================================================================

def t1_rebuild(m: int, fn_name: str, keys: Sequence[int]) -> None:
    """(Re)construit la table de l'onglet 1."""
    h = M.make_hash(fn_name, m, get_rng())
    st.session_state.t1_table = M.build_table(keys, h)
    st.session_state.t1_config = (m, fn_name)
    st.session_state.t1_highlight = None
    st.session_state.t1_last = None


with tab1:
    st.subheader("Visualiseur interactif de la table")
    c1, c2, c3, c4 = st.columns([1, 1.4, 1.4, 1])
    t1_m = int(c1.number_input("Taille m (alvéoles)", 1, 2000, 31, key="t1_m"))
    t1_fn = c2.selectbox("Fonction de hachage", list(M.HASH_FAMILIES), index=2, key="t1_fn")
    t1_kind = c3.selectbox("Jeu de données", M.DATASETS, key="t1_kind")
    t1_n = int(c4.number_input("Nombre de clés n", 0, 3000, 60, key="t1_n"))

    if "t1_table" not in st.session_state:
        t1_rebuild(t1_m, t1_fn, M.generate_dataset(t1_kind, t1_n, t1_m, rng=get_rng()))
    elif st.session_state.t1_config != (t1_m, t1_fn):
        # Changer m ou la fonction : on garde les clés et on réinsère tout.
        t1_rebuild(t1_m, t1_fn, st.session_state.t1_table.keys())

    table: M.HashTableChaining = st.session_state.t1_table

    b1, b2, b3, b4 = st.columns(4)
    if b1.button("🎲 Générer le jeu de données", width="stretch"):
        t1_rebuild(t1_m, t1_fn, M.generate_dataset(t1_kind, t1_n, t1_m, rng=get_rng()))
        table = st.session_state.t1_table
    if b2.button("🧹 Vider la table", width="stretch"):
        table.clear()
        st.session_state.t1_highlight = None
        st.session_state.t1_last = None
    if b3.button("🔁 Re-tirer la fonction (a, b)", width="stretch",
                 disabled=not table.hash_fn.randomized,
                 help="Nouvelle fonction de la famille, mêmes clés, même m."):
        table.rehash(rng=get_rng())
        st.session_state.t1_highlight = None
    b4.download_button("⬇️ Exporter les clés (CSV)", width="stretch",
                       data=pd.DataFrame({"cle": table.keys()}).to_csv(index=False),
                       file_name="cles.csv", mime="text/csv")

    # --- Opérations manuelles -------------------------------------------
    with st.container(border=True):
        o1, o2, o3, o4 = st.columns([2, 1, 1, 1], vertical_alignment="bottom")
        key_txt = o1.text_input("Clé (entier naturel, ou plusieurs séparées par des virgules)",
                                value="42", key="t1_key")
        try:
            manual_keys = [int(s) for s in key_txt.replace(";", ",").split(",") if s.strip()]
            bad = [k for k in manual_keys if not 0 <= k <= MAX_KEY]
            if bad:
                raise ValueError
        except ValueError:
            manual_keys = []
            o1.error(f"Entiers entre 0 et p − 1 = {MAX_KEY} attendus.")
        if o2.button("➕ Insérer", width="stretch", disabled=not manual_keys):
            recs = [table.insert(k) for k in manual_keys]
            st.session_state.t1_highlight = recs[-1].bucket
            st.session_state.t1_last = recs[-1]
        if o3.button("🔍 Rechercher", width="stretch", disabled=not manual_keys):
            rec = table.search(manual_keys[0])
            st.session_state.t1_highlight = rec.bucket
            st.session_state.t1_last = rec
        if o4.button("🗑️ Supprimer", width="stretch", disabled=not manual_keys):
            rec = table.delete(manual_keys[0])
            st.session_state.t1_highlight = rec.bucket
            st.session_state.t1_last = rec

        last: Optional[M.OperationRecord] = st.session_state.get("t1_last")
        if last is not None:
            r1, r2 = st.columns([1.2, 1])
            with r1:
                st.markdown(f"**Calcul de l'alvéole** pour la clé `{last.key}` :")
                st.latex(explain_latex(table.hash_fn, last.key))
                st.markdown("**Chaîne parcourue :** " + chain_markdown(table, last.key))
            with r2:
                verdict = {
                    "insertion": "insérée" if last.success else "déjà présente (doublon refusé)",
                    "recherche": "trouvée ✅" if last.success else "absente ❌",
                    "suppression": "supprimée" if last.success else "absente",
                }[last.operation]
                st.metric(f"{last.operation.capitalize()} — clé {verdict}",
                          f"{last.comparisons} comparaison(s)",
                          help="Nombre de clés de la chaîne comparées à la clé cherchée "
                               "(= accès mémoire aux maillons).")
                st.caption(f"Alvéole {last.bucket} · longueur de chaîne {last.chain_length}"
                           + (" · collision à l'insertion" if last.collision else ""))

    # --- Métriques ---------------------------------------------------------
    s = table.stats()
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Charge α = n/m", fr(s.load_factor), help=f"n = {s.n}, m = {s.m}")
    expected_lmax = (math.log(s.n) / math.log(math.log(s.n))) if s.n > 15 else None
    k2.metric("L_max", s.max_chain,
              help="Pour une fonction « idéale » et α ≈ 1, L_max = Θ(ln n / ln ln n)"
                   + (f" ≈ {fr(expected_lmax, 1)} ici." if expected_lmax else "."))
    k3.metric("Moy. non vides", fr(s.mean_nonempty_chain))
    k4.metric("Alvéoles vides", f"{fr(100 * s.empty_ratio, 1)} %",
              delta=f"e^(−α) = {fr(100 * math.exp(-s.load_factor), 1)} %",
              delta_color="off",
              help="Si les n clés tombent uniformément : (1 − 1/m)^n ≈ e^(−α).")
    k5.metric("Paires en collision", fr(s.colliding_pairs),
              delta=f"E ≤ {fr(s.n * (s.n - 1) / (2 * s.m), 1)}",
              delta_color="off", help="Σ C(Lᵢ, 2) : nombre de paires {x, y} avec h(x) = h(y).")
    k6.metric("Coût recherche", fr(s.expected_success_cost),
              delta=f"1 + α/2 = {fr(1 + s.load_factor / 2)}", delta_color="off",
              help="Exact : 1 + (paires en collision)/n.")

    show(bucket_figure(table, f"Répartition des {s.n} clés — {table.hash_fn.name}",
                       highlight=st.session_state.get("t1_highlight")))
    with st.expander("Formule et paramètres de la fonction courante"):
        show_hash_parameters(table.hash_fn)

    g1, g2 = st.columns(2)
    with g1:
        lengths = np.array(table.chain_lengths())
        counts = np.bincount(lengths) if lengths.size else np.array([0])
        ks = np.arange(len(counts))
        poisson = sps.poisson.pmf(ks, s.load_factor) * s.m if s.n else np.zeros_like(ks, float)
        hist = go.Figure()
        hist.add_bar(x=ks, y=counts, name="Observé", marker_color=UNI_COLOR,
                     hovertemplate="%{y} alvéole(s) de longueur %{x}<extra></extra>")
        hist.add_scatter(x=ks, y=poisson, name="Poisson(α) · m (hachage idéal)",
                         mode="lines+markers", line=dict(color=THEORY_COLOR, width=2, dash="dot"),
                         marker=dict(size=8),
                         hovertemplate="Attendu : %{y:.1f}<extra></extra>")
        hist.update_layout(title="Distribution des longueurs de chaînes", height=320,
                           margin=dict(l=10, r=10, t=45, b=10), bargap=0.15,
                           xaxis_title="Longueur L", yaxis_title="Nombre d'alvéoles",
                           legend=dict(orientation="h", y=-0.3))
        show(hist)
    with g2:
        st.markdown("**Journal des opérations** (manuelles)")
        log = pd.DataFrame([vars(r) for r in reversed(table.log)])
        if log.empty:
            st.info("Insérez, cherchez ou supprimez une clé pour remplir le journal.")
        else:
            st.dataframe(log.rename(columns={
                "operation": "Opération", "key": "Clé", "bucket": "Alvéole",
                "comparisons": "Comparaisons", "success": "Succès",
                "chain_length": "Long. chaîne", "collision": "Collision"}),
                hide_index=True, height=280)
        st.caption(f"Compteurs cumulés : {fr(table.insertions)} insertions · "
                   f"{fr(table.insert_collisions)} collisions à l'insertion · "
                   f"{fr(table.total_comparisons)} comparaisons au total.")


# ==========================================================================
# Onglet 2 — Laboratoire comparatif & attaque DoS
# ==========================================================================

def t2_build() -> None:
    """Reconstruit les deux tables (même jeu de clés) en mesurant le temps."""
    keys = st.session_state.t2_keys
    for side in ("det", "uni"):
        h = st.session_state[f"t2_{side}_fn"]
        start = time.perf_counter()
        tbl = M.build_table(keys, h)
        st.session_state[f"t2_{side}_table"] = tbl
        st.session_state[f"t2_{side}_ms"] = 1000 * (time.perf_counter() - start)


def t2_reset(m: int, det_name: str) -> None:
    """Nouvelles fonctions (déterministe + universelle tirée) pour la taille m."""
    st.session_state.t2_det_fn = M.make_hash(det_name, m)
    st.session_state.t2_uni_fn = M.CarterWegmanHash(m, rng=get_rng())
    st.session_state.t2_config = (m, det_name)
    st.session_state.t2_history = []


def t2_log_draw(event: str) -> None:
    """Ajoute une ligne à l'historique des tirages de (a, b)."""
    h: M.CarterWegmanHash = st.session_state.t2_uni_fn
    s = st.session_state.t2_uni_table.stats()
    st.session_state.t2_history.append({
        "Événement": event, "a": str(h.a), "b": str(h.b),
        "L_max (universelle)": s.max_chain,
        "Coût moyen": round(s.expected_success_cost, 3),
    })


with tab2:
    st.subheader("Laboratoire comparatif : déterministe vs universel sous attaque")
    st.markdown(
        "Les **deux tables reçoivent exactement les mêmes clés**. L'attaquant connaît "
        "le code source (donc la fonction déterministe) mais **pas** le tirage secret "
        "$(a, b)$ de la fonction universelle — sauf dans le scénario « fuite »."
    )
    p1, p2, p3, p4 = st.columns(4)
    t2_m = int(p1.number_input("Taille m", 2, 2000, 64, key="t2_m"))
    t2_n = int(p2.slider("Nombre de clés n", 10, 3000, 800, step=10, key="t2_n"))
    t2_c = int(p3.number_input("Alvéole visée c", 0, 1999, 0, key="t2_c")) % t2_m
    t2_det = p4.selectbox("Fonction de la colonne A",
                          [M.ModuloHash.name, M.KnuthHash.name], key="t2_det")

    if st.session_state.get("t2_config") != (t2_m, t2_det):
        t2_reset(t2_m, t2_det)
        st.session_state.t2_keys = M.keys_uniform(t2_n, rng=get_rng())
        st.session_state.t2_scenario = "Clés uniformes (trafic normal)"
        t2_build()
        t2_log_draw("Tirage initial")

    a1, a2, a3, a4 = st.columns(4)
    if a1.button("💣 Injecter l'attaque DoS", type="primary", width="stretch",
                 help="Clés forgées contre la fonction déterministe (connue de tous)."):
        st.session_state.t2_keys = M.keys_adversarial_for(st.session_state.t2_det_fn, t2_n, t2_c)
        st.session_state.t2_scenario = (
            f"Attaque HashDoS : {t2_n} clés forgées contre « {t2_det} » "
            + ("(kᵢ = i·m + c)" if t2_det == M.ModuloHash.name else "(inversion de A modulo 2³²)"))
        t2_build()
        t2_log_draw("Attaque contre la fonction déterministe")
    if a2.button("🕵️ Fuite de (a, b) : l'attaquant s'adapte", width="stretch",
                 help="Si (a, b) fuit, l'attaquant résout a·k + b ≡ c + j·m (mod p)."):
        st.session_state.t2_keys = M.keys_adversarial_known_universal(
            t2_n, st.session_state.t2_uni_fn, t2_c)
        st.session_state.t2_scenario = ("Attaquant adaptatif : il connaît (a, b) et vise "
                                        f"l'alvéole {t2_c} de la fonction universelle")
        t2_build()
        t2_log_draw("Clés forgées contre (a, b) divulgués")
    if a3.button("🎲 Re-tirer la fonction aléatoire (a, b)", width="stretch",
                 help="Nouveau couple (a, b) secret ; m inchangé ; mêmes clés."):
        st.session_state.t2_uni_fn.redraw(get_rng())
        t2_build()
        t2_log_draw("Re-tirage de (a, b)")
    if a4.button("🌐 Trafic normal (clés uniformes)", width="stretch"):
        st.session_state.t2_keys = M.keys_uniform(t2_n, rng=get_rng())
        st.session_state.t2_scenario = "Clés uniformes (trafic normal)"
        t2_build()
        t2_log_draw("Clés uniformes")

    if len(st.session_state.t2_keys) != t2_n:
        st.info(f"Le jeu courant contient {len(st.session_state.t2_keys)} clés : "
                "cliquez sur un scénario pour appliquer la nouvelle valeur de n.")
    st.markdown(f"**Scénario courant :** {st.session_state.t2_scenario}")

    det_tbl: M.HashTableChaining = st.session_state.t2_det_table
    uni_tbl: M.HashTableChaining = st.session_state.t2_uni_table
    sd, su = det_tbl.stats(), uni_tbl.stats()
    same_scale = st.toggle("Même échelle verticale pour les deux tables", value=True,
                           help="Rend visible l'écart d'ordre de grandeur.")
    y_max = max(sd.max_chain, su.max_chain, 1) if same_scale else None

    colA, colB = st.columns(2)
    for col, tbl, s_, ms, label, color in (
        (colA, det_tbl, sd, st.session_state.t2_det_ms, "A — Déterministe", DET_COLOR),
        (colB, uni_tbl, su, st.session_state.t2_uni_ms, "B — Universelle", UNI_COLOR),
    ):
        with col:
            with st.container(border=True):
                st.markdown(f"#### <span style='color:{color}'>■</span> {label}",
                            unsafe_allow_html=True)
                show_hash_parameters(tbl.hash_fn)
                m1, m2, m3 = st.columns(3)
                m1.metric("L_max", fr(s_.max_chain),
                          help="Plus longue chaîne = pire recherche.")
                m2.metric("Coût recherche", fr(s_.expected_success_cost),
                          help="Comparaisons moyennes, recherche fructueuse : 1 + paires/n.")
                m3.metric("Alvéoles occupées", f"{fr(s_.m - round(s_.empty_ratio * s_.m))}/{s_.m}")
                m4, m5 = st.columns(2)
                m4.metric("Comparaisons (construction)", fr(tbl.insert_comparisons),
                          help="Chaque insertion vérifie l'absence de doublon : "
                               "n(n−1)/2 si tout tombe dans la même alvéole.")
                m5.metric("Temps (construction)", f"{fr(ms, 1)} ms")
                regime = ("🟥 Effondrement : la table est devenue une liste — Θ(n)"
                          if s_.max_chain > max(3 * s_.load_factor, 8) and s_.max_chain > s_.n / 4
                          else "🟩 Dispersion saine — O(1) en moyenne")
                st.markdown(f"**{regime}**")
                show(bucket_figure(tbl, f"{tbl.hash_fn.name}", y_max=y_max, height=340),
                     key=f"t2_fig_{label}")

    ratio = sd.expected_success_cost / max(su.expected_success_cost, 1e-9)
    quad = det_tbl.insert_comparisons / max(uni_tbl.insert_comparisons, 1)
    st.success(
        f"Recherche : la table déterministe est **{fr(ratio, 1)}× plus lente** "
        f"({fr(sd.expected_success_cost)} vs {fr(su.expected_success_cost)} comparaisons). "
        f"Construction : **{fr(quad, 1)}×** plus de comparaisons "
        f"({fr(det_tbl.insert_comparisons)} vs {fr(uni_tbl.insert_comparisons)})."
        if ratio >= 1 else
        f"Ici la table universelle est plus chargée ({fr(su.expected_success_cost)} vs "
        f"{fr(sd.expected_success_cost)} comparaisons) : attaque ciblée ou tirage malchanceux. "
        "Re-tirez (a, b) !"
    )

    with st.expander("📜 Historique des tirages de (a, b) — l'attaque survit-elle au re-tirage ?",
                     expanded=True):
        st.dataframe(pd.DataFrame(st.session_state.t2_history[::-1]), hide_index=True)
        st.caption(
            "Scénario de soutenance : ① 💣 l'attaque écrase la colonne A, la colonne B ne bouge pas ; "
            "② 🕵️ si (a, b) fuit, l'attaquant écrase aussi la colonne B ; "
            "③ 🎲 un simple re-tirage — m inchangé, mêmes clés — détruit l'attaque instantanément."
        )


# ==========================================================================
# Onglet 3 — Validation empirique & benchmarks
# ==========================================================================

@st.cache_data(show_spinner="Tirage des fonctions (a, b)…", max_entries=32)
def cached_collision(x: int, y: int, m: int, trials: int, p: int,
                     seed: Optional[int]) -> M.CollisionExperiment:
    """Version mise en cache de ``moteur.collision_experiment``."""
    return M.collision_experiment(x, y, m, trials, p, seed)


@st.cache_data(show_spinner="Benchmark en cours…", max_entries=16)
def cached_benchmark(sizes: tuple[int, ...], alpha: float, probes: int,
                     seed: int) -> pd.DataFrame:
    """Version mise en cache de ``moteur.run_benchmark``, en DataFrame."""
    return pd.DataFrame([vars(p) for p in M.run_benchmark(sizes, alpha, probes, seed)])


with tab3:
    st.subheader("A. Test empirique de la borne 2-universelle")
    st.latex(r"\forall x \neq y,\quad \Pr_{(a,b)}\big[h_{a,b}(x) = h_{a,b}(y)\big] \;\le\; \frac{1}{m}")
    q1, q2, q3, q4, q5 = st.columns(5)
    cx = int(q1.number_input("Clé x", 0, MAX_INPUT_KEY, 7, key="t3_x"))
    cm = int(q3.number_input("Taille m", 2, 10_000, 10, key="t3_m"))
    adversarial_pair = q2.toggle("Paire piège y = x + m", value=True,
                                 help="Collision certaine pour k mod m et (k + b) mod m.")
    cy = (cx + cm) if adversarial_pair else int(q2.number_input("Clé y", 0, MAX_INPUT_KEY, 123_456,
                                                                 key="t3_y"))
    trials = int(q4.select_slider("Tirages N", [1_000, 5_000, 10_000, 50_000, 100_000, 200_000],
                                  value=50_000, key="t3_N"))
    p_mode = q5.radio("Premier p", ["2⁶¹ − 1", "plus petit p > max(x, y, m)"], key="t3_p")
    p_val = M.P_MERSENNE_61 if p_mode.startswith("2") else M.next_prime(max(cx, cy, cm))

    if cx == cy:
        st.error("Choisissez deux clés distinctes.")
    elif cy >= p_val:
        st.error("y doit être inférieure à p.")
    else:
        seed = st.session_state.get("seed_value")
        exp = cached_collision(cx, cy, cm, trials, p_val, seed if seed is not None else 12345)
        bound = 1 / cm
        sigma = math.sqrt(bound * (1 - bound) / trials)
        st.caption(f"x = {cx}, y = {cy}, m = {cm}, p = {p_val}, N = {fr(trials)} tirages "
                   "indépendants de (a, b).")

        v1, v2 = st.columns([1, 1.6])
        with v1:
            gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=exp.frequency,
                number=dict(valueformat=".4f"),
                delta=dict(reference=bound, valueformat=".4f",
                           increasing=dict(color=CRITICAL), decreasing=dict(color=GOOD)),
                title=dict(text="Fréquence de collision observée<br>"
                                "<span style='font-size:0.8em'>référence : 1/m</span>"),
                gauge=dict(
                    axis=dict(range=[0, max(2.5 * bound, exp.frequency * 1.2)]),
                    bar=dict(color=UNI_COLOR, thickness=0.35),
                    steps=[dict(range=[0, bound], color="rgba(12,163,12,0.18)"),
                           dict(range=[bound, max(2.5 * bound, exp.frequency * 1.2)],
                                color="rgba(208,59,59,0.12)")],
                    threshold=dict(line=dict(color=CRITICAL, width=3), value=bound),
                ),
            ))
            gauge.update_layout(height=300, margin=dict(l=20, r=20, t=70, b=10))
            show(gauge)
            z = (exp.frequency - bound) / sigma if sigma else 0.0
            st.metric("Probabilité exacte (calcul combinatoire)", fr(exp.exact, 6),
                      delta=f"1/m = {fr(bound, 6)}", delta_color="off",
                      help="Σ_c n_c(n_c − 1) / (p(p − 1)), où n_c = #{r < p : r ≡ c mod m}. "
                           "Indépendante de x et y !")
            st.caption(f"Écart observé à 1/m : {z:+.2f} σ (σ = √(p(1−p)/N) = {sigma:.5f}). "
                       "Un écart de quelques σ au-dessus de 1/m n'est pas une violation : "
                       "c'est la fluctuation d'échantillonnage.")

        with v2:
            n_pts = len(exp.running_frequency)
            idx = np.unique(np.geomspace(1, n_pts, num=min(n_pts, 800)).astype(int)) - 1
            freq = np.array(exp.running_frequency)[idx]
            tt = idx + 1
            band = 3 * np.sqrt(bound * (1 - bound) / tt)
            conv = go.Figure()
            conv.add_scatter(x=np.r_[tt, tt[::-1]],
                             y=np.r_[np.clip(bound + band, 0, 1), np.clip(bound - band, 0, 1)[::-1]],
                             fill="toself", fillcolor="rgba(124,123,119,0.15)",
                             line=dict(width=0), name="1/m ± 3σ", hoverinfo="skip")
            conv.add_scatter(x=tt, y=freq, mode="lines", name="Fréquence empirique",
                             line=dict(color=UNI_COLOR, width=2),
                             hovertemplate="N = %{x}<br>fréquence = %{y:.5f}<extra></extra>")
            conv.add_hline(y=bound, line=dict(color=CRITICAL, width=2, dash="dash"),
                           annotation_text="borne 1/m", annotation_position="top right")
            conv.update_layout(title="Convergence de la fréquence (loi des grands nombres)",
                               xaxis=dict(title="Nombre de tirages (échelle log)", type="log"),
                               yaxis=dict(title="Pr[h(x) = h(y)]", rangemode="tozero"),
                               height=380, margin=dict(l=10, r=10, t=45, b=10),
                               legend=dict(orientation="h", y=-0.25), hovermode="x unified")
            show(conv)

        shift_freq = M.random_shift_collision_frequency(cx, cy, cm, min(trials, 20_000), 7)
        st.markdown("**Même paire, autres familles** — le hasard seul ne suffit pas :")
        st.dataframe(pd.DataFrame([
            {"Famille": "k mod m (déterministe)", "Pr[h(x) = h(y)]": float(cx % cm == cy % cm),
             "≤ 1/m ?": "✅" if cx % cm != cy % cm else "❌ (collision certaine)"},
            {"Famille": "(k + b) mod m, b aléatoire", "Pr[h(x) = h(y)]": shift_freq,
             "≤ 1/m ?": "✅" if shift_freq <= bound + 3 * sigma else "❌ non universelle"},
            {"Famille": "Carter-Wegman ((ak + b) mod p) mod m", "Pr[h(x) = h(y)]": exp.frequency,
             "≤ 1/m ?": "✅" if exp.frequency <= bound + 3 * sigma else "⚠️ fluctuation"},
        ]), hide_index=True,
            column_config={"Pr[h(x) = h(y)]": st.column_config.NumberColumn(format="%.5f")})

    st.divider()
    st.subheader("B. Benchmark : coût moyen d'une recherche en fonction de n")
    st.markdown(
        "À facteur de charge **α constant** (la table est redimensionnée : $m = \\lceil n/\\alpha \\rceil$), "
        "on mesure le nombre moyen de comparaisons d'une recherche fructueuse, en cherchant "
        "réellement des clés présentes tirées au hasard."
    )
    w1, w2, w3, w4 = st.columns(4)
    n_max = int(w1.select_slider("n maximal", [500, 1000, 2000, 3000, 5000], value=2000, key="t3_nmax"))
    n_points = int(w2.slider("Nombre de points", 4, 15, 8, key="t3_pts"))
    b_alpha = float(w3.select_slider("α", [0.5, 1.0, 2.0, 4.0], value=1.0, key="t3_alpha"))
    log_y = w4.toggle("Axe vertical logarithmique", value=False, key="t3_log")
    sizes = tuple(sorted({int(v) for v in np.linspace(n_max / n_points, n_max, n_points)}))
    bench = cached_benchmark(sizes, b_alpha, 300, 0)

    fig = go.Figure()
    for scenario, color in SERIES.items():
        d = bench[bench.scenario == scenario]
        fig.add_scatter(x=d.n, y=d.mean_comparisons, mode="lines+markers", name=scenario,
                        line=dict(color=color, width=2), marker=dict(size=8),
                        customdata=np.c_[d.m, d.max_chain],
                        hovertemplate=("n = %{x}, m = %{customdata[0]}<br>"
                                       "%{y:.2f} comparaisons<br>L_max = %{customdata[1]}"
                                       "<extra>" + scenario + "</extra>"))
    nn = np.linspace(min(sizes), max(sizes), 100)
    fig.add_scatter(x=nn, y=(nn + 1) / 2, mode="lines", name="Théorie Θ(n) : (n + 1)/2",
                    line=dict(color=THEORY_COLOR, dash="dash", width=1.5))
    fig.add_scatter(x=nn, y=np.full_like(nn, 1 + b_alpha / 2), mode="lines",
                    name="Théorie O(1) : 1 + α/2", line=dict(color=THEORY_COLOR, dash="dot", width=1.5))
    fig.update_layout(title="Comparaisons par recherche fructueuse",
                      xaxis_title="Nombre de clés n", yaxis_title="Comparaisons moyennes",
                      yaxis_type="log" if log_y else "linear", height=460,
                      margin=dict(l=10, r=10, t=45, b=10), legend=dict(orientation="h", y=-0.2),
                      hovermode="closest")
    show(fig)

    e1, e2 = st.columns(2)
    with e1:
        tfig = go.Figure()
        for scenario, color in SERIES.items():
            d = bench[bench.scenario == scenario]
            tfig.add_scatter(x=d.n, y=d.micros_per_search, mode="lines+markers", name=scenario,
                             line=dict(color=color, width=2), marker=dict(size=8),
                             hovertemplate="n = %{x}<br>%{y:.2f} µs<extra>" + scenario + "</extra>")
        tfig.update_layout(title="Temps mesuré par recherche (µs)", height=360,
                           xaxis_title="n", yaxis_title="µs", showlegend=False,
                           margin=dict(l=10, r=10, t=45, b=10))
        show(tfig)
    with e2:
        ifig = go.Figure()
        for scenario, color in SERIES.items():
            d = bench[bench.scenario == scenario]
            ifig.add_scatter(x=d.n, y=d.insert_comparisons, mode="lines+markers", name=scenario,
                             line=dict(color=color, width=2), marker=dict(size=8),
                             hovertemplate="n = %{x}<br>%{y:,} comparaisons<extra>"
                                           + scenario + "</extra>")
        ifig.add_scatter(x=nn, y=nn * (nn - 1) / 2, mode="lines", name="n(n − 1)/2",
                         line=dict(color=THEORY_COLOR, dash="dash", width=1.5))
        ifig.update_layout(title="Coût total de construction : Θ(n²) sous attaque", height=360,
                           xaxis_title="n", yaxis_title="Comparaisons (doublons)",
                           showlegend=False, margin=dict(l=10, r=10, t=45, b=10))
        show(ifig)
    st.caption("Mêmes couleurs que le graphique principal. Les temps dépendent de la machine ; "
               "seule leur forme (constante vs linéaire) compte.")

    adv = bench[bench.scenario == "k mod m · clés adverse"]
    slope, intercept, r_value, _, _ = sps.linregress(adv.n, adv.mean_comparisons)
    uni = bench[bench.scenario == "Carter-Wegman · clés adverse"]
    st.info(f"Régression linéaire (k mod m, clés adverses) : coût ≈ {fr(slope, 3)}·n "
            f"{'+' if intercept >= 0 else '−'} {fr(abs(intercept))} (R² = {fr(r_value ** 2, 4)}) "
            "— pente théorique 0,5.  \n"
            f"Carter-Wegman sur les mêmes clés (moyenne sur 5 tirages de (a, b)) : coût moyen "
            f"entre {fr(uni.mean_comparisons.min())} et {fr(uni.mean_comparisons.max())}, "
            f"borne en espérance 1 + α/2 = {fr(1 + b_alpha / 2)}.  \n"
            "Sur des clés en progression arithmétique, la loi du coût selon (a, b) est à "
            "*queue lourde* : la plupart des tirages font mieux que la borne, de rares tirages "
            "bien pire. La moyenne empirique fluctue donc autour de la borne — c'est une "
            "garantie en espérance, pas en pire cas ; un tirage malchanceux se corrige en re-tirant.")
    with st.expander("Voir les données brutes du benchmark"):
        st.dataframe(bench.rename(columns={
            "scenario": "Scénario", "mean_comparisons": "Comparaisons moy.",
            "max_chain": "L_max", "insert_comparisons": "Comparaisons construction",
            "micros_per_search": "µs / recherche"}), hide_index=True)


# ==========================================================================
# Onglet 4 — Fiche soutenance
# ==========================================================================

with tab4:
    st.subheader("1. Synthèse théorique")
    th1, th2 = st.columns(2)
    with th1:
        st.markdown("**Définition (famille 2-universelle, Carter & Wegman 1979).** "
                    "Soit $U$ l'univers des clés et $\\mathcal{H}$ une famille finie de fonctions "
                    "$h : U \\to \\{0,\\dots,m-1\\}$. $\\mathcal{H}$ est *universelle* si")
        st.latex(r"\forall x \neq y \in U,\quad "
                 r"\frac{\#\{h \in \mathcal{H} : h(x) = h(y)\}}{\#\mathcal{H}} \le \frac{1}{m}.")
        st.markdown("**Construction.** $p$ premier, $p > \\max(U, m)$ :")
        st.latex(r"h_{a,b}(k) = \big((ak+b) \bmod p\big) \bmod m")
        st.latex(r"\mathcal{H}_{p,m} = \{\, h_{a,b} : a \in \mathbb{Z}_p^{*},\ b \in \mathbb{Z}_p \,\},"
                 r"\qquad |\mathcal{H}_{p,m}| = p(p-1).")
        st.markdown(
            "**Théorème.** $\\mathcal{H}_{p,m}$ est universelle.  \n"
            "*Preuve.* Pour $x \\neq y$, posons $r = (ax+b) \\bmod p$ et $s = (ay+b) \\bmod p$. "
            "Alors $r - s \\equiv a(x-y) \\not\\equiv 0 \\pmod p$ car $p$ est premier "
            "(intégrité de $\\mathbb{Z}_p$) : **aucune collision avant la réduction modulo $m$**. "
            "Le système $\\{ax+b \\equiv r,\\ ay+b \\equiv s\\}$ a une unique solution "
            "$a = (r-s)(x-y)^{-1}$, $b = r - ax$ : $(a,b) \\mapsto (r,s)$ est une bijection vers "
            "les $p(p-1)$ couples $r \\neq s$. Pour $r$ fixé, au plus $\\lceil p/m \\rceil - 1 "
            "\\le (p-1)/m$ valeurs $s \\neq r$ vérifient $s \\equiv r \\pmod m$. "
            "D'où $\\Pr[h(x)=h(y)] \\le \\frac{p \\cdot (p-1)/m}{p(p-1)} = \\frac1m$. $\\blacksquare$"
        )
    with th2:
        st.markdown("**Corollaire (coût en espérance, chaînage).** Pour un ensemble $S$ de $n$ clés "
                    "**fixé à l'avance, même par un adversaire**, et $x$ quelconque :")
        st.latex(r"\mathbb{E}_h\big[\,|T[h(x)]|\,\big] = \sum_{y \in S} \Pr[h(x)=h(y)]")
        st.latex(r"\le \alpha \ \text{ si } x \notin S, \qquad "
                 r"\le 1 + \tfrac{n-1}{m} < 1 + \alpha \ \text{ si } x \in S.")
        st.markdown("En moyennant sur $x \\in S$ (chaque paire comptée une fois) : recherche "
                    "fructueuse $\\le 1 + \\frac{n-1}{2m} \\approx 1 + \\alpha/2$. Avec "
                    "$m = \\Theta(n)$ : **$O(1)$ en espérance pour tout jeu de clés**.")
        st.markdown(
            "| | Pire cas (une exécution) | Espérance |\n|---|---|---|\n"
            "| Déterministe | $\\Theta(n)$ — **atteint** par un adversaire | $O(1)$ *si* les clés sont "
            "« aléatoires » (hypothèse sur les données) |\n"
            "| Universelle | $\\Theta(n)$ — tirage malchanceux, probabilité infime | $O(1+\\alpha)$ "
            "**pour toutes** les clés (hasard sur l'algorithme) |\n"
            "| AVL / rouge-noir | $\\Theta(\\log n)$ garanti | $\\Theta(\\log n)$ |"
        )
        st.markdown("**Probabilité qu'une chaîne soit longue** (Markov) : "
                    "$\\Pr[|T[h(x)]| \\ge t] \\le (1+\\alpha)/t$. Le nombre de paires en collision "
                    "a pour espérance $\\le \\binom{n}{2}/m$ ; avec $m = n^2$ il vaut $< 1/2$ — "
                    "c'est l'idée du **hachage parfait** (FKS, 1984).")
        st.markdown("**Point clé : où est le hasard ?** Dans le *choix de l'algorithme* (de $h$), "
                    "pas dans les données. C'est la même philosophie que le tri rapide randomisé : "
                    "l'adversaire choisit l'entrée **avant** le tirage, il ne peut donc pas la "
                    "rendre mauvaise en moyenne.")

    st.divider()
    st.subheader("2. Matrice comparative interactive")
    mc1, mc2, mc3 = st.columns([1, 1, 2])
    mat_n = int(mc1.select_slider("n", [10**3, 10**4, 10**5, 10**6, 10**7], value=10**6,
                                  format_func=fr, key="t4_n"))
    mat_alpha = float(mc2.slider("Facteur de charge α", 0.1, 0.95, 0.75, 0.05, key="t4_alpha"))
    costs = M.theoretical_costs(mat_n, mat_alpha)
    qualitative = {
        "Chaînage + déterministe (k mod m)": (
            "Données « gentilles » (hypothèse)", "❌ trivialement attaquable", "Non", "n + m pointeurs"),
        "Chaînage + universelle (Carter-Wegman)": (
            "Espérance sur h, ∀ clés", "✅ si (a, b) reste secret", "Non", "n + m pointeurs"),
        "Adressage ouvert — sondage linéaire": (
            "Espérance (clés aléatoires) ; α < 1 obligatoire", "❌ (clustering primaire)",
            "Non", "m cases, excellent cache"),
        "Adressage ouvert — double hachage": (
            "Espérance (≈ hachage uniforme) ; α < 1", "⚠️ selon les fonctions", "Non", "m cases"),
        "Arbre AVL": (
            "Déterministe, pire cas", "✅ par construction", "Oui (parcours infixe, rang)",
            "n nœuds + hauteur"),
        "Arbre rouge-noir": (
            "Déterministe, pire cas", "✅ par construction", "Oui (parcours infixe, rang)",
            "n nœuds + 1 bit"),
    }
    rows = []
    for name, c in costs.items():
        g, dos, order, mem = qualitative[name]
        rows.append({"Structure": name, "Recherche fructueuse (moy.)": c["fructueuse"],
                     "Recherche infructueuse (moy.)": c["infructueuse"],
                     "Pire cas (comparaisons)": c["pire"], "Nature de la garantie": g,
                     "Résistance HashDoS": dos, "Requêtes ordonnées": order, "Mémoire": mem})
    df = pd.DataFrame(rows)
    shown = mc3.multiselect("Structures affichées", list(costs), default=list(costs), key="t4_sel")
    st.dataframe(df[df.Structure.isin(shown)], hide_index=True, column_config={
        c: st.column_config.NumberColumn(format="%.2f")
        for c in ("Recherche fructueuse (moy.)", "Recherche infructueuse (moy.)",
                  "Pire cas (comparaisons)")})
    st.caption("Formules : chaînage 1 + α/2 et α ; sondage linéaire ½(1 + 1/(1−α)) et "
               "½(1 + 1/(1−α)²) ; double hachage (1/α)·ln(1/(1−α)) et 1/(1−α) ; "
               "hauteur AVL ≤ 1,44·log₂(n+2), rouge-noir ≤ 2·log₂(n+1).")

    ns = np.unique(np.geomspace(10, mat_n, 60).astype(int))
    cfig = go.Figure()
    curves = {
        "Hachage universel (espérance, ∀ clés)": (UNI_COLOR, "solid",
                                                 [1 + mat_alpha / 2 for _ in ns]),
        "Hachage déterministe sous attaque": (DET_COLOR, "solid", [(v + 1) / 2 for v in ns]),
        "AVL (pire cas)": ("#1baf7a", "solid", [1.44 * math.log2(v + 2) for v in ns]),
        "Rouge-noir (pire cas)": ("#4a3aa7", "dash", [2 * math.log2(v + 1) for v in ns]),
    }
    for name, (color, dash, ys) in curves.items():
        cfig.add_scatter(x=ns, y=ys, mode="lines", name=name, line=dict(color=color, dash=dash, width=2),
                         hovertemplate="n = %{x:,}<br>%{y:.1f} comparaisons<extra>" + name + "</extra>")
    cfig.update_layout(title="Comparaisons par recherche : O(1) vs Θ(log n) vs Θ(n)",
                       xaxis=dict(type="log", title="n (échelle log)"),
                       yaxis=dict(type="log", title="Comparaisons (échelle log)"),
                       height=420, margin=dict(l=10, r=10, t=45, b=10),
                       legend=dict(orientation="h", y=-0.22))
    show(cfig)
    st.markdown(
        "**À retenir.** Les arbres équilibrés offrent une garantie *déterministe* "
        "$\\Theta(\\log n)$ et l'ordre des clés ; le hachage universel offre $O(1)$ *en espérance* "
        "sans hypothèse sur les données. Les deux approches sont combinées en pratique : depuis "
        "Java 8, `HashMap` transforme une alvéole qui atteint 8 éléments (si la table a au moins 64 alvéoles) en **arbre rouge-noir**, "
        "ce qui borne le pire cas à $O(\\log n)$ même sous attaque."
    )

    st.divider()
    st.subheader("3. Questions pièges du jury")
    qa = [
        ("Pourquoi ne pas simplement utiliser SHA-256 ?",
         "- **Coût** : SHA-256 traite des blocs de 512 bits en 64 tours ; c'est des dizaines de fois "
         "plus lent que $((ak+b) \\bmod p) \\bmod m$ (une multiplication, deux réductions). Or la "
         "fonction est appelée à **chaque** opération.\n"
         "- **Mauvais objectif** : SHA-256 est *fixe et public*. Résistance aux collisions sur 256 bits "
         "≠ résistance aux collisions **après réduction modulo $m$** : pour $m = 2^{16}$, trouver "
         "des clés de même alvéole par force brute coûte ~$m$ essais par clé — trivial. "
         "Une fonction publique, même cryptographique, reste attaquable par HashDoS.\n"
         "- **La vraie réponse** : il faut un *secret* tiré au hasard → une famille indexée par une "
         "clé. C'est ce que fait Carter-Wegman (garantie combinatoire prouvée) et, en pratique, "
         "**SipHash** (fonction pseudo-aléatoire à clé de 128 bits), adoptée notamment par Python 3.4 "
         "(PEP 456) et Rust après l'attaque HashDoS de 2011 (Klink & Wälde, 28C3)."),
        ("Que se passe-t-il si l'attaquant connaît a et b ?",
         "- La garantie s'effondre : il résout $ak + b \\equiv c + jm \\pmod p$, soit "
         "$k_j = (c + jm - b)\\,a^{-1} \\bmod p$, et obtient autant de clés que voulu dans "
         "l'alvéole $c$ — démontré par le bouton « 🕵️ Fuite » de l'onglet ②.\n"
         "- **Toute** fonction connue est attaquable (principe des tiroirs : $|U| > m$). La "
         "garantie est une espérance **sur le tirage secret**, l'adversaire devant choisir ses "
         "clés *avant* / sans connaître ce tirage (adversaire *oblivious*).\n"
         "- Parades : tirage à chaque démarrage du processus, **re-tirage** (rehash) dès qu'une "
         "chaîne dépasse un seuil (coût $O(n)$, amorti), garder $(a,b)$ hors de toute fuite "
         "(temps de réponse, ordre d'itération du dictionnaire !). Carter-Wegman n'est **pas** une "
         "PRF : quelques collisions observées peuvent révéler $(a,b)$ ; d'où SipHash en production."),
        ("Pourquoi p doit-il être premier et supérieur aux clés ?",
         "- **Premier** : $\\mathbb{Z}_p$ est un corps, donc $a(x-y) \\equiv 0 \\Rightarrow x \\equiv y$ "
         "(pas de diviseur de zéro) et $a$ est inversible : c'est ce qui donne la bijection "
         "$(a,b) \\leftrightarrow (r,s)$ de la preuve. Avec $p = 2^{32}$ et $x - y = 2^{31}$, "
         "$a(x-y) \\bmod 2^{32}$ ne prend que 2 valeurs : la propriété tombe.\n"
         "- **$p > \\max U$** : sinon deux clés $x \\equiv y \\pmod p$ distinctes sont "
         "confondues **pour tout** $(a,b)$ → probabilité 1. **$p > m$** pour que chaque alvéole soit atteinte.\n"
         "- On prend souvent un premier de Mersenne ($2^{61}-1$) : la réduction modulo $p$ se fait "
         "par décalages et additions."),
        ("« O(1) » : en pire cas ou en moyenne ? Et L_max ?",
         "- **En espérance**, sur le tirage de $h$, pour **tout** jeu de clés : $\\le 1 + \\alpha/2$. "
         "Le pire cas d'une exécution reste $\\Theta(n)$ mais avec probabilité négligeable.\n"
         "- Pour une fonction déterministe, le « $O(1)$ moyen » suppose des **données** aléatoires — "
         "hypothèse qu'un adversaire viole à volonté.\n"
         "- Avec une fonction idéalement aléatoire et $\\alpha = 1$, $L_{\\max} = \\Theta(\\ln n / \\ln\\ln n)$ "
         "avec forte probabilité. Une famille seulement 2-universelle garantit moins : "
         "$\\mathbb{E}[L_{\\max}] = O(\\sqrt{n})$ (via $\\binom{L_{\\max}}{2} \\le$ #paires en collision). "
         "Des familles $k$-indépendantes (polynômes de degré $k-1$) resserrent cette borne."),
        ("Le hachage déterministe est-il toujours mauvais ?",
         "- Non : sur des clés séquentielles, $k \\bmod m$ est **parfait** (une clé par alvéole). "
         "Le hachage universel ne cherche pas le meilleur cas, il **supprime le pire**.\n"
         "- Même sans adversaire, des données structurées (adresses alignées sur 8 octets, $m$ pair) "
         "peuvent dégrader $k \\bmod m$ : le hasard protège aussi contre la malchance.\n"
         "- Anecdote utile : en CPython, `hash(n)` d'un entier vaut $n \\bmod (2^{61}-1)$ et n'est "
         "**pas** randomisé (seuls `str` et `bytes` passent par SipHash) : des entiers distants "
         "d'un multiple de $2^{61}-1$ ont exactement le même haché."),
    ]
    for question, answer in qa:
        with st.expander(f"❓ {question}"):
            st.markdown(answer)

    st.divider()
    st.caption("Références : J. L. Carter & M. N. Wegman, *Universal classes of hash functions*, "
               "JCSS 18 (1979) · Cormen, Leiserson, Rivest, Stein, *Introduction to Algorithms*, "
               "chap. 11 · D. Knuth, *TAOCP* vol. 3, §6.4 · S. Crosby & D. Wallach, *Denial of "
               "Service via Algorithmic Complexity Attacks*, USENIX Security 2003 · "
               "J.-P. Aumasson & D. J. Bernstein, *SipHash*, 2012.")
