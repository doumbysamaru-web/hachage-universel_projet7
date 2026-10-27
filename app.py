"""Projet 7 — Hachage universel et résistance aux collisions.

Démonstrateur Streamlit pensé pour une soutenance courte : deux tables de
hachage reçoivent exactement les mêmes clés, seule la fonction change.

* Table A : h(k) = k mod m            (fonction fixe, connue de tous)
* Table B : h(k) = ((a·k + b) mod p) mod m, (a, b) tirés au hasard et secrets

Quatre onglets suivent l'ordre de l'exposé :
    1. L'attaque           clés piégées : A s'effondre, B tient
    2. Pr[collision] ≤ 1/m vérification Monte-Carlo de la définition
    3. Coût quand n grandit linéaire (A) contre constant (B)
    4. Si (a, b) fuit      la limite : la garantie repose sur le secret

    streamlit run app.py

Toute la logique algorithmique est dans ``moteur.py`` ; ce fichier n'est que
l'interface.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import asdict
from typing import Any, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import stats as sps

import moteur as M

st.set_page_config(page_title="Hachage universel", page_icon=":material/tag:",
                   layout="wide", initial_sidebar_state="expanded")

# Couleurs des séries, lisibles sur fond clair comme sur fond sombre.
COL_A = "#D9623B"       # A · fonction fixe
COL_B = "#2A9D8F"       # B · famille universelle
COL_REF = "#8C959C"     # repères théoriques
SEED_SOUTENANCE = 2026

st.markdown("""
<style>
  .block-container {padding-top: 2.2rem; max-width: 1240px;}
  [data-testid="stMetricValue"] {font-variant-numeric: tabular-nums;}
  [data-testid="stMetricLabel"] p {font-size: .86rem;}
  h1 {letter-spacing: -.01em;}
  .lede {font-size: 1.08rem; opacity: .82; margin: -.4rem 0 1.1rem 0; max-width: 62ch;}
  .say {margin: .6rem 0 .3rem 0; font-family: "IBM Plex Serif", serif; font-size: 1.06rem;
        line-height: 1.5; max-width: 75ch;}
  .say-k {font-family: "IBM Plex Sans", sans-serif; font-size: .72rem; font-weight: 600;
          letter-spacing: .08em; text-transform: uppercase; opacity: .6; margin-right: .6rem;}
</style>
""", unsafe_allow_html=True)


# ==========================================================================
# Utilitaires
# ==========================================================================

def fr(x: float | int, d: int = 2) -> str:
    """Nombre au format français (espace fine pour les milliers, virgule)."""
    if isinstance(x, (int, np.integer)):
        return f"{int(x):,}".replace(",", " ")
    return f"{x:,.{d}f}".replace(",", " ").replace(".", ",")


def times(r: float) -> str:
    return f"× {fr(r, 0)}" if r >= 10 else f"× {fr(r, 1)}"


def plot(fig: go.Figure, key: str) -> None:
    fig.update_layout(font=dict(size=13), hoverlabel=dict(font_size=13), separators=", ",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, xanchor="left"))
    if any(tr.showlegend is not False for tr in fig.data) and len(fig.data) > 1:
        fig.update_layout(margin=dict(t=44))
    st.plotly_chart(fig, theme="streamlit", key=key, width="stretch",
                    config={"displayModeBar": False})


def say(text: str) -> None:
    """Phrase-clé à dire pendant la soutenance."""
    st.markdown(f"<div class='say'><span class='say-k'>À retenir</span>{text}</div>", unsafe_allow_html=True)


def draw_ab() -> None:
    """Tire un nouveau couple (a, b) secret dans la famille de Carter-Wegman."""
    rng = st.session_state.rng
    st.session_state.a = rng.randrange(1, M.P_MERSENNE_61)
    st.session_state.b = rng.randrange(0, M.P_MERSENNE_61)
    st.session_state.draw_id += 1


def reset_rng(reproducible: bool, seed: int) -> None:
    st.session_state.rng = random.Random(seed) if reproducible else random.SystemRandom()
    st.session_state.draw_id = 0
    draw_ab()


# ==========================================================================
# Calculs (mis en cache)
# ==========================================================================

@st.cache_data(max_entries=64, show_spinner=False)
def make_keys(profile: str, n: int, m: int, c: int, seed: int,
              a: int = 1, b: int = 0) -> tuple[int, ...]:
    if profile == "uniform":
        return tuple(M.keys_uniform(n, rng=random.Random(seed)))
    if profile == "sequential":
        return tuple(M.keys_sequential(n))
    if profile == "adversarial":
        return tuple(M.keys_adversarial_modulo(n, m, c))
    return tuple(M.keys_adversarial_known_universal(n, M.CarterWegmanHash(m, a=a, b=b), c))


@st.cache_data(max_entries=128, show_spinner=False)
def analyse(keys: tuple[int, ...], m: int, kind: str, a: int, b: int) -> dict[str, Any]:
    """Construit la table A (``kind="det"``) ou B (``"uni"``) et renvoie ses mesures."""
    h: M.HashFunction = M.ModuloHash(m) if kind == "det" else M.CarterWegmanHash(m, a=a, b=b)
    t0 = time.perf_counter()
    table = M.build_table(keys, h)
    build_ms = 1000 * (time.perf_counter() - t0)
    preview = [", ".join(map(str, list(ch)[:3])) + (" …" if len(ch) > 3 else "") for ch in table.buckets]
    return {"stats": asdict(table.stats()), "lengths": table.chain_lengths(), "preview": preview,
            "insert_cmp": table.insert_comparisons, "build_ms": build_ms}


@st.cache_data(max_entries=32, show_spinner="Tirages de (a, b) en cours…")
def monte_carlo(x: int, y: int, m: int, trials: int, seed: int) -> dict[str, Any]:
    e = M.collision_experiment(x, y, m, trials, seed=seed)
    n_pts = len(e.running_frequency)
    idx = np.unique(np.geomspace(1, n_pts, num=min(n_pts, 500)).astype(int)) - 1
    return {"freq": e.frequency, "exact": e.exact, "hits": e.hits,
            "t": (idx + 1).tolist(), "running": np.asarray(e.running_frequency)[idx].tolist()}


@st.cache_data(max_entries=16, show_spinner="Mesure du coût pour plusieurs tailles de table…")
def benchmark(sizes: tuple[int, ...], alpha: float, seed: int) -> pd.DataFrame:
    return pd.DataFrame([asdict(p) for p in M.run_benchmark(sizes, alpha, 300, seed, draws=5)])


# ==========================================================================
# Graphiques
# ==========================================================================

def bucket_chart(res: dict[str, Any], color: str, y_max: float, key: str) -> None:
    """Nombre de clés dans chaque case : une barre par case."""
    lengths = res["lengths"]
    alpha = res["stats"]["load_factor"]
    fig = go.Figure(go.Bar(
        x=list(range(len(lengths))), y=lengths, marker=dict(color=color, line=dict(width=0)),
        customdata=res["preview"],
        hovertemplate="Case %{x} : %{y} clé(s)<br>%{customdata}<extra></extra>", showlegend=False))
    fig.add_hline(y=alpha, line=dict(color=COL_REF, dash="dot", width=1.2),
                  annotation=dict(text=f"moyenne α = {fr(alpha, 1)}", font=dict(color=COL_REF, size=12)),
                  annotation_position="top right")
    fig.update_layout(height=260, margin=dict(l=8, r=8, t=8, b=8),
                      bargap=0.15 if len(lengths) <= 128 else 0,
                      xaxis=dict(title="case de la table", range=[-0.6, len(lengths) - 0.4]),
                      yaxis=dict(title="clés", range=[0, y_max * 1.08]))
    plot(fig, key)


def regime(s: dict[str, Any]) -> tuple[str, str]:
    """Verdict lisible sur la table (texte, couleur de badge)."""
    if s["max_chain"] > max(3 * s["load_factor"], 8) and s["max_chain"] > s["n"] / 4:
        return "Effondrement : une seule longue liste", "red"
    if s["max_chain"] > 3 * max(s["load_factor"], 1) + 4:
        return "Répartition irrégulière", "orange"
    return "Bien réparti", "green"


def table_card(title: str, color_name: str, res: dict[str, Any],
               color: str, y_max: float, key: str) -> None:
    s = res["stats"]
    with st.container(border=True):
        st.markdown(f"##### :{color_name}[{title}]")
        label, badge_color = regime(s)
        st.badge(label, color=badge_color)
        c1, c2 = st.columns(2)
        c1.metric("Plus longue liste", fr(s["max_chain"]),
                  help="Le coût d'une recherche dans le pire des cas : il faut parcourir cette liste.")
        c2.metric("Collisions", fr(s["colliding_pairs"]),
                  help="Nombre de paires de clés rangées dans la même case. C'est aussi le nombre "
                       "de comparaisons faites pour construire la table (on vérifie les doublons).")
        bucket_chart(res, color, y_max, key)
        st.caption(f"Cases vides : {fr(100 * s['empty_ratio'], 1)} % · survolez une barre pour voir les clés.")


# ==========================================================================
# État initial et barre latérale
# ==========================================================================

if "a" not in st.session_state:
    st.session_state.reproducible = True
    st.session_state.seed = SEED_SOUTENANCE
    st.session_state.data_seed = 1
    st.session_state.history = []
    reset_rng(True, SEED_SOUTENANCE)

with st.sidebar:
    st.markdown("## Hachage universel")
    st.caption("Projet 7 · L3 Informatique · Université de Caen Normandie")

    st.markdown("##### La table")
    m = st.slider("Nombre de cases m", 8, 512, 64, step=1)
    n = st.slider("Nombre de clés n", 10, 3000, 640, step=10,
                  help="α = n / m est le nombre moyen de clés par case.")
    st.caption(f"Facteur de charge α = n / m = **{fr(n / m, 1)}** clés par case en moyenne.")

    st.markdown("##### Le secret de la table B")
    if st.button("Nouveau tirage de (a, b)", icon=":material/casino:", type="primary", width="stretch",
                 help="Tire une autre fonction dans la famille. Les clés ne changent pas."):
        draw_ab()
    st.caption(f"Tirage n° {st.session_state.draw_id} · p = 2⁶¹ − 1")
    with st.expander("Voir a et b"):
        st.code(f"a = {st.session_state.a}\nb = {st.session_state.b}\np = 2^61 − 1", language=None)

    st.markdown("##### Soutenance")
    repro = st.toggle("Tirages reproductibles", value=st.session_state.reproducible,
                      help="Rejoue toujours la même suite de tirages, pour retrouver les chiffres de l'exposé.")
    if repro != st.session_state.reproducible:
        st.session_state.reproducible = repro
        reset_rng(repro, st.session_state.seed)
        st.rerun()
    with st.expander("Déroulé de la démo (≈ 4 min)"):
        st.markdown(
            "1. **L'attaque**, clés piégées : A met tout dans une case, B reste équilibrée.\n"
            "2. Cliquer **Nouveau tirage de (a, b)** : B reste bonne à chaque tirage.\n"
            "3. **Pr[collision] ≤ 1/m** : la fréquence mesurée colle à 1/m.\n"
            "4. **Coût quand n grandit** : A monte en ligne droite, B reste plate.\n"
            "5. **Si (a, b) fuit** : B tombe à son tour. Un nouveau tirage la répare.")

a, b = st.session_state.a, st.session_state.b

# Historique des tirages : une valeur de L_max par tirage, pour la config de l'onglet 1
def record_history(cfg: tuple, la: int, lb: int) -> list[dict[str, int]]:
    if st.session_state.get("history_cfg") != cfg:
        st.session_state.history_cfg = cfg
        st.session_state.history = []
    hist = st.session_state.history
    if not hist or hist[-1]["draw"] != st.session_state.draw_id:
        hist.append({"draw": st.session_state.draw_id, "A": la, "B": lb})
    return hist


# ==========================================================================
# En-tête
# ==========================================================================

st.title("Hachage universel")
st.markdown("<p class='lede'>Deux tables de hachage reçoivent exactement les mêmes clés. "
            "Seule la fonction qui choisit la case change.</p>", unsafe_allow_html=True)

h1, h2 = st.columns(2)
with h1.container(border=True):
    st.markdown(":orange[**Table A · fonction fixe**]")
    st.markdown("`h(k) = k mod m` · la même pour tout le monde, donc connue de l'attaquant.")
with h2.container(border=True):
    st.markdown(":blue[**Table B · famille universelle**]")
    st.markdown("`h(k) = ((a·k + b) mod p) mod m` · (a, b) tirés au hasard au démarrage et gardés secrets.")

tab1, tab2, tab3, tab4 = st.tabs(["1 · L'attaque", "2 · Pr[collision] ≤ 1/m",
                                  "3 · Coût quand n grandit", "4 · Si (a, b) fuit"])


# ==========================================================================
# Onglet 1 — L'attaque
# ==========================================================================

PROFILES = {"adversarial": "Clés piégées  kᵢ = i·m + c", "uniform": "Clés aléatoires",
            "sequential": "Clés consécutives  1, 2, …, n"}

with tab1:
    st.markdown("**Quelles clés l'attaquant envoie-t-il ?**")
    profile = st.segmented_control("Jeu de clés", list(PROFILES), format_func=PROFILES.get,
                                   default="adversarial", required=True, label_visibility="collapsed",
                                   key="profile")
    c = 0
    if profile == "adversarial":
        st.caption(f"Toutes ces clés ont le même reste modulo m = {m} : avec k mod m, elles visent la case c = 0.")
    elif profile == "uniform":
        cc1, _ = st.columns([1, 3])
        if cc1.button("Autres clés aléatoires", icon=":material/refresh:"):
            st.session_state.data_seed += 1

    keys = make_keys(profile, n, m, c, st.session_state.data_seed)
    det = analyse(keys, m, "det", 0, 0)
    uni = analyse(keys, m, "uni", a, b)
    sd, su = det["stats"], uni["stats"]
    y_max = max(sd["max_chain"], su["max_chain"], 1)

    colA, colB = st.columns(2, gap="medium")
    with colA:
        table_card("A · fonction fixe", "orange", det, COL_A, y_max, "bars_a")
    with colB:
        table_card("B · famille universelle", "blue", uni, COL_B, y_max, "bars_b")

    search_ratio = sd["expected_success_cost"] / max(su["expected_success_cost"], 1e-9)
    build_ratio = det["insert_cmp"] / max(uni["insert_cmp"], 1)
    r1, r2, r3 = st.columns(3)
    r1.metric("Recherche A / B", times(search_ratio), border=True,
              help=f"Comparaisons par recherche : {fr(sd['expected_success_cost'], 1)} pour A, "
                   f"{fr(su['expected_success_cost'], 1)} pour B.")
    r2.metric("Construction A / B", times(build_ratio), border=True,
              help=f"Comparaisons pour tout insérer : {fr(det['insert_cmp'])} pour A, {fr(uni['insert_cmp'])} pour B.")
    bound = n * (n - 1) / (2 * m)
    r3.metric("Collisions de B", fr(su["colliding_pairs"]), delta=f"borne : {fr(bound, 0)}",
              delta_color="off", delta_arrow="off", border=True,
              help="La théorie promet en moyenne au plus C(n, 2)/m paires en collision, quelles que soient les clés.")

    if profile == "adversarial":
        say("Avec une fonction fixe et connue, l'attaquant met toutes les clés dans la même case. "
            "Avec un tirage secret, les mêmes clés restent bien réparties.")
    elif profile == "uniform":
        say("Sur des clés ordinaires, les deux tables se valent : le problème n'apparaît qu'avec des clés choisies.")
    else:
        say("Ici, k mod m est même parfait. Le hachage universel ne vise pas le meilleur cas : il supprime le pire.")

    hist = record_history((profile, n, m, st.session_state.data_seed), sd["max_chain"], su["max_chain"])
    if len(hist) > 1:
        hd = pd.DataFrame(hist)
        hd["tirage"] = range(1, len(hd) + 1)
        fig = go.Figure()
        fig.add_scatter(x=hd.tirage, y=hd.A, name="A · fonction fixe", mode="lines+markers",
                        line=dict(color=COL_A, width=2, dash="dot"), marker=dict(size=7))
        fig.add_scatter(x=hd.tirage, y=hd.B, name="B · famille universelle", mode="lines+markers",
                        line=dict(color=COL_B, width=2.5), marker=dict(size=8))
        fig.update_layout(height=240, margin=dict(l=8, r=8, t=8, b=8), hovermode="x unified",
                          xaxis=dict(title="tirage de (a, b) pendant la démo", dtick=1),
                          yaxis=dict(title="plus longue liste", rangemode="tozero"))
        st.space("small")
        st.markdown("**Plus longue liste à chaque nouveau tirage**")
        plot(fig, "history")
    else:
        st.caption("Astuce : cliquez plusieurs fois sur « Nouveau tirage de (a, b) » dans la barre latérale ; "
                   "un graphique montrera que B reste bonne à chaque tirage.")


# ==========================================================================
# Onglet 2 — Pr[collision] ≤ 1/m
# ==========================================================================

with tab2:
    st.markdown("**On fixe deux clés x ≠ y, on tire (a, b) un grand nombre de fois, "
                "et on compte la proportion de tirages où h(x) = h(y).**")
    i1, i2, i3 = st.columns([1, 1, 2])
    x = int(i1.number_input("Clé x", 0, 2**53 - 1, 7))
    y = int(i2.number_input("Clé y", 0, 2**53 - 1, 7 + m,
                            help="Par défaut y = x + m : avec k mod m, ces deux clés collisionnent toujours."))
    trials = int(i3.slider("Nombre de tirages N", 1_000, 50_000, 20_000, step=1_000))

    if x == y:
        st.error("Choisissez deux clés différentes.")
    else:
        mc_seed = 7
        mc = monte_carlo(x, y, m, trials, mc_seed)
        p_th = 1 / m
        sigma = math.sqrt(p_th * (1 - p_th) / trials)
        z = (mc["freq"] - p_th) / sigma
        passed = mc["freq"] <= p_th + 3 * sigma
        det_collide = (x - y) % m == 0

        q1, q2, q3, q4 = st.columns(4)
        q1.metric("Fréquence mesurée (B)", fr(mc["freq"], 4), border=True,
                  help=f"{fr(mc['hits'])} collisions sur {fr(trials)} tirages.")
        q2.metric("Théorie : 1/m", fr(p_th, 4), border=True)
        q3.metric("Écart", f"{z:+.2f} σ".replace(".", ","), border=True,
                  help=f"σ = √(P(1 − P)/N) = {fr(sigma, 5)} mesure les fluctuations normales du hasard. "
                       "Un écart de moins de 3 σ est normal.")
        q4.metric("Avec A (k mod m)", "toujours" if det_collide else "jamais", border=True,
                  help=f"{x} ≡ {y} (mod {m})" if det_collide else f"{x} ≢ {y} (mod {m})")
        if passed:
            st.badge("Vérifié : la fréquence mesurée est compatible avec la borne 1/m", icon=":material/check:",
                     color="green")
        else:
            st.badge("Écart anormal : fréquence au-dessus de 1/m + 3σ", icon=":material/close:", color="red")

        t = np.array(mc["t"])
        band = 3 * np.sqrt(p_th * (1 - p_th) / t)
        conv = go.Figure()
        conv.add_scatter(x=np.r_[t, t[::-1]],
                         y=np.r_[np.clip(p_th + band, 0, 1), np.clip(p_th - band, 0, 1)[::-1]],
                         fill="toself", fillcolor="rgba(140,149,156,.18)", line=dict(width=0),
                         name="fluctuations normales du hasard (± 3σ)", hoverinfo="skip")
        conv.add_scatter(x=t, y=mc["running"], name="fréquence mesurée", mode="lines",
                         line=dict(color=COL_B, width=2.5),
                         hovertemplate="après %{x} tirages : %{y:.4f}<extra></extra>")
        conv.add_hline(y=p_th, line=dict(color=COL_REF, dash="dash", width=1.4),
                       annotation=dict(text="1/m", font=dict(color=COL_REF)), annotation_position="top right")
        top = min(1, max(4 * p_th, max(mc["running"][5:] or [p_th]) * 1.1))
        conv.update_layout(height=340, margin=dict(l=8, r=8, t=8, b=8), hovermode="x unified",
                           xaxis=dict(title="nombre de tirages de (a, b) (échelle log)", type="log"),
                           yaxis=dict(title="proportion de collisions", range=[0, top]))
        plot(conv, "conv")
        say("Plus on fait de tirages, plus la fréquence se rapproche de 1/m : "
            "la propriété universelle se vérifie par l'expérience.")

        with st.expander("Pour aller plus loin"):
            st.markdown(
                f"- **Probabilité exacte** pour cette famille : {fr(mc['exact'], 6)}, la même pour toute "
                f"paire x ≠ y (preuve : la bijection (a, b) ↔ (r, s)).\n"
                f"- **Test** : chaque tirage donne collision ou non. Sur N tirages, la fréquence a pour "
                f"écart-type σ = √(P(1 − P)/N). On accepte si la fréquence ≤ 1/m + 3σ (≈ 99,9 %).")
            uk = make_keys("uniform", n, m, 0, st.session_state.data_seed)
            ures = analyse(uk, m, "uni", a, b)
            us = ures["stats"]
            lengths = np.asarray(ures["lengths"])
            kmax = int(max(lengths.max(), sps.poisson.ppf(0.999, us["load_factor"])))
            ks = np.arange(kmax + 1)
            observed = np.bincount(lengths, minlength=kmax + 1)[: kmax + 1] / us["m"]
            expected = sps.poisson.pmf(ks, us["load_factor"])
            pf = go.Figure()
            pf.add_bar(x=ks, y=observed, name="table B, clés aléatoires",
                       marker=dict(color=COL_B, line=dict(width=0)))
            pf.add_scatter(x=ks, y=expected, name=f"loi de Poisson (α = {fr(us['load_factor'], 1)})",
                           mode="lines+markers", line=dict(color=COL_REF, dash="dot"), marker=dict(size=7))
            pf.update_layout(height=280, margin=dict(l=8, r=8, t=8, b=8), bargap=0.15,
                             xaxis_title="longueur d'une liste", yaxis_title="proportion des cases")
            st.markdown("**Longueur des listes de B avec des clés aléatoires, comparée à un hasard parfait**")
            plot(pf, "poisson")
            st.caption("L'universalité ne promet pas une loi de Poisson ; elle borne seulement les collisions. "
                       "Avec seulement m cases, cette comparaison reste bruitée.")


# ==========================================================================
# Onglet 3 — Coût quand n grandit
# ==========================================================================

with tab3:
    st.markdown("**On agrandit la table en même temps que n (α constant), avec des clés piégées, "
                "et on compte les comparaisons faites par une recherche.**")
    b1, b2, b3 = st.columns([1, 1, 1])
    n_max = int(b1.select_slider("n maximal", [500, 1000, 2000, 3000, 4000], value=2000))
    alpha = float(b2.select_slider("Clés par case α", [0.5, 1.0, 2.0, 4.0], value=1.0,
                                   help="La table grandit avec n : m = ⌈n / α⌉."))
    show_uniform = b3.toggle("Ajouter les clés aléatoires", value=False)
    sizes = tuple(sorted({int(v) for v in np.linspace(n_max / 8, n_max, 8)}))
    bench = benchmark(sizes, alpha, 0)

    adv = bench[bench.scenario == "k mod m · clés adverse"]
    cw = bench[bench.scenario == "Carter-Wegman · clés adverse"]
    slope_a = sps.linregress(adv.n, adv.mean_comparisons).slope
    slope_b = sps.linregress(cw.n, cw.mean_comparisons).slope
    last = adv.mean_comparisons.iloc[-1] / cw.mean_comparisons.iloc[-1]
    s1, s2, s3 = st.columns(3)
    s1.metric("A : comparaisons en plus par clé ajoutée", fr(slope_a, 2), border=True,
              help="Théorie : 1/2, car une recherche parcourt en moyenne (n + 1)/2 clés de la liste unique.")
    s2.metric("B : comparaisons en plus par clé ajoutée", fr(abs(slope_b), 4), border=True,
              help=f"Théorie : 0, coût constant autour de 1 + α/2 = {fr(1 + alpha / 2, 1)}.")
    s3.metric(f"Écart à n = {fr(int(adv.n.iloc[-1]))}", times(last), border=True)

    series = [("k mod m · clés adverse", COL_A, "A · clés piégées", "solid"),
              ("Carter-Wegman · clés adverse", COL_B, "B · clés piégées", "solid")]
    if show_uniform:
        series += [("k mod m · clés uniforme", COL_A, "A · clés aléatoires", "dot"),
                   ("Carter-Wegman · clés uniforme", COL_B, "B · clés aléatoires", "dot")]
    nn = np.linspace(min(sizes), max(sizes), 100)
    fig = go.Figure()
    for scen, col, label, dash in series:
        d = bench[bench.scenario == scen]
        fig.add_scatter(x=d.n, y=d.mean_comparisons, name=label, mode="lines+markers",
                        line=dict(color=col, width=2.5, dash=dash), marker=dict(size=7),
                        hovertemplate="n = %{x}<br>%{y:.1f} comparaisons<extra>" + label + "</extra>")
    fig.add_scatter(x=nn, y=(nn + 1) / 2, name="théorie A : (n + 1)/2", mode="lines",
                    line=dict(color=COL_REF, dash="dash", width=1.2), hoverinfo="skip")
    fig.add_scatter(x=nn, y=np.full_like(nn, 1 + alpha / 2), name="théorie B : 1 + α/2", mode="lines",
                    line=dict(color=COL_REF, dash="dot", width=1.2), hoverinfo="skip")
    fig.update_layout(height=400, margin=dict(l=8, r=8, t=8, b=8), hovermode="x unified",
                      xaxis_title="nombre de clés n", yaxis_title="comparaisons par recherche")
    plot(fig, "cmp")
    say("Sous attaque, la recherche dans A coûte de plus en plus cher, en ligne droite : Θ(n). "
        "Dans B, elle reste constante : O(1).")

    with st.expander("Pour aller plus loin : temps mesurés et coût de construction"):
        e1, e2 = st.columns(2)
        allser = [("k mod m · clés adverse", COL_A, "A · clés piégées"),
                  ("Carter-Wegman · clés adverse", COL_B, "B · clés piégées")]
        with e1:
            tf = go.Figure()
            for scen, col, label in allser:
                d = bench[bench.scenario == scen]
                tf.add_scatter(x=d.n, y=d.micros_per_search, name=label, mode="lines+markers",
                               line=dict(color=col, width=2))
            tf.update_layout(height=300, margin=dict(l=8, r=8, t=8, b=8),
                             xaxis_title="n", yaxis_title="µs par recherche")
            st.markdown("**Temps par recherche** (bruité : serveur partagé)")
            plot(tf, "time")
        with e2:
            bf = go.Figure()
            for scen, col, label in allser:
                d = bench[bench.scenario == scen]
                bf.add_bar(x=d.n, y=d.insert_comparisons, name=label, marker=dict(color=col, line=dict(width=0)))
            bf.add_scatter(x=nn, y=nn * (nn - 1) / 2, name="n(n − 1)/2", mode="lines",
                           line=dict(color=COL_REF, dash="dash", width=1.2))
            bf.update_layout(height=300, margin=dict(l=8, r=8, t=8, b=8), barmode="group",
                             xaxis_title="n", yaxis=dict(title="comparaisons (log)", type="log"))
            st.markdown("**Coût de construction de la table**")
            plot(bf, "build")
        st.dataframe(bench, hide_index=True, width="stretch")


# ==========================================================================
# Onglet 4 — Si (a, b) fuit
# ==========================================================================

with tab4:
    st.markdown("**Si l'attaquant apprend (a, b), il calcule des clés qui tombent toutes "
                "dans la même case de B.**")
    st.caption("Clés forgées : kᵢ = (c + i·m − b) · a⁻¹ mod p, avec c = 0.")
    # None : l'attaquant connaît le secret actuel ; sinon (a, b, n° du tirage) qu'il a appris.
    leak = st.session_state.get("leak")
    la, lb, ldraw = leak or (a, b, st.session_state.draw_id)

    l1, l2, _ = st.columns([1.2, 1.2, 2])
    if l1.button("Changer le secret", icon=":material/casino:", width="stretch",
                 help="Nouveau tirage de (a, b) : les clés forgées visaient l'ancien secret."):
        st.session_state.leak = (la, lb, ldraw)
        draw_ab()
        st.rerun()
    if l2.button("Divulguer le secret actuel", icon=":material/lock_open:", width="stretch",
                 disabled=leak is None):
        st.session_state.leak = None
        st.rerun()

    leaked_now = (la, lb) == (a, b)
    if leaked_now:
        st.badge(f"Le secret du tirage n° {ldraw} est connu de l'attaquant", icon=":material/lock_open:",
                 color="red")
    else:
        st.badge(f"Clés forgées contre le tirage n° {ldraw}, mais la table B utilise le tirage n° "
                 f"{st.session_state.draw_id}", icon=":material/lock:", color="green")

    fkeys = make_keys("adaptive", n, m, 0, 0, la, lb)
    fdet = analyse(fkeys, m, "det", 0, 0)
    funi = analyse(fkeys, m, "uni", a, b)
    fy = max(fdet["stats"]["max_chain"], funi["stats"]["max_chain"], 1)
    f1, f2 = st.columns(2, gap="medium")
    with f1:
        table_card("A · fonction fixe", "orange", fdet, COL_A, fy, "leak_a")
    with f2:
        table_card("B · famille universelle", "blue", funi, COL_B, fy, "leak_b")
    if leaked_now:
        say("Si le secret fuit, la famille universelle s'effondre comme une fonction fixe : "
            "la protection vient du secret du tirage. Universel ne veut pas dire cryptographique.")
    else:
        say("Un nouveau tirage suffit : les clés forgées contre l'ancien secret sont de nouveau bien réparties.")
