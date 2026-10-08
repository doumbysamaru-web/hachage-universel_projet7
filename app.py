"""Projet 7 — Hachage universel et résistance aux collisions.

Tableau de bord Streamlit : la barre latérale règle (m, n, profil de clés,
tirage (a, b)) ; trois onglets comparent hachage déterministe et famille
2-universelle de Carter-Wegman.

    streamlit run app.py

La logique algorithmique est dans ``moteur.py`` ; ce fichier n'est que
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
import plotly.io as pio
import streamlit as st
from scipy import stats as sps

import moteur as M

# ==========================================================================
# Thème
# ==========================================================================

st.set_page_config(page_title="Hachage universel · Projet 7", page_icon="🔐",
                   layout="wide", initial_sidebar_state="expanded")

INK = "#e8e6e1"
INK_2 = "#a9a79f"
GRID = "#2b2c30"
SURFACE = "#16171b"
DET = "#e8703a"        # déterministe  (k mod m)
UNI = "#3d8bfd"        # universelle   (Carter-Wegman)
AQUA = "#1fb88a"
GOLD = "#e0a100"
GOOD = "#22b14c"
WARN = "#f0a35e"
BAD = "#e5484d"

pio.templates["lab"] = go.layout.Template(layout=dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, system-ui, sans-serif", color=INK_2, size=12),
    title=dict(font=dict(color=INK, size=14), x=0, xanchor="left"),
    colorway=[UNI, DET, AQUA, GOLD],
    xaxis=dict(gridcolor=GRID, zeroline=False, linecolor=GRID, ticks="", automargin=True,
               title=dict(standoff=8)),
    yaxis=dict(gridcolor=GRID, zeroline=False, linecolor=GRID, ticks="", automargin=True,
               title=dict(standoff=8)),
    hoverlabel=dict(bgcolor=SURFACE, bordercolor=GRID, font=dict(color=INK)),
    legend=dict(orientation="h", yanchor="top", y=-0.2, x=0, bgcolor="rgba(0,0,0,0)"),
    margin=dict(l=56, r=16, t=48, b=40),
    separators=", ",
))
pio.templates.default = "lab"

st.markdown(f"""
<style>
  .block-container {{padding-top: 1.6rem; padding-bottom: 1rem;}}
  [data-testid="stMetric"] {{background: {SURFACE}; border: 1px solid {GRID};
      border-radius: 10px; padding: 8px 12px;}}
  [data-testid="stMetricValue"] {{font-variant-numeric: tabular-nums; font-size: 1.45rem;}}
  [data-testid="stMetricLabel"] p {{font-size: .8rem; color: {INK_2};}}
  .lab-head {{display:flex; align-items:center; gap:10px; margin: 2px 0 8px 0;}}
  .lab-dot {{width:12px; height:12px; border-radius:3px; display:inline-block;}}
  .lab-title {{font-size:1.15rem; font-weight:650; color:{INK};}}
  .lab-sub {{font-family: ui-monospace, monospace; font-size:.85rem; color:{INK_2};}}
  .badge {{display:inline-block; padding:3px 10px; border-radius:999px; font-weight:700;
      font-size:.82rem; letter-spacing:.03em;}}
  .badge-ok {{background:rgba(34,177,76,.16); color:{GOOD}; border:1px solid rgba(34,177,76,.45);}}
  .badge-ko {{background:rgba(229,72,77,.16); color:{BAD}; border:1px solid rgba(229,72,77,.45);}}
  .badge-mid {{background:rgba(240,163,94,.16); color:{WARN}; border:1px solid rgba(240,163,94,.45);}}
  code.ab {{font-size:.78rem;}}
</style>
""", unsafe_allow_html=True)


# ==========================================================================
# Utilitaires
# ==========================================================================

def fr(x: float | int, d: int = 2) -> str:
    """Nombre au format français."""
    if isinstance(x, (int, np.integer)):
        return f"{int(x):,}".replace(",", " ")
    return f"{x:,.{d}f}".replace(",", " ").replace(".", ",")


def plot(fig: go.Figure, key: Optional[str] = None) -> None:
    """Affiche une figure avec le thème du tableau de bord."""
    st.plotly_chart(fig, theme=None, key=key,
                    config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]})


def badge(text: str, kind: str) -> str:
    return f"<span class='badge badge-{kind}'>{text}</span>"


def header(color: str, title: str, sub: str) -> None:
    st.markdown(f"<div class='lab-head'><span class='lab-dot' style='background:{color}'></span>"
                f"<span class='lab-title'>{title}</span>"
                f"<span class='lab-sub'>{sub}</span></div>", unsafe_allow_html=True)


def draw_ab() -> None:
    """Tire un nouveau couple (a, b) secret dans la famille de Carter-Wegman."""
    rng = st.session_state.get("rng") or random.SystemRandom()
    st.session_state.a = rng.randrange(1, M.P_MERSENNE_61)
    st.session_state.b = rng.randrange(0, M.P_MERSENNE_61)
    st.session_state.draw_id = st.session_state.get("draw_id", 0) + 1


# ==========================================================================
# Calculs (mis en cache)
# ==========================================================================

PROFILES = {
    "uniform": "Aléatoire uniforme",
    "sequential": "Séquentiel  1, 2, …, n",
    "adversarial": "Attaque DoS  kᵢ = i·m + c",
    "adaptive": "Attaque adaptative  (a, b) divulgués",
}


@st.cache_data(max_entries=64, show_spinner=False)
def make_keys(profile: str, n: int, m: int, c: int, seed: int,
              a: int = 1, b: int = 0) -> tuple[int, ...]:
    """Jeu de clés du profil choisi (``a``, ``b`` : cible de l'attaque adaptative)."""
    if profile == "uniform":
        return tuple(M.keys_uniform(n, rng=random.Random(seed)))
    if profile == "sequential":
        return tuple(M.keys_sequential(n))
    if profile == "adversarial":
        return tuple(M.keys_adversarial_modulo(n, m, c))
    target = M.CarterWegmanHash(m, a=a, b=b)
    return tuple(M.keys_adversarial_known_universal(n, target, c))


@st.cache_data(max_entries=128, show_spinner=False)
def analyse(keys: tuple[int, ...], m: int, kind: str, a: int, b: int) -> dict[str, Any]:
    """Construit la table (``kind`` = "det" ou "uni") et renvoie ses mesures."""
    h: M.HashFunction = M.ModuloHash(m) if kind == "det" else M.CarterWegmanHash(m, a=a, b=b)
    t0 = time.perf_counter()
    table = M.build_table(keys, h)
    build_ms = 1000 * (time.perf_counter() - t0)
    lengths = table.chain_lengths()
    preview = [", ".join(map(str, list(ch)[:4])) + (" …" if len(ch) > 4 else "")
               for ch in table.buckets]
    return {"stats": asdict(table.stats()), "lengths": lengths, "preview": preview,
            "insert_cmp": table.insert_comparisons, "build_ms": build_ms}


@st.cache_data(max_entries=32, show_spinner="Tirages Monte-Carlo…")
def monte_carlo(x: int, y: int, m: int, trials: int, seed: int) -> dict[str, Any]:
    e = M.collision_experiment(x, y, m, trials, seed=seed)
    n_pts = len(e.running_frequency)
    idx = np.unique(np.geomspace(1, n_pts, num=min(n_pts, 600)).astype(int)) - 1
    return {"freq": e.frequency, "exact": e.exact, "hits": e.hits,
            "t": (idx + 1).tolist(), "running": np.asarray(e.running_frequency)[idx].tolist()}


@st.cache_data(max_entries=16, show_spinner="Benchmark en cours…")
def benchmark(sizes: tuple[int, ...], alpha: float, seed: int) -> pd.DataFrame:
    return pd.DataFrame([asdict(p) for p in M.run_benchmark(sizes, alpha, 300, seed, draws=5)])


# ==========================================================================
# Barre latérale
# ==========================================================================

if "a" not in st.session_state:
    st.session_state.rng = random.SystemRandom()
    draw_ab()
    st.session_state.data_seed = 1
    st.session_state.history = []

with st.sidebar:
    st.markdown("### 🔐 Projet 7")
    st.caption("Hachage universel · résistance aux collisions")

    st.markdown("##### Table")
    m = st.slider("Taille de la table  m", 8, 512, 64, step=1,
                  help="Nombre d'alvéoles (buckets).")
    n = st.slider("Clés insérées  n", 10, 3000, 640, step=10,
                  help="Facteur de charge α = n / m.")

    st.markdown("##### Profil des clés")
    label = st.radio("Profil", list(PROFILES.values()), label_visibility="collapsed", index=2)
    profile = next(k for k, v in PROFILES.items() if v == label)
    c = 0
    if profile in ("adversarial", "adaptive"):
        c = st.number_input("Alvéole ciblée  c", 0, m - 1, 0, help="Case visée par l'attaquant.")
    if profile == "uniform" and st.button("Nouvelles clés", icon="🎲", width="stretch"):
        st.session_state.data_seed += 1
    if profile == "adaptive":
        forge_sig = (n, m, c)
        forge = st.button("Forger contre (a, b) courant", icon="🕵️", width="stretch",
                          help="L'attaquant connaît (a, b) et vise l'alvéole c.")
        if forge or st.session_state.get("forged_sig") != forge_sig:
            st.session_state.forged_sig = forge_sig
            st.session_state.forged_ab = (st.session_state.a, st.session_state.b)

    st.markdown("##### Fonction universelle")
    if st.button("Rééchantillonner (a, b)", icon="🔁", type="primary", width="stretch",
                 help="Nouvelle fonction tirée au hasard dans la famille. m et les clés sont inchangés."):
        draw_ab()
    st.markdown(f"<code class='ab'>a = {st.session_state.a}</code><br>"
                f"<code class='ab'>b = {st.session_state.b}</code><br>"
                f"<code class='ab'>p = 2⁶¹ − 1</code>", unsafe_allow_html=True)

    with st.expander("Options"):
        if st.toggle("Graine fixe (démo reproductible)", value=False):
            seed = int(st.number_input("Graine", 0, 10**6, 2026))
            if st.session_state.get("seed_used") != seed:
                st.session_state.rng = random.Random(seed)
                st.session_state.seed_used = seed
                draw_ab()
        elif st.session_state.get("seed_used") is not None:
            st.session_state.rng = random.SystemRandom()
            st.session_state.seed_used = None

a, b = st.session_state.a, st.session_state.b
fa, fb = st.session_state.get("forged_ab", (a, b))
keys = make_keys(profile, n, m, c, st.session_state.data_seed, fa, fb)
det = analyse(keys, m, "det", 0, 0)
uni = analyse(keys, m, "uni", a, b)
sd, su = det["stats"], uni["stats"]

# Historique des tirages (un point par couple (a, b), par configuration)
cfg = (profile, n, m, c, st.session_state.data_seed, fa, fb)
if st.session_state.get("history_cfg") != cfg:
    st.session_state.history_cfg = cfg
    st.session_state.history = []
hist = st.session_state.history
if not hist or hist[-1]["draw"] != st.session_state.draw_id:
    hist.append({"draw": st.session_state.draw_id, "uni": su["max_chain"], "det": sd["max_chain"]})


# ==========================================================================
# En-tête
# ==========================================================================

st.markdown("## Hachage universel & résistance aux collisions")
tab1, tab2, tab3 = st.tabs(["⚔️  Stress test", "🎯  Bornes Monte-Carlo", "📈  Complexité"])


# ==========================================================================
# Onglet 1 — Stress test
# ==========================================================================

def bucket_status(length: int, alpha: float) -> str:
    """idéale ≤ ⌈α⌉ · modérée ≤ 3α · critique au-delà."""
    if length <= max(1, math.ceil(alpha - 1e-12)):
        return "ok"
    return "mid" if length <= 3 * max(alpha, 1.0) else "ko"


def bucket_chart(res: dict[str, Any], title: str, y_max: Optional[float]) -> go.Figure:
    lengths = res["lengths"]
    alpha = res["stats"]["load_factor"]
    status = [bucket_status(L, alpha) for L in lengths]
    colors = {"ok": GOOD, "mid": WARN, "ko": BAD}
    fig = go.Figure(go.Bar(
        x=list(range(len(lengths))), y=lengths,
        marker=dict(color=[colors[s] for s in status], line=dict(width=0)),
        customdata=res["preview"],
        hovertemplate="Alvéole %{x}<br>%{y} clé(s)<br>%{customdata}<extra></extra>",
        showlegend=False,
    ))
    for s, label in (("ok", "≤ ⌈α⌉"), ("mid", "≤ 3α"), ("ko", "> 3α")):
        fig.add_trace(go.Bar(x=[None], y=[None], name=label, marker_color=colors[s]))
    fig.add_hline(y=alpha, line=dict(color=INK, dash="dot", width=1.2),
                  annotation=dict(text=f"α = {fr(alpha)}", font=dict(color=INK, size=11)),
                  annotation_position="top right")
    fig.update_layout(title=title, height=330, bargap=0.12 if len(lengths) <= 128 else 0,
                      xaxis=dict(title="alvéole", range=[-0.6, len(lengths) - 0.4]),
                      yaxis=dict(title="clés", range=[0, (y_max or max(lengths + [1])) * 1.08]),
                      legend=dict(orientation="h", y=1.0, x=1, xanchor="right", yanchor="bottom"),
                      margin=dict(t=56, b=40))
    return fig


def metrics_row(s: dict[str, Any]) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("α", fr(s["load_factor"]), help="α = n / m")
    c2.metric("L_max", fr(s["max_chain"]), help="Plus longue chaîne : coût de la pire recherche.")
    c3.metric("Vides", f"{fr(100 * s['empty_ratio'], 1)} %",
              help=f"Hachage idéal : e^(−α) = {fr(100 * math.exp(-s['load_factor']), 1)} %")
    c4.metric("Collisions", fr(s["colliding_pairs"]),
              help=f"Paires {{x, y}} avec h(x) = h(y) : Σ C(Lᵢ, 2). "
                   f"Espérance universelle ≤ C(n, 2)/m = {fr(s['n'] * (s['n'] - 1) / (2 * s['m']), 1)}.")


def regime(s: dict[str, Any]) -> str:
    if s["max_chain"] > max(3 * s["load_factor"], 8) and s["max_chain"] > s["n"] / 4:
        return badge("EFFONDREMENT  Θ(n)", "ko")
    if s["max_chain"] > 3 * max(s["load_factor"], 1) + 4:
        return badge("CHARGE IRRÉGULIÈRE", "mid")
    return badge("DISPERSION  O(1)", "ok")


with tab1:
    same_scale = st.toggle("Échelle commune", value=True, help="Même axe vertical pour A et B.")
    y_max = max(sd["max_chain"], su["max_chain"], 1) if same_scale else None
    colA, colB = st.columns(2, gap="large")
    with colA:
        header(DET, "A · Déterministe", "h(k) = k mod m")
        metrics_row(sd)
        plot(bucket_chart(det, "Profil de charge des alvéoles", y_max), key="bars_det")
    with colB:
        header(UNI, "B · Universelle", "h(k) = ((a·k + b) mod p) mod m")
        metrics_row(su)
        plot(bucket_chart(uni, "Profil de charge des alvéoles", y_max), key="bars_uni")

    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(f"**A** &nbsp; {regime(sd)}", unsafe_allow_html=True)
    k2.markdown(f"**B** &nbsp; {regime(su)}", unsafe_allow_html=True)
    speed = sd["expected_success_cost"] / max(su["expected_success_cost"], 1e-9)
    build = det["insert_cmp"] / max(uni["insert_cmp"], 1)
    k3.metric("Recherche  A / B", f"× {fr(speed, 1 if speed >= 1 else 2)}",
              help=f"Comparaisons moyennes : {fr(sd['expected_success_cost'])} (A) "
                   f"vs {fr(su['expected_success_cost'])} (B).")
    k4.metric("Construction  A / B", f"× {fr(build, 1 if build >= 1 else 2)}",
              help=f"Comparaisons à l'insertion : {fr(det['insert_cmp'])} (A) vs "
                   f"{fr(uni['insert_cmp'])} (B) · {fr(det['build_ms'], 1)} ms vs {fr(uni['build_ms'], 1)} ms.")

    if len(hist) > 1:
        h = pd.DataFrame(hist)
        h["tirage"] = range(1, len(h) + 1)
        fig = go.Figure()
        fig.add_scatter(x=h.tirage, y=h.det, name="A · k mod m", mode="lines+markers",
                        line=dict(color=DET, width=2), marker=dict(size=8))
        fig.add_scatter(x=h.tirage, y=h.uni, name="B · Carter-Wegman", mode="lines+markers",
                        line=dict(color=UNI, width=2), marker=dict(size=8))
        fig.update_layout(title="L_max au fil des rééchantillonnages de (a, b)", height=280, margin=dict(b=70),
                          xaxis=dict(title="tirage", dtick=1), yaxis=dict(title="L_max", rangemode="tozero"),
                          hovermode="x unified")
        plot(fig, key="history")


# ==========================================================================
# Onglet 2 — Bornes Monte-Carlo
# ==========================================================================

with tab2:
    st.markdown("#### Propriété 2-universelle")
    i1, i2, i3, i4 = st.columns([1, 1, 2, 1])
    x = int(i1.number_input("Clé x", 0, 2**53 - 1, 7))
    y = int(i2.number_input("Clé y", 0, 2**53 - 1, 7 + m,
                            help="Par défaut y = x + m : collision certaine pour k mod m."))
    trials = int(i3.slider("Tirages indépendants de (a, b)  N", 1_000, 50_000, 20_000, step=1_000))
    mc_seed = int(i4.number_input("Graine MC", 0, 10**6, 7, help="Change l'échantillon de tirages."))

    if x == y:
        st.error("x ≠ y requis.")
    else:
        mc = monte_carlo(x, y, m, trials, mc_seed)
        bound = 1 / m
        sigma = math.sqrt(bound * (1 - bound) / trials)
        passed = mc["freq"] <= bound + 3 * sigma
        strict = mc["freq"] <= bound

        r1, r2, r3, r4, r5 = st.columns(5)
        r1.metric("P̂ observée", fr(mc["freq"], 5), help=f"{fr(mc['hits'])} collisions sur {fr(trials)} tirages.")
        r2.metric("Borne 1/m", fr(bound, 5))
        r3.metric("P exacte", fr(mc["exact"], 5), help="Σ n_c(n_c − 1) / (p(p − 1)) : indépendante de x et y.")
        r4.metric("Écart", f"{(mc['freq'] - bound) / sigma:+.2f} σ".replace(".", ","),
                  help=f"σ = √(p(1 − p)/N) = {fr(sigma, 5)}")
        r5.metric("k mod m", "1" if (x - y) % m == 0 else "0",
                  help="Probabilité de collision de la fonction déterministe pour cette paire.")
        st.markdown(
            (badge("PASS", "ok") if passed else badge("FAIL", "ko")) + "&nbsp;&nbsp;"
            + ("P̂ ≤ 1/m" if strict else "P̂ ≤ 1/m + 3σ  (fluctuation d'échantillonnage)" if passed
               else "P̂ > 1/m + 3σ"),
            unsafe_allow_html=True, help="Test unilatéral à 3σ (≈ 99,9 %).")

        g1, g2 = st.columns([1, 1.7], gap="large")
        with g1:
            top = max(2.2 * bound, mc["freq"] * 1.15)
            gauge = go.Figure(go.Indicator(
                mode="gauge+number", value=mc["freq"],
                number=dict(valueformat=".4f", font=dict(color=GOOD if passed else BAD, size=40)),
                gauge=dict(
                    axis=dict(range=[0, top], tickcolor=INK_2, tickformat=".3f"),
                    bar=dict(color=UNI, thickness=0.28),
                    bgcolor="rgba(0,0,0,0)", borderwidth=0,
                    steps=[dict(range=[0, bound], color="rgba(34,177,76,.22)"),
                           dict(range=[bound, bound + 3 * sigma], color="rgba(240,163,94,.22)"),
                           dict(range=[bound + 3 * sigma, top], color="rgba(229,72,77,.18)")],
                    threshold=dict(line=dict(color=INK, width=3), thickness=0.9, value=bound)),
                title=dict(text="P̂  vs  1/m", font=dict(color=INK_2, size=13)),
            ))
            gauge.update_layout(height=300, margin=dict(l=40, r=40, t=50, b=0))
            plot(gauge, key="gauge")
        with g2:
            t = np.array(mc["t"])
            band = 3 * np.sqrt(bound * (1 - bound) / t)
            conv = go.Figure()
            conv.add_scatter(x=np.r_[t, t[::-1]],
                             y=np.r_[np.clip(bound + band, 0, 1), np.clip(bound - band, 0, 1)[::-1]],
                             fill="toself", fillcolor="rgba(169,167,159,.12)", line=dict(width=0),
                             name="1/m ± 3σ", hoverinfo="skip")
            conv.add_scatter(x=t, y=mc["running"], name="P̂ cumulée", mode="lines",
                             line=dict(color=UNI, width=2),
                             hovertemplate="N = %{x}<br>P̂ = %{y:.5f}<extra></extra>")
            conv.add_hline(y=bound, line=dict(color=INK, dash="dash", width=1.4),
                           annotation=dict(text="1/m", font=dict(color=INK)), annotation_position="top right")
            conv.update_layout(title="Convergence", height=320, hovermode="x unified", margin=dict(b=70),
                               xaxis=dict(title="N (log)", type="log"),
                               yaxis=dict(title="P̂", range=[0, min(1, max(4 * bound, max(mc["running"][5:] or [bound]) * 1.1))]))
            plot(conv, key="conv")

    st.markdown("#### Longueurs de chaînes  vs  Poisson(α)")
    p1, p2 = st.columns([1, 3])
    which = p1.radio("Table", ["B · Universelle", "A · Déterministe"], horizontal=False)
    res = uni if which.startswith("B") else det
    s = res["stats"]
    lengths = np.asarray(res["lengths"])
    kmax = int(max(lengths.max(), sps.poisson.ppf(0.999, s["load_factor"])))
    ks = np.arange(kmax + 1)
    observed = np.bincount(lengths, minlength=kmax + 1)[: kmax + 1] / s["m"]
    expected = sps.poisson.pmf(ks, s["load_factor"])
    tv = 0.5 * (np.abs(observed - expected).sum() + max(0.0, 1 - expected.sum()))
    p1.metric("Distance (variation totale)", fr(tv, 3),
              help="½ Σ |fréquence observée − Poisson(α)|. 0 = accord parfait, 1 = disjoint.")
    bound_pairs = s["n"] * (s["n"] - 1) / (2 * s["m"])
    ratio_pairs = s["colliding_pairs"] / max(bound_pairs, 1e-9)
    p1.metric("Collisions / C(n, 2)/m", fr(ratio_pairs),
              help="Paires en collision rapportées à leur espérance maximale pour une famille universelle.")
    p1.markdown(badge("≤ BORNE UNIVERSELLE", "ok") if ratio_pairs <= 1.25
                else badge("HORS BORNE", "ko"), unsafe_allow_html=True)
    color = UNI if which.startswith("B") else DET
    hfig = go.Figure()
    hfig.add_bar(x=ks, y=observed, name="observé", marker=dict(color=color, line=dict(width=0)),
                 hovertemplate="L = %{x}<br>%{y:.3f}<extra>observé</extra>")
    hfig.add_scatter(x=ks, y=expected, name=f"Poisson(α = {fr(s['load_factor'])})", mode="lines+markers",
                     line=dict(color=INK, width=1.5, dash="dot"), marker=dict(size=8, color=INK),
                     hovertemplate="L = %{x}<br>%{y:.3f}<extra>Poisson</extra>")
    hfig.update_layout(height=360, bargap=0.15, xaxis_title="longueur de chaîne L", margin=dict(b=70),
                       yaxis_title="proportion d'alvéoles")
    with p2:
        plot(hfig, key="poisson")


# ==========================================================================
# Onglet 3 — Profilage de complexité
# ==========================================================================

SERIES = {
    "k mod m · clés adverse": (DET, "A · pire cas (attaque)"),
    "Carter-Wegman · clés adverse": (UNI, "B · mêmes clés adverses"),
    "k mod m · clés uniforme": (GOLD, "A · clés uniformes"),
    "Carter-Wegman · clés uniforme": (AQUA, "B · clés uniformes"),
}

with tab3:
    b1, b2, b3, b4 = st.columns(4)
    n_max = int(b1.select_slider("n maximal", [500, 1000, 2000, 3000, 4000], value=2000))
    pts = int(b2.slider("Points", 4, 14, 8))
    alpha = float(b3.select_slider("α constant", [0.5, 1.0, 2.0, 4.0], value=1.0,
                                   help="La table grandit avec n : m = ⌈n / α⌉."))
    log_y = b4.toggle("Échelle log", value=True)
    sizes = tuple(sorted({int(v) for v in np.linspace(n_max / pts, n_max, pts)}))
    bench = benchmark(sizes, alpha, 0)

    adv = bench[bench.scenario == "k mod m · clés adverse"]
    cw = bench[bench.scenario == "Carter-Wegman · clés adverse"]
    slope, intercept, rv, _, _ = sps.linregress(adv.n, adv.mean_comparisons)
    cw_slope = sps.linregress(cw.n, cw.mean_comparisons).slope
    last_ratio = adv.mean_comparisons.iloc[-1] / cw.mean_comparisons.iloc[-1]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Pente A (pire cas)", fr(slope, 3), help=f"Théorie : 1/2 · R² = {fr(rv ** 2, 4)}")
    m2.metric("Pente B", fr(cw_slope, 5), help="Théorie : 0 (coût constant).")
    m3.metric("Coût B moyen", fr(cw.mean_comparisons.mean()),
              help=f"Borne en espérance : 1 + α/2 = {fr(1 + alpha / 2)}")
    m4.metric(f"Écart à n = {fr(int(adv.n.iloc[-1]))}", f"× {fr(last_ratio, 0)}")

    nn = np.linspace(min(sizes), max(sizes), 120)

    def series_fig(column: str, title: str, ytitle: str, theory: bool) -> go.Figure:
        fig = go.Figure()
        for scen, (col, label) in SERIES.items():
            d = bench[bench.scenario == scen]
            fig.add_scatter(x=d.n, y=d[column], name=label, mode="lines+markers",
                            line=dict(color=col, width=2.2), marker=dict(size=8),
                            customdata=np.c_[d.m, d.max_chain],
                            hovertemplate="n = %{x} · m = %{customdata[0]}<br>%{y:.2f}"
                                          "<br>L_max = %{customdata[1]}<extra>" + label + "</extra>")
        if theory:
            fig.add_scatter(x=nn, y=(nn + 1) / 2, name="(n + 1)/2", mode="lines",
                            line=dict(color=INK_2, dash="dash", width=1.2), hoverinfo="skip")
            fig.add_scatter(x=nn, y=np.full_like(nn, 1 + alpha / 2), name="1 + α/2", mode="lines",
                            line=dict(color=INK_2, dash="dot", width=1.2), hoverinfo="skip")
        fig.update_layout(title=title, height=440, xaxis_title="n", margin=dict(b=90),
                          yaxis=dict(title=ytitle, type="log" if log_y else "linear"))
        return fig

    f1, f2 = st.columns(2, gap="large")
    with f1:
        plot(series_fig("mean_comparisons", "Comparaisons par recherche", "comparaisons", True), key="cmp")
    with f2:
        plot(series_fig("micros_per_search", "Temps par recherche", "µs", False), key="time")

    ifig = go.Figure()
    for scen, (col, label) in SERIES.items():
        d = bench[bench.scenario == scen]
        ifig.add_bar(x=d.n, y=d.insert_comparisons, name=label, marker=dict(color=col, line=dict(width=0)))
    ifig.add_scatter(x=nn, y=nn * (nn - 1) / 2, name="n(n − 1)/2", mode="lines",
                     line=dict(color=INK_2, dash="dash", width=1.2))
    ifig.update_layout(title="Coût de construction (vérification des doublons)", height=360, margin=dict(b=70),
                       barmode="group", bargap=0.2, xaxis_title="n",
                       yaxis=dict(title="comparaisons", type="log" if log_y else "linear"))
    plot(ifig, key="build")

    with st.expander("Données brutes"):
        st.dataframe(bench, hide_index=True, width="stretch")
