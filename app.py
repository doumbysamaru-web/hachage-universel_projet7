"""Projet 7 — Hachage universel et résistance aux collisions.

Tableau de bord Streamlit pour une soutenance courte (5 min + 5 min de
questions). Les explications sont données à l'oral : l'écran ne montre que
des réglages, des métriques et des graphiques.

* Table A : h(k) = k mod m                       fonction fixe, connue de tous
* Table B : h(k) = ((a·k + b) mod p) mod m       (a, b) tirés au hasard, secrets

Trois vues :
    Attaque       mêmes clés dans A et B (aléatoires, piégées, ou forgées
                  contre un secret divulgué), métriques A / B en direct
    Pr[collision] Monte-Carlo : fréquence mesurée contre 1/m
    Complexité    comparaisons et temps par recherche quand n grandit

    streamlit run app.py

Toute la logique algorithmique est dans ``moteur.py``.
"""

from __future__ import annotations

import math
import random
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import stats as sps

import moteur as M

st.set_page_config(page_title="Hachage universel", page_icon=":material/tag:",
                   layout="wide", initial_sidebar_state="expanded")

SEED_SOUTENANCE = 2026
MC_SEED = 7
PROBES = 300


def _is_dark() -> bool:
    try:
        return st.context.theme.type == "dark"
    except Exception:  # anciennes versions de Streamlit
        return False


DARK = _is_dark()
COL_A = "#EE8A63" if DARK else "#C2512A"      # A · fonction fixe
COL_B = "#4DBDB3" if DARK else "#1F7F7A"      # B · famille universelle
COL_REF = "#8C959C"                           # repères théoriques
BAND = "rgba(140,149,156,.20)"

st.markdown(f"""
<style>
  .block-container {{padding-top: 1.4rem; padding-bottom: 2rem; max-width: 1320px;}}
  h1 {{letter-spacing: -.01em; font-size: 2.1rem !important; padding: 0 0 .2rem 0 !important;}}
  .fns {{display:flex; flex-wrap:wrap; gap:.5rem; margin:.35rem 0 .2rem 0;}}
  .fn {{font-family:"IBM Plex Mono", monospace; font-size:.86rem; padding:.22rem .6rem;
        border-radius:.4rem; border:1px solid rgba(128,128,128,.28);}}
  .fn b {{font-family:"IBM Plex Sans", sans-serif; font-weight:600; margin-right:.4rem;}}
  .fa b {{color:{COL_A};}} .fb b {{color:{COL_B};}}
  .kpi {{border:1px solid rgba(128,128,128,.28); border-radius:.6rem; padding:.75rem 1rem .7rem;
         height:100%; display:flex; flex-direction:column; gap:.3rem; margin-bottom:.9rem;}}
  .kpi .lbl {{font-size:.7rem; letter-spacing:.08em; text-transform:uppercase; opacity:.62; font-weight:600;}}
  .kpi .vals {{display:flex; flex-direction:column; gap:.1rem;}}
  .kpi .v {{font-family:"IBM Plex Mono", monospace; font-size:1.5rem; font-weight:500; line-height:1.15;
            font-variant-numeric:tabular-nums; white-space:nowrap;}}
  .kpi .v i {{font-style:normal; font-family:"IBM Plex Sans", sans-serif; font-size:.72rem;
              font-weight:600; margin-right:.45rem; display:inline-block; width:.8rem; vertical-align:middle;}}
  .kpi .foot {{font-size:.8rem; opacity:.72; margin-top:auto; display:flex; gap:.5rem; align-items:center; flex-wrap:wrap;}}
  .x {{font-family:"IBM Plex Mono", monospace; font-weight:600; font-size:.82rem; padding:.08rem .5rem;
       border-radius:99px; background:{COL_B}26; color:{COL_B};}}
  .x.xa {{background:{COL_A}26; color:{COL_A};}}
  .va {{color:{COL_A};}} .vb {{color:{COL_B};}}
  .panel-h {{display:flex; justify-content:space-between; align-items:center; gap:.5rem; flex-wrap:wrap;
             font-weight:600; margin-bottom:-.3rem;}}
  [data-testid="stMetricValue"] {{font-variant-numeric: tabular-nums;}}
</style>
""", unsafe_allow_html=True)


# ==========================================================================
# Utilitaires
# ==========================================================================

def fr(x: float | int, d: int = 1) -> str:
    """Nombre au format français."""
    if isinstance(x, (int, np.integer)):
        return f"{int(x):,}".replace(",", " ")
    return f"{x:,.{d}f}".replace(",", " ").replace(".", ",")


def num(x: float) -> str:
    """Comme fr, sans « ,0 » inutile."""
    return fr(int(round(x))) if abs(x - round(x)) < 1e-9 else fr(x)


def times(r: float) -> str:
    return f"× {fr(r, 0)}" if r >= 9.95 else f"× {fr(r, 1)}"


def ratio(va: float, vb: float) -> str:
    """Pastille : combien de fois la pire table est plus chère que l'autre."""
    va, vb = max(va, 1e-9), max(vb, 1e-9)
    if max(va, vb) / min(va, vb) < 1.15:
        return "<span class='x'>A ≈ B</span>"
    worse, r, cls = ("A", va / vb, "x xa") if va > vb else ("B", vb / va, "x")
    return f"<span class='{cls}'>{worse} {times(r)}</span>"


def kpi(label: str, a: str | None, b: str | None, foot: str = "",
        single: str | None = None, color: str | None = None) -> None:
    """Tuile de métrique : A et B côte à côte, ou une seule valeur."""
    if single is not None:
        style = f" style='color:{color}'" if color else ""
        vals = f"<span class='v'{style}>{single}</span>"
    else:
        vals = (f"<span class='v va'><i>A</i>{a}</span>"
                f"<span class='v vb'><i>B</i>{b}</span>")
    st.markdown(f"<div class='kpi'><span class='lbl'>{label}</span><div class='vals'>{vals}</div>"
                f"<div class='foot'>{foot}</div></div>", unsafe_allow_html=True)


def plot(fig: go.Figure, key: str) -> None:
    fig.update_layout(font=dict(size=13), hoverlabel=dict(font_size=13), separators=", ",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, xanchor="left"))
    st.plotly_chart(fig, theme="streamlit", key=key, width="stretch",
                    config={"displayModeBar": False})


def draw_ab() -> None:
    """Nouveau couple (a, b) secret dans la famille de Carter-Wegman."""
    rng = st.session_state.rng
    st.session_state.a = rng.randrange(1, M.P_MERSENNE_61)
    st.session_state.b = rng.randrange(0, M.P_MERSENNE_61)
    st.session_state.draw_id += 1


def reset_rng(reproducible: bool) -> None:
    st.session_state.rng = random.Random(SEED_SOUTENANCE) if reproducible else random.SystemRandom()
    st.session_state.draw_id = 0
    draw_ab()


# ==========================================================================
# Calculs
# ==========================================================================

@st.cache_data(max_entries=64, show_spinner=False)
def make_keys(profile: str, n: int, m: int, seed: int, a: int = 1, b: int = 0) -> tuple[int, ...]:
    if profile == "uniform":
        return tuple(M.keys_uniform(n, rng=random.Random(seed)))
    if profile == "adversarial":
        return tuple(M.keys_adversarial_modulo(n, m, 0))
    # L'attaquant connaît (a, b) : il vise des valeurs internes c + r·m (r au hasard, c = 0),
    # puis inverse a·k + b ≡ c + r·m (mod p). Toutes ces clés tombent dans la case 0.
    p = M.P_MERSENNE_61
    inv_a = pow(a, -1, p)
    targets = random.Random(seed).sample(range((p - 1) // m), n)
    return tuple(((r * m - b) * inv_a) % p for r in targets)


@st.cache_resource(max_entries=64, show_spinner=False)
def build(keys: tuple[int, ...], m: int, kind: str, a: int, b: int) -> M.HashTableChaining:
    """Table A (``kind="det"``) ou B (``"uni"``), construite une fois par configuration."""
    h: M.HashFunction = M.ModuloHash(m) if kind == "det" else M.CarterWegmanHash(m, a=a, b=b)
    return M.build_table(keys, h)


def live_search(table: M.HashTableChaining, keys: tuple[int, ...]) -> tuple[float, float]:
    """Recherches réelles, chronométrées à chaque affichage : (comparaisons, µs)."""
    probes = random.Random(1).sample(keys, min(PROBES, len(keys)))
    return M.measure_search(table, probes)


@st.cache_data(max_entries=32, show_spinner="Tirages de (a, b)…")
def monte_carlo(x: int, y: int, m: int, trials: int, seed: int) -> dict[str, Any]:
    e = M.collision_experiment(x, y, m, trials, seed=seed)
    n_pts = len(e.running_frequency)
    idx = np.unique(np.geomspace(1, n_pts, num=min(n_pts, 500)).astype(int)) - 1
    return {"freq": e.frequency, "hits": e.hits, "t": (idx + 1).tolist(),
            "running": np.asarray(e.running_frequency)[idx].tolist()}


@st.cache_data(max_entries=16, show_spinner="Mesures pour plusieurs tailles de table…")
def benchmark(sizes: tuple[int, ...], alpha: float) -> pd.DataFrame:
    """Mêmes clés et mêmes tirages à chaque passe : on garde le meilleur temps de 3 passes."""
    runs = [pd.DataFrame([vars(p) for p in M.run_benchmark(sizes, alpha, PROBES, 0, draws=5)])
            for _ in range(3)]
    df = runs[0].copy()
    df["micros_per_search"] = np.minimum.reduce([r["micros_per_search"].to_numpy() for r in runs])
    return df


# ==========================================================================
# Graphiques
# ==========================================================================

def bucket_chart(lengths: list[int], alpha: float, color: str, y_max: float, key: str) -> None:
    fig = go.Figure(go.Bar(x=list(range(len(lengths))), y=lengths,
                           marker=dict(color=color, line=dict(width=0)),
                           hovertemplate="case %{x} · %{y} clé(s)<extra></extra>", showlegend=False))
    fig.add_hline(y=alpha, line=dict(color=COL_REF, dash="dot", width=1.2),
                  annotation=dict(text=f"α = {num(alpha)}", font=dict(color=COL_REF, size=12)),
                  annotation_position="top right")
    fig.update_layout(height=245, margin=dict(l=4, r=4, t=6, b=4),
                      bargap=0.15 if len(lengths) <= 128 else 0,
                      xaxis=dict(title="case", range=[-0.6, len(lengths) - 0.4]),
                      yaxis=dict(title="clés", range=[0, y_max * 1.06]))
    plot(fig, key)


def regime(s: M.TableStats) -> tuple[str, str]:
    if s.max_chain > max(3 * s.load_factor, 8) and s.max_chain > s.n / 4:
        return "Effondrement", "red"
    if s.max_chain > 3 * max(s.load_factor, 1) + 4:
        return "Irrégulière", "orange"
    return "Bien répartie", "green"


# ==========================================================================
# État et barre latérale
# ==========================================================================

if "a" not in st.session_state:
    st.session_state.reproducible = True
    st.session_state.data_seed = 1
    st.session_state.history = []
    st.session_state.leak = None
    st.session_state.mc_run = 0
    reset_rng(True)

with st.sidebar:
    st.markdown("#### Table")
    m = st.slider("Cases m", 8, 512, 64)
    n = st.slider("Clés n", 10, 3000, 640, step=10)
    st.markdown(f"<span class='x'>α = n/m = {num(n / m)}</span>", unsafe_allow_html=True)

    st.divider()
    st.markdown("#### Secret de B")
    if st.button("Nouveau tirage (a, b)", icon=":material/casino:", type="primary", width="stretch",
                 key="draw"):
        draw_ab()
    s1, s2 = st.columns([1, 1], vertical_alignment="center")
    s1.markdown(f"**Tirage n° {st.session_state.draw_id}**")
    with s2.popover("a, b", icon=":material/key:", width="stretch"):
        st.code(f"a = {st.session_state.a}\nb = {st.session_state.b}\np = 2^61 − 1", language=None)

    st.divider()
    repro = st.toggle("Tirages reproductibles", value=st.session_state.reproducible,
                      help="Graine 2026 : mêmes tirages, mêmes chiffres que le diaporama.")
    if repro != st.session_state.reproducible:
        st.session_state.reproducible = repro
        reset_rng(repro)
        st.rerun()

a, b = st.session_state.a, st.session_state.b


# ==========================================================================
# En-tête
# ==========================================================================

st.title("Hachage universel")
st.markdown("<div class='fns'><span class='fn fa'><b>A</b>h(k) = k mod m</span>"
            "<span class='fn fb'><b>B</b>h(k) = ((a·k + b) mod p) mod m · p = 2⁶¹ − 1</span></div>",
            unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs([":material/bolt: Attaque", ":material/casino: Pr[collision] ≤ 1/m",
                            ":material/trending_up: Complexité"])


# ==========================================================================
# Vue 1 — Attaque
# ==========================================================================

SCEN = {"uniform": "Clés aléatoires", "adversarial": "Attaque sur k mod m", "leak": "Fuite de (a, b)"}

with tab1:
    c1, c2 = st.columns([4, 1], vertical_alignment="center")
    scen = c1.segmented_control("Scénario", list(SCEN), format_func=SCEN.get, default="adversarial",
                                required=True, label_visibility="collapsed", key="scen")
    same_scale = c2.toggle("Même échelle", value=True)
    if scen == "leak":
        if st.session_state.leak is None:
            st.session_state.leak = (a, b, st.session_state.draw_id)
        la, lb, ldraw = st.session_state.leak
        leaked_now = (la, lb) == (a, b)
        d1, d2, _ = st.columns([3, 1, 2], vertical_alignment="center")
        with d1:
            if leaked_now:
                st.badge(f"L'attaquant connaît le secret n° {ldraw}", icon=":material/lock_open:", color="red")
            else:
                st.badge(f"Clés forgées contre le secret n° {ldraw} · table sur le n° {st.session_state.draw_id}",
                         icon=":material/lock:", color="green")
        if d2.button("Divulguer", icon=":material/lock_open:", disabled=leaked_now, width="stretch"):
            st.session_state.leak = (a, b, st.session_state.draw_id)
            st.rerun()
        keys = make_keys("leak", n, m, ldraw, la, lb)
    else:
        st.session_state.leak = None
        if scen == "uniform":
            d1, _ = st.columns([1, 5])
            if d1.button("Autres clés", icon=":material/refresh:", width="stretch"):
                st.session_state.data_seed += 1
        keys = make_keys(scen, n, m, st.session_state.data_seed)

    ta, tb = build(keys, m, "det", 0, 0), build(keys, m, "uni", a, b)
    sa, sb = ta.stats(), tb.stats()
    cmp_a, us_a = live_search(ta, keys)
    cmp_b, us_b = live_search(tb, keys)
    bound = n * (n - 1) / (2 * m)

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        kpi("Liste max", fr(sa.max_chain), fr(sb.max_chain),
            f"moyenne α = {num(n / m)}")
    with k2:
        kpi("Collisions", fr(sa.colliding_pairs), fr(sb.colliding_pairs),
            f"{ratio(sa.colliding_pairs, sb.colliding_pairs)}"
            f"borne B : {fr(bound, 0)}")
    with k3:
        kpi("Comparaisons", fr(sa.expected_success_cost), fr(sb.expected_success_cost),
            f"{ratio(sa.expected_success_cost, sb.expected_success_cost)}par recherche")
    with k4:
        kpi("Temps mesuré", f"{fr(us_a, 2)} µs", f"{fr(us_b, 2)} µs",
            f"{ratio(us_a, us_b)}par recherche · {PROBES} essais")

    y_max = max(sa.max_chain, sb.max_chain, 1)
    g1, g2 = st.columns(2, gap="medium")
    for col, name, s, table, color, key in ((g1, "A · k mod m", sa, ta, COL_A, "ba"),
                                            (g2, "B · Carter-Wegman", sb, tb, COL_B, "bb")):
        with col.container(border=True):
            h1, h2 = st.columns([2, 1], vertical_alignment="center")
            h1.markdown(f"<span style='color:{color};font-weight:600'>{name}</span>", unsafe_allow_html=True)
            label, bcol = regime(s)
            with h2:
                st.badge(label, color=bcol)
            bucket_chart(table.chain_lengths(), s.load_factor, color,
                         y_max if same_scale else max(s.max_chain, 1), key)

    # Historique : une valeur de L_max par tirage, pour cette configuration
    cfg = (scen, n, m, st.session_state.data_seed, st.session_state.leak)
    if st.session_state.get("history_cfg") != cfg:
        st.session_state.history_cfg = cfg
        st.session_state.history = []
    hist = st.session_state.history
    if not hist or hist[-1]["draw"] != st.session_state.draw_id:
        hist.append({"draw": st.session_state.draw_id, "A": sa.max_chain, "B": sb.max_chain})
    hd = pd.DataFrame(hist)
    fig = go.Figure()
    fig.add_scatter(x=hd.draw, y=hd.A, name="A", mode="lines+markers",
                    line=dict(color=COL_A, width=2, dash="dot"), marker=dict(size=7))
    fig.add_scatter(x=hd.draw, y=hd.B, name="B", mode="lines+markers",
                    line=dict(color=COL_B, width=2.5), marker=dict(size=8))
    fig.update_layout(height=200, margin=dict(l=4, r=4, t=30, b=4), hovermode="x unified",
                      xaxis=dict(title="tirage de (a, b)", dtick=1), yaxis=dict(title="liste max", rangemode="tozero"))
    with st.container(border=True):
        st.markdown("<div class='panel-h'>Plus longue liste à chaque tirage</div>", unsafe_allow_html=True)
        plot(fig, "history")


# ==========================================================================
# Vue 2 — Pr[collision] ≤ 1/m
# ==========================================================================

with tab2:
    i1, i2, i3, i4 = st.columns([1, 1, 2, 1], vertical_alignment="bottom")
    x = int(i1.number_input("Clé x", 0, 2**53 - 1, 7))
    y = int(i2.number_input("Clé y", 0, 2**53 - 1, 7 + m))
    trials = int(i3.slider("Tirages de (a, b)", 1_000, 50_000, 20_000, step=1_000))
    if i4.button("Relancer", icon=":material/replay:", width="stretch"):
        st.session_state.mc_run += 1

    if x == y:
        st.error("Deux clés différentes.")
    else:
        mc = monte_carlo(x, y, m, trials, MC_SEED + st.session_state.mc_run)
        p_th = 1 / m
        sigma = math.sqrt(p_th * (1 - p_th) / trials)
        z = (mc["freq"] - p_th) / sigma
        passed = mc["freq"] <= p_th + 3 * sigma
        det = (x - y) % m == 0

        q1, q2, q3, q4 = st.columns(4)
        with q1:
            kpi("Fréquence mesurée · B", None, None, f"{fr(mc['hits'])} collisions / {fr(trials)}",
                single=fr(mc["freq"], 4), color=COL_B)
        with q2:
            kpi("Théorie · 1/m", None, None, f"m = {m}", single=fr(p_th, 4))
        with q3:
            verdict = "<span class='x'>vérifié ≤ 1/m + 3σ</span>" if passed else "<span class='x'>hors borne</span>"
            kpi("Écart", None, None, verdict, single=f"{z:+.2f} σ".replace(".", ","))
        with q4:
            kpi("Fréquence · A", None, None, f"{x} {'≡' if det else '≢'} {y} (mod {m})",
                single="100 %" if det else "0 %", color=COL_A)

        t = np.array(mc["t"])
        band = 3 * np.sqrt(p_th * (1 - p_th) / t)
        conv = go.Figure()
        conv.add_scatter(x=np.r_[t, t[::-1]],
                         y=np.r_[np.clip(p_th + band, 0, 1), np.clip(p_th - band, 0, 1)[::-1]],
                         fill="toself", fillcolor=BAND, line=dict(width=0), name="± 3σ", hoverinfo="skip")
        conv.add_scatter(x=t, y=mc["running"], name="fréquence mesurée (B)", mode="lines",
                         line=dict(color=COL_B, width=2.5),
                         hovertemplate="%{x} tirages · %{y:.4f}<extra></extra>")
        conv.add_hline(y=p_th, line=dict(color=COL_REF, dash="dash", width=1.4),
                       annotation=dict(text="1/m", font=dict(color=COL_REF)), annotation_position="top right")
        top = min(1, max(4 * p_th, max(mc["running"][5:] or [p_th]) * 1.1))
        conv.update_layout(height=430, margin=dict(l=4, r=4, t=30, b=4), hovermode="x unified",
                           xaxis=dict(title="tirages de (a, b) · échelle log", type="log",
                                      tickvals=[1, 10, 100, 1_000, 10_000, 50_000]),
                           yaxis=dict(title="Pr[h(x) = h(y)]", range=[0, top]))
        with st.container(border=True):
            st.markdown("<div class='panel-h'>Convergence vers 1/m</div>", unsafe_allow_html=True)
            plot(conv, "conv")


# ==========================================================================
# Vue 3 — Complexité
# ==========================================================================

with tab3:
    b1, b2, b3 = st.columns([1, 1, 1], vertical_alignment="bottom")
    n_max = int(b1.select_slider("n maximal", [500, 1000, 2000, 3000, 4000], value=2000))
    alpha = float(b2.select_slider("α (clés par case)", [0.5, 1.0, 2.0, 4.0], value=1.0))
    show_uniform = b3.toggle("Clés aléatoires en plus", value=False)
    sizes = tuple(sorted({int(v) for v in np.linspace(n_max / 8, n_max, 8)}))
    bench = benchmark(sizes, alpha)

    adv = bench[bench.scenario == "k mod m · clés adverse"]
    cw = bench[bench.scenario == "Carter-Wegman · clés adverse"]
    slope_a = sps.linregress(adv.n, adv.mean_comparisons).slope
    slope_b = sps.linregress(cw.n, cw.mean_comparisons).slope
    n_last = int(adv.n.iloc[-1])

    r1, r2, r3, r4 = st.columns(4)
    with r1:
        kpi("Pente", fr(slope_a, 2), fr(abs(slope_b), 3),
            "comparaisons par clé ajoutée")
    with r2:
        kpi(f"Comparaisons à n = {fr(n_last)}", fr(adv.mean_comparisons.iloc[-1]),
            fr(cw.mean_comparisons.iloc[-1]), f"{ratio(adv.mean_comparisons.iloc[-1], cw.mean_comparisons.iloc[-1])}")
    with r3:
        kpi(f"Temps à n = {fr(n_last)}", f"{fr(adv.micros_per_search.iloc[-1])} µs",
            f"{fr(cw.micros_per_search.iloc[-1], 2)} µs", f"{ratio(adv.micros_per_search.iloc[-1], cw.micros_per_search.iloc[-1])}")
    with r4:
        kpi("Croissance", "Θ(n)", "O(1)", "recherche, clés piégées")

    series = [("k mod m · clés adverse", COL_A, "A · piégées", "solid"),
              ("Carter-Wegman · clés adverse", COL_B, "B · piégées", "solid")]
    if show_uniform:
        series += [("k mod m · clés uniforme", COL_A, "A · aléatoires", "dot"),
                   ("Carter-Wegman · clés uniforme", COL_B, "B · aléatoires", "dot")]
    nn = np.linspace(min(sizes), max(sizes), 100)

    e1, e2 = st.columns(2, gap="medium")
    fig = go.Figure()
    for scen_name, col, label, dash in series:
        d = bench[bench.scenario == scen_name]
        fig.add_scatter(x=d.n, y=d.mean_comparisons, name=label, mode="lines+markers",
                        line=dict(color=col, width=2.5, dash=dash), marker=dict(size=7),
                        hovertemplate="n = %{x} · %{y:.1f}<extra>" + label + "</extra>")
    fig.add_scatter(x=nn, y=(nn + 1) / 2, name="(n + 1)/2", mode="lines",
                    line=dict(color=COL_REF, dash="dash", width=1.2), hoverinfo="skip")
    fig.add_scatter(x=nn, y=np.full_like(nn, 1 + alpha / 2), name="1 + α/2", mode="lines",
                    line=dict(color=COL_REF, dash="dot", width=1.2), hoverinfo="skip")
    fig.update_layout(height=400, margin=dict(l=4, r=4, t=30, b=4), hovermode="x unified",
                      xaxis_title="n", yaxis_title="comparaisons / recherche")
    with e1.container(border=True):
        st.markdown("<div class='panel-h'>Comparaisons par recherche</div>", unsafe_allow_html=True)
        plot(fig, "cmp")

    tf = go.Figure()
    for scen_name, col, label, dash in series:
        d = bench[bench.scenario == scen_name]
        tf.add_scatter(x=d.n, y=d.micros_per_search, name=label, mode="lines+markers",
                       line=dict(color=col, width=2.5, dash=dash), marker=dict(size=7),
                       hovertemplate="n = %{x} · %{y:.2f} µs<extra>" + label + "</extra>")
    tf.update_layout(height=400, margin=dict(l=4, r=4, t=30, b=4), hovermode="x unified",
                     xaxis_title="n", yaxis_title="µs / recherche")
    with e2.container(border=True):
        st.markdown("<div class='panel-h'>Temps mesuré par recherche</div>", unsafe_allow_html=True)
        plot(tf, "time")
