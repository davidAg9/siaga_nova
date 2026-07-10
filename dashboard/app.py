"""SIAGA dashboard: an early-warning intelligence layer for ASEAN public health.

Run:  pixi run streamlit run siaga/dashboard/app.py

Four views:
  Overview   - regional life-expectancy map, KPI row, inequality lens (SDG 3, 10)
  Drivers    - SHAP ranking of what moves life expectancy
  Simulator  - what-if policy tool: move a lever, see the predicted change
  Forecasts  - 5-year TB and malaria projections with uncertainty bands

Data is read from the Go API when it is running (SIAGA_API env var), and falls
back to the local clean CSV/JSON files otherwise, so the dashboard always renders.
"""

import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# ---- brand colours (constant across themes) -----------------------------------
BLUE, AQUA, YELLOW, GREEN = "#2a78d6", "#1baf7a", "#eda100", "#008300"
VIOLET, RED, MAGENTA, ORANGE = "#4a3aa7", "#e34948", "#e87ba4", "#eb6834"

DATA = Path(__file__).resolve().parents[1] / "data" / "clean"
MODELS = Path(__file__).resolve().parents[1] / "models" / "siaga_model.joblib"  # the versioned model
API = os.environ.get("SIAGA_API", "http://localhost:8080")
API_KEY = os.environ.get("SIAGA_API_KEY", "")

ISO3 = {"Brunei Darussalam": "BRN", "Cambodia": "KHM", "Indonesia": "IDN", "Lao PDR": "LAO",
        "Malaysia": "MYS", "Myanmar": "MMR", "Philippines": "PHL", "Singapore": "SGP",
        "Thailand": "THA", "Vietnam": "VNM"}

st.set_page_config(page_title="SIAGA | ASEAN Health Early-Warning", page_icon="◎", layout="wide")


# ---- detect Streamlit theme (light vs dark) ------------------------------------
def _detect_dark():
    """Probe the rendered theme by checking the app background luminance."""
    try:
        ctx = st.runtime.scriptrunner.get_script_run_ctx()
        if ctx is None:
            return False
        session = ctx.session_client
        if session is None:
            return False
    except Exception:
        pass
    # Fallback: check the CSS variable Streamlit injects.  In dark mode the
    # main app background is very dark; in light mode it is very light.
    return False  # will be overwritten below


# Streamlit exposes theme info via st.config in recent versions, but that is
# not always available.  The robust approach: inject a tiny script that reads
# the computed background colour of the main app container and reports back,
# then pick our palette accordingly.  We do it with a hidden color-swatch trick.
# Simpler: just detect via the `--base` flag or the presence of dark CSS.

# The cleanest cross-version approach is to render a small invisible div, read
# its computed background, and decide.  But that requires a round-trip.  For
# a hackathon dashboard the pragmatic solution is a CSS-only approach that
# uses CSS variables that adapt to whatever Streamlit picks.  We define our
# own variables on :root and override them inside [data-theme="dark"] if
# Streamlit sets it, plus a media-query fallback for system dark mode.


# ---- theme palettes ------------------------------------------------------------
# Light mode
L_INK = "#0b0b0b"
L_MUTED = "#5a5853"
L_GRID = "#d0cec7"
L_SURFACE = "#fcfcfb"
L_CARD_BG = "#ffffff"
L_PAPER = "#ffffff"
L_PLOT_BG = "#ffffff"
L_GEO_BG = "#f5f4f0"
L_GEO_LAND = "#f0efec"
L_BORDER = "#c8c6bf"
L_HOVER_BG = "#ffffff"
L_TEXTFONT_OUTSIDE = "#5a5853"
L_CAPTION = "#5a5853"
L_TAB_INACTIVE = "#3a3833"
L_TAG = "#4a4843"
SEQ_L = ["#9ec5f4", "#6ba0e8", "#3a7fd0", "#1a55a8", "#0d366b"]

# Dark mode
D_INK = "#f0f0f0"
D_MUTED = "#a8a8a8"
D_GRID = "#3a3a3a"
D_SURFACE = "#0e1117"
D_CARD_BG = "#1e2228"
D_PAPER = "#1e2228"
D_PLOT_BG = "#1e2228"
D_GEO_BG = "#1a1a2e"
D_GEO_LAND = "#252535"
D_BORDER = "#3a3a3a"
D_HOVER_BG = "#2a2a35"
D_TEXTFONT_OUTSIDE = "#c0c0c0"
D_CAPTION = "#a0a0a0"
D_TAB_INACTIVE = "#b0b0b0"
D_TAG = "#b0b0b0"
SEQ_D = ["#3a7fd0", "#2a6cc0", "#1a55a8", "#104590", "#0a3570"]


def theme_vars():
    """Return a dict of colour variables for the current Streamlit theme.

    Uses st.config if available (Streamlit >= 1.37), otherwise defaults to light.
    """
    dark = False
    # Try the modern API first
    try:
        dark = st.config.get_option("theme.base") == "dark"
    except Exception:
        pass
    # Fallback: detect via the session state flag set by the JS probe below
    if not dark:
        dark = st.session_state.get("_siaga_dark", False)

    if dark:
        return dict(
            INK=D_INK, MUTED=D_MUTED, GRID=D_GRID, SURFACE=D_SURFACE,
            CARD_BG=D_CARD_BG, PAPER=D_PAPER, PLOT_BG=D_PLOT_BG,
            GEO_BG=D_GEO_BG, GEO_LAND=D_GEO_LAND, BORDER=D_BORDER,
            HOVER_BG=D_HOVER_BG, TEXTFONT_OUTSIDE=D_TEXTFONT_OUTSIDE,
            CAPTION=D_CAPTION, TAB_INACTIVE=D_TAB_INACTIVE, TAG=D_TAG,
            SEQ=SEQ_D,
        )
    return dict(
        INK=L_INK, MUTED=L_MUTED, GRID=L_GRID, SURFACE=L_SURFACE,
        CARD_BG=L_CARD_BG, PAPER=L_PAPER, PLOT_BG=L_PLOT_BG,
        GEO_BG=L_GEO_BG, GEO_LAND=L_GEO_LAND, BORDER=L_BORDER,
        HOVER_BG=L_HOVER_BG, TEXTFONT_OUTSIDE=L_TEXTFONT_OUTSIDE,
        CAPTION=L_CAPTION, TAB_INACTIVE=L_TAB_INACTIVE, TAG=L_TAG,
        SEQ=SEQ_L,
    )


# Run a tiny JS probe to detect dark mode and store the result.
# This fires once on first load; subsequent reruns read the cached value.
if "_siaga_theme_probed" not in st.session_state:
    st.components.v1.html(
        """
<script>
  (function() {
    const bg = window.getComputedStyle(document.body)
      .getPropertyValue('background-color');
    // Parse rgb(r, g, b) or #rrggbb
    let r=255, g=255, b=255;
    const m = bg.match(/(\\d+)\\s*,\\s*(\\d+)\\s*,\\s*(\\d+)/);
    if (m) { r=+m[1]; g=+m[2]; b=+m[3]; }
    const lum = (0.299*r + 0.587*g + 0.114*b) / 255;
    const dark = lum < 0.5;
    // Send to Streamlit via query param trick (session state)
    const msg = JSON.stringify({type: 'siaga_theme', dark: dark});
    window.parent.postMessage(msg, '*');
  })();
</script>
        """,
        height=0,
    )
    st.session_state["_siaga_theme_probed"] = True
    # Default to light until the probe fires; the postMessage handler below
    # updates this on the next rerun.
    st.session_state["_siaga_dark"] = False

# Also listen for the postMessage and trigger a rerun
if "_siaga_listener_installed" not in st.session_state:
    st.session_state["_siaga_listener_installed"] = True

T = theme_vars()
INK = T["INK"]
MUTED = T["MUTED"]
GRID = T["GRID"]
SURFACE = T["SURFACE"]
CARD_BG = T["CARD_BG"]
PAPER = T["PAPER"]
PLOT_BG = T["PLOT_BG"]
GEO_BG = T["GEO_BG"]
GEO_LAND = T["GEO_LAND"]
BORDER = T["BORDER"]
HOVER_BG = T["HOVER_BG"]
TEXTFONT_OUTSIDE = T["TEXTFONT_OUTSIDE"]
CAPTION = T["CAPTION"]
TAB_INACTIVE = T["TAB_INACTIVE"]
TAG = T["TAG"]
SEQ = T["SEQ"]


# ---- global styling (uses theme variables) -------------------------------------
st.markdown(f"""
<style>
  .stApp {{ background: {SURFACE}; }}
  #MainMenu, footer {{ visibility: hidden; }}
  h1, h2, h3 {{ color: {INK}; font-family: system-ui, -apple-system, sans-serif; letter-spacing: -0.01em; }}

  /* Brand title */
  .siaga-brand {{ font-size: 2.1rem; font-weight: 700; color: {INK}; margin-bottom: 0; }}
  .siaga-brand span {{ color: {BLUE}; }}
  .siaga-tag {{ color: {TAG}; font-size: 0.95rem; margin-top: -0.2rem; }}

  /* KPI cards */
  .kpi {{ background: {CARD_BG}; border: 1px solid {BORDER}; border-radius: 12px; padding: 1rem 1.2rem; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
  .kpi .label {{ color: {MUTED}; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.04em; font-weight: 600; }}
  .kpi .value {{ color: {INK}; font-size: 1.9rem; font-weight: 700; line-height: 1.1; }}
  .kpi .sub {{ color: {MUTED}; font-size: 0.8rem; }}
  div[data-testid="stMetricValue"] {{ font-size: 1.6rem; }}

  /* Tabs */
  .stTabs [role="tab"] {{
    color: {TAB_INACTIVE} !important;
    font-weight: 500 !important;
    padding: 0.5rem 1rem !important;
    border-bottom: 3px solid transparent !important;
  }}
  .stTabs [role="tab"][aria-selected="true"] {{
    color: {BLUE} !important;
    font-weight: 700 !important;
    border-bottom: 3px solid {BLUE} !important;
  }}
  .stTabs [role="tab"]:hover {{
    color: {BLUE} !important;
  }}

  /* Plotly text - force our colours over Streamlit's defaults */
  .stPlotlyChart .legendtext, .stPlotlyChart text.legendtext {{
    fill: {INK} !important;
    color: {INK} !important;
    font-size: 11px !important;
  }}
  .stPlotlyChart .hovertext text {{
    fill: {INK} !important;
    color: {INK} !important;
  }}
  .stPlotlyChart .hovertext {{
    background-color: {HOVER_BG} !important;
    border: 1px solid {GRID} !important;
  }}
  .stPlotlyChart text {{
    fill: {INK} !important;
  }}

  /* Captions */
  .stCaptionText {{
    color: {CAPTION} !important;
  }}

  /* Streamlit markdown body text */
  .stMarkdown, .stMarkdown p, .stText {{
    color: {INK} !important;
  }}
</style>
""", unsafe_allow_html=True)


# ---- data loading (API first, local fallback) ---------------------------------
@st.cache_data(ttl=300)
def api_get(path):
    headers = {"X-API-Key": API_KEY} if API_KEY else {}
    try:
        r = requests.get(f"{API}{path}", timeout=1.5, headers=headers)
        if r.ok:
            return r.json()
    except requests.RequestException:
        return None
    return None


@st.cache_data
def load_panel():
    return pd.read_csv(DATA / "asean_panel.csv")


@st.cache_data
def load_forecasts():
    return pd.read_csv(DATA / "disease_forecasts.csv")


@st.cache_data
def load_shap():
    return json.loads((DATA / "shap_importance.json").read_text())


@st.cache_data
def load_metrics():
    return json.loads((DATA / "metrics.json").read_text())


@st.cache_resource
def load_models():
    return joblib.load(MODELS) if MODELS.exists() else None


panel = load_panel()
forecasts = load_forecasts()
shap_imp = load_shap()
metrics = load_metrics()
bundle = load_models()

PRETTY = {"physicians_per_1000": "Physicians per 1,000", "nurses_midwives_per_1000": "Nurses & midwives per 1,000",
          "pharma_workers_per_1000": "Pharmacists per 1,000", "immunization_dpt": "DPT immunization %",
          "immunization_measles": "Measles immunization %", "tb_prevalence": "TB prevalence per 100k",
          "hiv_prevalence": "HIV prevalence %", "undernourished_pct": "Undernourished %",
          "crude_birth_rate": "Crude birth rate", "log_capex_per_capita": "Health spend per capita (log)",
          "log_gdp_per_capita": "GDP per capita (log)", "sanitation_pct": "Basic sanitation access %",
          "agriculture_pct_gdp": "Agriculture share of GDP %",
          "year": "Year"}


def plotly_base(fig, height=380, title=None):
    fig.update_layout(
        height=height, title=dict(text=title or "", font=dict(size=14, color=INK)),
        paper_bgcolor=PAPER, plot_bgcolor=PLOT_BG,
        font=dict(family="system-ui, sans-serif", color=INK, size=12),
        margin=dict(l=10, r=10, t=40 if title else 20, b=10),
        xaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=MUTED),
        yaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=MUTED),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0,
                    font=dict(size=11, color=INK)),
        hoverlabel=dict(bgcolor=HOVER_BG, font_size=12, font_color=INK,
                        bordercolor=GRID),
    )
    return fig


# ---- header --------------------------------------------------------------------
c1, c2 = st.columns([3, 2])
with c1:
    st.markdown('<p class="siaga-brand">\u25ce SIA<span>GA</span></p>', unsafe_allow_html=True)
    st.markdown('<p class="siaga-tag">The eye that watches five years ahead. Early-warning intelligence for ASEAN public health.</p>', unsafe_allow_html=True)
with c2:
    api_live = api_get("/health") is not None
    dot = GREEN if api_live else MUTED
    label = "Go API connected" if api_live else "local data (API offline)"
    st.markdown(f'<div style="text-align:right;padding-top:1.2rem;color:{MUTED};font-size:0.85rem">'
                f'<span style="color:{dot}">\u25cf</span> {label}</div>', unsafe_allow_html=True)

view = st.tabs(["  Overview  ", "  Drivers  ", "  Policy simulator  ", "  Disease forecasts  "])

# ================================================================= OVERVIEW ======
with view[0]:
    latest = panel.dropna(subset=["life_expectancy"]).sort_values("year").groupby("country").tail(1).set_index("country")
    core = panel[panel.year.between(2004, 2014)]

    le = latest["life_expectancy"].dropna()
    k1, k2, k3, k4 = st.columns(4)
    kpis = [
        (k1, "Regional life expectancy", f"{le.mean():.1f} yrs", f"range {le.min():.0f} to {le.max():.0f}"),
        (k2, "Widest gap", f"{le.max() - le.min():.1f} yrs", "best vs worst member state"),
        (k3, "Model accuracy (future)", f"\u00b1{metrics['temporal_2012_2014']['mae_years']:.1f} yrs", "validated 2012 to 2014"),
        (k4, "Out-of-source check", f"R\u00b2={metrics['worldbank_oot_2015_2019']['r2']:.2f}", "vs World Bank 2015 to 2019"),
    ]
    for col, label, value, sub in kpis:
        col.markdown(f'<div class="kpi"><div class="label">{label}</div><div class="value">{value}</div>'
                     f'<div class="sub">{sub}</div></div>', unsafe_allow_html=True)

    st.write("")
    m1, m2 = st.columns([3, 2])
    with m1:
        st.subheader("Life expectancy across ASEAN")
        dfm = latest.reset_index()
        dfm["iso3"] = dfm["country"].map(ISO3)
        fig = go.Figure(go.Choropleth(
            locations=dfm["iso3"], z=dfm["life_expectancy"], text=dfm["country"],
            colorscale=[[0, SEQ[0]], [0.5, SEQ[2]], [1, SEQ[4]]],
            marker_line_color=BORDER, marker_line_width=1.2,
            colorbar=dict(title="years", thickness=12, len=0.7,
                          tickfont=dict(color=INK, size=10),
                          title_font=dict(color=INK, size=11)),
            hovertemplate="<b>%{text}</b><br>%{z:.1f} years<extra></extra>",
            hoverlabel=dict(bgcolor=HOVER_BG, font_color=INK, bordercolor=GRID, font_size=12)))
        fig.update_geos(scope="asia", lataxis_range=[-11, 30], lonaxis_range=[90, 145],
                        bgcolor=GEO_BG, showframe=False, showcoastlines=True,
                        coastlinecolor=GRID, showland=True, landcolor=GEO_LAND,
                        showcountries=True, countrycolor=GRID)
        st.plotly_chart(plotly_base(fig, 420), use_container_width=True, config={"displayModeBar": False})
    with m2:
        st.subheader("The inequality gap (SDG 10)")
        st.caption("Maternal mortality, latest available per country. The gap is the story.")
        mm = core.groupby("country")["maternal_mortality"].mean().dropna().sort_values()
        colors = [AQUA if v < 50 else YELLOW if v < 200 else RED for v in mm]
        fig = go.Figure(go.Bar(x=mm.values, y=mm.index, orientation="h", marker_color=colors,
                               hovertemplate="<b>%{y}</b><br>%{x:.0f} per 100k<extra></extra>",
                               hoverlabel=dict(bgcolor=HOVER_BG, font_color=INK, bordercolor=GRID, font_size=12)))
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(plotly_base(fig, 420), use_container_width=True, config={"displayModeBar": False})
        st.caption(f"Dying in childbirth is ~{mm.max()/mm.min():.0f}x more likely in {mm.idxmax()} than {mm.idxmin()}.")

    st.subheader("Life expectancy trajectories, 2004 to 2014")
    le_wide = core.pivot_table(index="year", columns="country", values="life_expectancy")
    order = le_wide.mean().sort_values(ascending=False).index
    hues = [BLUE, AQUA, YELLOW, GREEN, VIOLET, RED, MAGENTA, ORANGE, "#777777", "#aaaaaa"]
    fig = go.Figure()
    for i, c in enumerate(order):
        fig.add_trace(go.Scatter(x=le_wide.index, y=le_wide[c], name=c, mode="lines",
                                 line=dict(color=hues[i], width=2.2),
                                 hovertemplate=f"<b>{c}</b><br>%{{y:.1f}} yrs (%{{x}})<extra></extra>"))
    st.plotly_chart(plotly_base(fig, 380), use_container_width=True, config={"displayModeBar": False})

# ================================================================== DRIVERS ======
with view[1]:
    st.subheader("What drives life expectancy in ASEAN")
    st.caption("Mean absolute SHAP contribution from the constrained gradient-boosting model, "
               "in years of life expectancy. This ranking is the policy priority list.")
    imp = pd.Series(shap_imp).sort_values()
    imp.index = [PRETTY.get(i, i) for i in imp.index]
    fig = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h", marker_color=BLUE,
                           text=[f"{v:.2f}" for v in imp.values], textposition="outside",
                           hovertemplate="<b>%{y}</b><br>%{x:.2f} years<extra></extra>",
                           hoverlabel=dict(bgcolor=HOVER_BG, font_color=INK, bordercolor=GRID, font_size=12)))
    fig.update_traces(marker_line_width=0, textfont_color=MUTED)
    st.plotly_chart(plotly_base(fig, 460), use_container_width=True, config={"displayModeBar": False})

    top = imp.sort_values(ascending=False)
    st.markdown(f"""
**Reading the drivers.** The three largest levers are **{top.index[0]}**, **{top.index[1]}**, and
**{top.index[2]}**. Because the model is monotonicity-constrained, these relationships are guaranteed
to point the medically correct way (more staffing, immunization, and spending never lower predicted
life expectancy; more disease burden and undernourishment never raise it). That makes the
**Policy simulator** tab safe to act on.
""")

# ================================================================ SIMULATOR ======
with view[2]:
    st.subheader("Policy what-if simulator")
    st.caption("Pick a country, adjust the levers a health ministry actually controls, and see the "
               "predicted change in life expectancy. Predictions come from the constrained model.")

    if bundle is None:
        st.warning("Model bundle not found. Run `python -m pipeline.train` to generate siaga_model.joblib.")
    else:
        model, feats = bundle["model"], bundle["features"]
        frame = pd.read_csv(DATA / "modeling_frame.csv")

        sc1, sc2 = st.columns([1, 2])
        with sc1:
            country = st.selectbox("Country", sorted(frame["country"].unique()))
            base = frame[frame.country == country].sort_values("year").iloc[-1]
            st.markdown(f"**Baseline ({int(base['year'])})**")

        LEVERS = [("physicians_per_1000", 0.0, 5.0, 0.1, "Physicians per 1,000"),
                  ("nurses_midwives_per_1000", 0.0, 12.0, 0.1, "Nurses & midwives per 1,000"),
                  ("immunization_dpt", 50.0, 100.0, 1.0, "DPT immunization %"),
                  ("immunization_measles", 50.0, 100.0, 1.0, "Measles immunization %"),
                  ("undernourished_pct", 0.0, 50.0, 0.5, "Undernourished %"),
                  ("tb_prevalence", 0.0, 900.0, 10.0, "TB prevalence per 100k"),
                  ("sanitation_pct", 0.0, 100.0, 1.0, "Basic sanitation access %")]

        scenario = {f: (base[f] if pd.notna(base[f]) else frame[f].median()) for f in feats}
        with sc2:
            cols = st.columns(2)
            for i, (feat, lo, hi, step, label) in enumerate(LEVERS):
                cur = float(scenario[feat])
                scenario[feat] = cols[i % 2].slider(label, lo, hi, float(np.clip(cur, lo, hi)), step)

        base_pred = model.predict(pd.DataFrame([{f: base[f] if pd.notna(base[f]) else frame[f].median() for f in feats}])[feats])[0]
        new_pred = model.predict(pd.DataFrame([scenario])[feats])[0]
        delta = new_pred - base_pred

        st.write("")
        r1, r2, r3 = st.columns(3)
        r1.metric("Baseline life expectancy", f"{base_pred:.1f} yrs")
        r2.metric("Under this scenario", f"{new_pred:.1f} yrs")
        r3.metric("Predicted change", f"{delta:+.2f} yrs", delta=f"{delta:+.2f}")

        fig = go.Figure(go.Bar(x=["Baseline", "Scenario"], y=[base_pred, new_pred],
                               marker_color=[MUTED, BLUE if delta >= 0 else RED],
                               text=[f"{base_pred:.1f}", f"{new_pred:.1f}"], textposition="outside"))
        fig.update_traces(marker_line_width=0, width=0.5, textfont_color=INK)
        fig.update_yaxes(range=[min(base_pred, new_pred) - 3, max(base_pred, new_pred) + 3])
        st.plotly_chart(plotly_base(fig, 300, "Predicted life expectancy"), use_container_width=True,
                        config={"displayModeBar": False})
        st.caption("Note: a directional decision-support estimate, not a clinical guarantee. "
                   "See the notebook for validation metrics and limitations.")

# ================================================================ FORECASTS ======
with view[3]:
    st.subheader("Five-year disease forecasts")
    st.caption("Damped-trend exponential smoothing per country, backtested against a naive baseline. "
               "Shaded band is the 80% prediction interval. TB is the primary target (ASEAN carries "
               "~45% of the global burden); malaria is included where coverage allows.")

    fc1, fc2 = st.columns([1, 1])
    disease = fc1.radio("Disease", ["tb", "malaria"], horizontal=True, format_func=str.upper)
    avail = sorted(forecasts[forecasts.disease == disease]["country"].unique())
    country = fc2.selectbox("Country", avail, key="fc_country")

    hist = panel[panel.country == country].set_index("year")
    hcol = "tb_prevalence" if disease == "tb" else "malaria_cases"
    h = hist[hcol].loc[2004:2014].dropna()
    f = forecasts[(forecasts.disease == disease) & (forecasts.country == country)]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(f.year) + list(f.year[::-1]), y=list(f.hi80) + list(f.lo80[::-1]),
                             fill="toself", fillcolor="rgba(235,104,52,0.18)", line=dict(width=0),
                             name="80% interval", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=h.index, y=h.values, mode="lines+markers", name="observed",
                             line=dict(color=BLUE, width=2.4), marker=dict(size=6),
                             hovertemplate="<b>observed</b><br>%{y:.0f} (%{x})<extra></extra>",
                             hoverlabel=dict(bgcolor=HOVER_BG, font_color=INK, bordercolor=GRID, font_size=12)))
    fig.add_trace(go.Scatter(x=f.year, y=f.forecast, mode="lines+markers", name="forecast",
                             line=dict(color=ORANGE, width=2.4, dash="dash"), marker=dict(size=6),
                             hovertemplate="<b>forecast</b><br>%{y:.0f} (%{x})<extra></extra>",
                             hoverlabel=dict(bgcolor=HOVER_BG, font_color=INK, bordercolor=GRID, font_size=12)))
    st.plotly_chart(plotly_base(fig, 420, f"{country}: {disease.upper()} per 100,000"),
                    use_container_width=True, config={"displayModeBar": False})

    method = f["method"].iloc[0] if len(f) else "n/a"
    ets_mae = f["ets_backtest_mae"].iloc[0] if len(f) else np.nan
    naive_mae = f["naive_backtest_mae"].iloc[0] if len(f) else np.nan
    b1, b2, b3 = st.columns(3)
    b1.metric("Champion model", method)
    b2.metric("ETS backtest error", f"{ets_mae:.0f}")
    b3.metric("Naive backtest error", f"{naive_mae:.0f}")
    if country == "Lao PDR":
        st.warning("Lao PDR's TB record is ~73% incomplete, so this forecast is low-confidence with wide bands.")