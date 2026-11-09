"""
Universal Bank - Personal Loan Campaign Intelligence (Streamlit)
Run from the repo root:   streamlit run personal_loan/app.py
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import engine as E

st.set_page_config(page_title="Personal Loan Intelligence", page_icon="🏦", layout="wide",
                   initial_sidebar_state="expanded")

# ============================================================== design tokens
SURFACE, CARD = "#161b22", "#161b22"
INK, INK2, MUTED, GRID, AXIS = "#f0efec", "#c3c2b7", "#898781", "#2c2c2a", "#383835"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = ("#3987e5", "#d95926", "#199e70", "#c98500",
                                                           "#d55181", "#0ca30c", "#9085e9", "#e66767")
ACC, DEC = BLUE, "#5f5e5a"                       # accepted / declined
CAT = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, VIOLET, GREEN, RED]
SEQ = [[0, "#0d1f36"], [0.2, "#0d366b"], [0.45, "#184f95"], [0.7, "#2a78d6"], [0.88, "#6da7ec"], [1, "#cde2fb"]]
DIV = [[0, "#c24141"], [0.5, "#2b2b29"], [1, "#2a78d6"]]
TIER_COL = {"Hot (50%+)": "#e5484d", "Warm (20-50%)": "#f08c3a", "Cool (5-20%)": "#c98500", "Cold (<5%)": "#5f5e5a"}
PCFG = {"displaylogo": False, "scrollZoom": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]}

st.markdown(f"""
<style>
  .block-container {{padding-top: 1.3rem; padding-bottom: 3rem; max-width: 1500px;}}
  .hero {{padding: 24px 28px; border-radius: 18px; margin-bottom: 16px;
          background: radial-gradient(1100px 280px at 0% 0%, rgba(57,135,229,.28), transparent 60%),
                      radial-gradient(900px 280px at 100% 0%, rgba(25,158,112,.20), transparent 60%), {CARD};
          border: 1px solid rgba(255,255,255,.08);}}
  .hero h1 {{margin: 0; font-size: 2.0rem; font-weight: 800; letter-spacing: -.02em;
             background: linear-gradient(90deg, #f0efec, #86b6ef 55%, #5fd3a6); -webkit-background-clip: text;
             -webkit-text-fill-color: transparent;}}
  .hero p {{margin: 6px 0 0; color: {INK2}; font-size: .98rem;}}
  .kpi {{background: {CARD}; border: 1px solid rgba(255,255,255,.08); border-radius: 14px; padding: 14px 16px;
         height: 100%; position: relative; overflow: hidden;}}
  .kpi::before {{content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 4px; background: var(--accent, {BLUE});}}
  .kpi .lbl {{color: {MUTED}; font-size: .74rem; text-transform: uppercase; letter-spacing: .06em; font-weight: 600;}}
  .kpi .val {{color: {INK}; font-size: 1.7rem; font-weight: 750; margin-top: 4px; line-height: 1.1;}}
  .kpi .sub {{color: {INK2}; font-size: .8rem; margin-top: 4px;}}
  .insight {{background: linear-gradient(135deg, rgba(57,135,229,.10), rgba(144,133,233,.05));
             border: 1px solid rgba(57,135,229,.25); border-radius: 12px; padding: 12px 15px; margin: 4px 0 12px;
             color: {INK}; font-size: .92rem; line-height: 1.55;}}
  .insight b {{color: #86b6ef;}}
  .warn {{background: rgba(208,59,59,.10); border-color: rgba(208,59,59,.35);}} .warn b {{color: #ef7b7b;}}
  .good {{background: rgba(12,163,12,.10); border-color: rgba(12,163,12,.35);}} .good b {{color: #5fd35f;}}
  .sec {{margin: 22px 0 2px; font-size: 1.18rem; font-weight: 700; color: {INK};}}
  .secsub {{color: {MUTED}; font-size: .87rem; margin-bottom: 8px;}}
  .card {{background: {CARD}; border: 1px solid rgba(255,255,255,.08); border-radius: 14px; padding: 14px 16px; height: 100%;}}
  .card h4 {{margin: 0 0 6px; font-size: .8rem; letter-spacing: .08em; text-transform: uppercase; color: var(--accent, {BLUE});}}
  .card .head {{font-size: 1.02rem; font-weight: 700; color: {INK}; margin-bottom: 6px; line-height: 1.35;}}
  .card ul {{margin: 0; padding-left: 18px; color: {INK2}; font-size: .87rem; line-height: 1.5;}}
  .seg {{background: {CARD}; border: 1px solid rgba(255,255,255,.08); border-top: 4px solid var(--accent); border-radius: 12px;
         padding: 12px 14px; height: 100%;}}
  .seg .n {{font-weight: 700; color: {INK}; font-size: .95rem; line-height: 1.25; min-height: 2.4em;}}
  .seg .r {{font-size: 1.55rem; font-weight: 750; color: {INK}; margin-top: 4px;}}
  .seg .s {{color: {INK2}; font-size: .8rem;}}
  div[data-testid="stTabs"] button[role="tab"] {{font-size: .95rem; font-weight: 600; padding: 9px 12px;}}
  div[data-testid="stTabs"] button[aria-selected="true"] {{color: #86b6ef;}}
  section[data-testid="stSidebar"] {{border-right: 1px solid rgba(255,255,255,.06);}}
</style>
""", unsafe_allow_html=True)


# ============================================================== helpers
def style(fig, h=380, legend=True, title=None):
    fig.update_layout(
        height=h, template="plotly_dark", paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif", color=INK2, size=12),
        margin=dict(l=10, r=10, t=(52 if title else 22) + (30 if legend else 0), b=10), barcornerradius=4, bargap=0.25,
        title=dict(text=title, font=dict(size=15, color=INK), x=0.01, y=1, yanchor="top", yref="container",
                   pad=dict(t=12)) if title else None,
        hoverlabel=dict(bgcolor="#0d1117", bordercolor="#383835", font=dict(color=INK, size=12)),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0, xanchor="left", bgcolor="rgba(0,0,0,0)", font=dict(color=INK2)),
        showlegend=bool(legend), uniformtext=dict(minsize=9, mode="hide"),
    )
    if legend == "inside":
        fig.update_layout(legend=dict(orientation="v", x=0.98, xanchor="right", y=0.04, yanchor="bottom",
                                      bgcolor="rgba(13,17,23,.85)", bordercolor=AXIS, borderwidth=1),
                          margin=dict(t=52 if title else 22))
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=AXIS, linecolor=AXIS, tickfont=dict(color=MUTED), title_font=dict(color=INK2))
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=AXIS, linecolor=AXIS, tickfont=dict(color=MUTED), title_font=dict(color=INK2))
    return fig


def show(fig, h=380, legend=True, title=None, key=None, select=False):
    style(fig, h, legend, title)
    if select:
        return st.plotly_chart(fig, width="stretch", config=PCFG, key=key, on_select="rerun", selection_mode="points")
    st.plotly_chart(fig, width="stretch", config=PCFG, key=key)


def nomath(s):
    """Streamlit renders $...$ as LaTeX; use the HTML entity so dollar amounts stay literal."""
    return str(s).replace("$", "&#36;")


def md(s):
    """Escape $ for plain Markdown blocks."""
    return str(s).replace("$", "\\$")


def kpi(col, label, value, sub="", accent=BLUE):
    col.markdown(nomath(f'<div class="kpi" style="--accent:{accent}"><div class="lbl">{label}</div>'
                        f'<div class="val">{value}</div><div class="sub">{sub}</div></div>'), unsafe_allow_html=True)


def insight(html, kind=""):
    st.markdown(nomath(f'<div class="insight {kind}">💡 {html}</div>'), unsafe_allow_html=True)


def section(title, sub=""):
    st.markdown(nomath(f'<div class="sec">{title}</div>' + (f'<div class="secsub">{sub}</div>' if sub else "")),
                unsafe_allow_html=True)


def pct(x, d=1):
    return "–" if x is None or pd.isna(x) else f"{x * 100:.{d}f}%"


def money(v):
    if pd.isna(v):
        return "–"
    s, v = ("-" if v < 0 else ""), abs(v)
    return f"{s}${v:,.0f}" if v < 1e6 else f"{s}${v / 1e6:,.2f}M"


def times(a, b):
    return f"{a / b:.1f}×" if b else "∞×"


def avg_line(fig, y, text="Average"):
    fig.add_hline(y=y, line_dash="dot", line_color=YELLOW, line_width=1.5,
                  annotation_text=f"{text}: {pct(y)}", annotation_position="top left",
                  annotation_font=dict(color=YELLOW, size=11))


def rate_bars(t, x, title=None, overall=None, color=ACC, h=360, key=None, select=False, horizontal=False):
    """Acceptance-rate bar chart with 'rate% · loans/customers' labels inside every bar."""
    txt = [f"<b>{pct(r)}</b><br>{int(l):,}/{int(c):,}" for r, l, c in zip(t["rate"], t["loans"], t["customers"])]
    if horizontal:
        fig = go.Figure(go.Bar(y=t[x].astype(str), x=t["rate"], orientation="h", marker_color=color, text=txt,
                               textposition="auto", insidetextanchor="end",
                               hovertemplate="%{y}<br>Acceptance %{x:.1%}<extra></extra>"))
        fig.update_xaxes(tickformat=".0%", title="Acceptance rate")
        fig.update_yaxes(autorange="reversed")
    else:
        mx = max(t["rate"].max(), 1e-9)
        tpos = ["inside" if r >= 0.18 * mx else "outside" for r in t["rate"]]
        fig = go.Figure(go.Bar(x=t[x].astype(str), y=t["rate"], marker_color=color, text=txt, textposition=tpos, cliponaxis=False,
                               hovertemplate="%{x}<br>Acceptance %{y:.1%}<br>%{customdata[0]:,} loans of %{customdata[1]:,}<extra></extra>",
                               customdata=t[["loans", "customers"]].values))
        fig.update_yaxes(tickformat=".0%", title="Acceptance rate")
        fig.update_xaxes(type="category")
        if overall is not None:
            avg_line(fig, overall)
    return show(fig, h, False, title, key=key, select=select)


def heat(z, x, y, text, title, h=420, scale=SEQ, zfmt=".0%", zmid=None, cbar="Rate"):
    x = [wrap(v) if len(str(v)) > 11 and "<br>" not in str(v) else str(v) for v in x]
    fig = go.Figure(go.Heatmap(z=z, x=x, y=list(y), text=text, texttemplate="%{text}", colorscale=scale, zmid=zmid,
                               textfont=dict(size=11.5), hovertemplate="%{y} × %{x}<br>%{text}<extra></extra>", xgap=2, ygap=2,
                               colorbar=dict(title=cbar, tickformat=zfmt, thickness=12, outlinewidth=0)))
    fig.update_xaxes(type="category", side="bottom", showgrid=False, tickangle=0, tickfont=dict(color=INK2, size=11))
    fig.update_yaxes(type="category", autorange="reversed", showgrid=False)
    show(fig, h, False, title)


def wrap(name):
    """Two-line label for segment names on crowded axes."""
    return str(name).replace(" ", "<br>", 1)


def hierarchy(df, path):
    """ids/parents table for drill-down sunburst & treemap, with acceptance rate per node."""
    rows = [{"id": "All", "label": "All customers", "parent": "", "n": len(df), "loans": int(df[E.TARGET].sum())}]
    for d in range(1, len(path) + 1):
        g = df.groupby(path[:d], observed=True)[E.TARGET].agg(["size", "sum"]).reset_index()
        for _, r in g.iterrows():
            keys = [str(r[c]) for c in path[:d]]
            rows.append({"id": "All/" + "/".join(keys), "label": keys[-1],
                         "parent": "All" if d == 1 else "All/" + "/".join(keys[:-1]), "n": int(r["size"]), "loans": int(r["sum"])})
    h = pd.DataFrame(rows)
    h["rate"] = h["loans"] / h["n"]
    return h


def drill_chart(h, kind="sunburst", title=None, height=560):
    common = dict(ids=h["id"], labels=h["label"], parents=h["parent"], values=h["n"], branchvalues="total",
                  customdata=np.stack([h["rate"], h["loans"]], axis=1),
                  marker=dict(colors=h["rate"], colorscale=SEQ, cmin=0, cmax=max(h["rate"].max(), 0.01),
                              colorbar=dict(title="Accept %", tickformat=".0%", thickness=12, outlinewidth=0)),
                  hovertemplate="<b>%{label}</b><br>%{value:,} customers<br>%{customdata[1]:,} loans<br>"
                                "Acceptance %{customdata[0]:.1%}<extra></extra>",
                  texttemplate="<b>%{label}</b><br>%{customdata[0]:.0%} accept<br>%{value:,} cust.",
                  insidetextfont=dict(size=11))
    fig = go.Figure(go.Sunburst(**common, maxdepth=3) if kind == "sunburst" else go.Treemap(**common, maxdepth=3,
                    tiling=dict(pad=2), pathbar=dict(visible=True, textfont=dict(size=12))))
    show(fig, height, False, title)


# ============================================================== cached data
@st.cache_data(show_spinner="Loading customer data …")
def get_base():
    return E.load_data()


@st.cache_resource(show_spinner="Training classification models (about 5 seconds) …")
def get_models():
    return E.train_models(E.load_data())


@st.cache_data(show_spinner="Segmenting customers …")
def get_segments(k, method):
    return E.segment(E.load_data(), k, method)


@st.cache_data(show_spinner=False)
def get_kdiag():
    return E.k_diagnostics(E.load_data())


@st.cache_data(show_spinner="Mining association rules …")
def get_rules(df, sup, conf, profile, loan_only, max_len):
    return E.mine_rules(df, sup, conf, profile, loan_only, max_len)


@st.cache_data(show_spinner=False)
def get_quality():
    return E.data_quality()


base = get_base().copy()
models = get_models()
BEST = models["best"]
base["Propensity"] = models["scores"]
base["Tier"] = E.tier(base["Propensity"])
for col, lab in [("CD Account", "CD"), ("Securities Account", "Securities"), ("Online", "Online Banking"),
                 ("CreditCard", "Credit Card")]:
    base[lab] = pd.Categorical(np.where(base[col] == 1, f"Has {lab}", f"No {lab}"), [f"No {lab}", f"Has {lab}"], ordered=True)
base["No. of Products"] = pd.Categorical(base["Products Held"].astype(str) + " products",
                                         [f"{i} products" for i in range(5)], ordered=True)

DIMS = {"Income band": "Income Band", "Education": "Education Level", "Family size": "Family Size",
        "Card spend / month": "Card Spend Band", "Age band": "Age Band", "Mortgage": "Mortgage Band",
        "CD account": "CD", "Securities account": "Securities", "Online banking": "Online Banking",
        "Bank credit card": "Credit Card", "No. of products held": "No. of Products", "Region": "Region",
        "Segment (cluster)": "Segment"}

# ============================================================== sidebar
with st.sidebar:
    st.markdown("## 🎛️ Control panel")
    st.caption("Filters apply to every tab. Models and segments are built on all 5,000 customers, then the view is filtered.")
    with st.expander("🧩 Segmentation settings", expanded=False):
        method_lbl = st.radio("Clustering method", ["K-Means", "Gaussian Mixture (LCA-style)"],
                              help="K-Means: fast, hard boundaries. Gaussian Mixture: a latent-profile model, the "
                                   "continuous-data version of Latent Class Analysis; finds smaller, sharper niches.")
        k = st.slider("Number of segments", 3, 6, 4)
    with st.expander("💰 Campaign cost assumptions", expanded=False):
        cost = st.number_input("Cost per customer contacted ($)", 1.0, 1000.0, 10.0, 1.0,
                               help="All % results stay the same whatever you enter here; only $ figures change.")
        value = st.number_input("Net value of one new personal loan ($)", 0.0, 100000.0, 500.0, 50.0)

method = "K-Means" if method_lbl == "K-Means" else "Gaussian Mixture"
seg, prof_all = get_segments(k, method)
base["Segment"] = seg
SEGS = prof_all["Segment"].tolist()
SEG_COL = {s: CAT[i % len(CAT)] for i, s in enumerate(SEGS)}

with st.sidebar:
    st.markdown("### 🔎 Filters")
    f_seg = st.multiselect("Segment", SEGS, SEGS)
    f_reg = st.multiselect("Region", sorted(base["Region"].unique()), sorted(base["Region"].unique()))
    f_edu = st.multiselect("Education", E.EDU_ORDER, E.EDU_ORDER)
    f_fam = st.multiselect("Family size", E.FAMILY_ORDER, E.FAMILY_ORDER)
    f_inc = st.slider("Annual income ($000)", int(base.Income.min()), int(base.Income.max()),
                      (int(base.Income.min()), int(base.Income.max())))
    f_age = st.slider("Age", int(base.Age.min()), int(base.Age.max()), (int(base.Age.min()), int(base.Age.max())))
    c1, c2 = st.columns(2)
    f_onl = c1.selectbox("Online banking", ["All", "Yes", "No"])
    f_cc = c2.selectbox("Bank credit card", ["All", "Yes", "No"])

mask = (base["Segment"].isin(f_seg) & base["Region"].isin(f_reg) & base["Education Level"].isin(f_edu)
        & base["Family Size"].isin(f_fam) & base["Income"].between(*f_inc) & base["Age"].between(*f_age))
if f_onl != "All":
    mask &= base["Online"] == (f_onl == "Yes")
if f_cc != "All":
    mask &= base["CreditCard"] == (f_cc == "Yes")
f = base[mask].copy()

with st.sidebar:
    st.markdown(f"**{len(f):,}** of {len(base):,} customers selected")
    st.download_button("⬇️ Download filtered data (CSV)", f.drop(columns=["Tier"]).to_csv(index=False).encode(),
                       "universal_bank_filtered.csv", "text/csv", width="stretch")

st.markdown(f"""<div class="hero"><h1>🏦 Personal Loan Campaign Intelligence</h1>
<p>Universal Bank · {len(f):,} of {len(base):,} customers in view · Who said yes to the last personal-loan campaign,
why, who will say yes next, and how to get the same loans for half the spend (or less).</p></div>""", unsafe_allow_html=True)

if len(f) < 30 or f[E.TARGET].sum() == 0:
    st.warning("The current filters leave too few customers (or no loan takers) to analyse. Widen the filters in the sidebar.")
    st.stop()

# common numbers (filtered view)
N, LOANS = len(f), int(f[E.TARGET].sum())
RATE = LOANS / N
y_f, s_f = f[E.TARGET].values, f["Propensity"].values


def cond_rate(d, m):
    sub = d[m]
    return (sub[E.TARGET].mean() if len(sub) else np.nan), len(sub)


tabs = st.tabs(["🏠 Executive Summary", "📊 Descriptive", "🔍 Diagnostic", "🧩 Segments", "🔗 Product Affinity",
                "🤖 Predictive", "🎯 Prescriptive", "📖 Method & Data"])

# ====================================================================================== 0. EXECUTIVE
with tabs[0]:
    sim50 = E.simulate_budget(s_f, y_f, 0.5, cost, value)
    b95 = E.smallest_budget_for(s_f, y_f, 0.95)
    sim95 = E.simulate_budget(s_f, y_f, b95, cost, value)
    c = st.columns(5)
    kpi(c[0], "Customers in view", f"{N:,}", f"{pct(N / len(base))} of the customer base", BLUE)
    kpi(c[1], "Took the loan last time", pct(RATE), f"{LOANS:,} of {N:,} customers contacted", AQUA)
    kpi(c[2], "Contacts that did not convert", pct(1 - RATE), f"{N - LOANS:,} contacts paid for, no loan", RED)
    kpi(c[3], "Budget needed for 95% of loans", pct(b95, 0), f"of last campaign's spend (was 100%)", GREEN)
    kpi(c[4], "Conversion at half the budget", pct(sim50["new_rate"]), f"vs {pct(RATE)} last time · {times(sim50['new_rate'], RATE)} better", VIOLET)

    hi = f["Income"] >= 100
    r_hi, n_hi = cond_rate(f, hi)
    r_lo, n_lo = cond_rate(f, ~hi)
    r_cd, _ = cond_rate(f, f["CD Account"] == 1)
    r_nocd, _ = cond_rate(f, f["CD Account"] == 0)
    r_g, _ = cond_rate(f, f["Education"] >= 2)
    r_u, _ = cond_rate(f, f["Education"] == 1)
    share_hi_loans = f.loc[hi, E.TARGET].sum() / LOANS
    g_f = E.gains_table(s_f, y_f, 20)
    top10 = g_f.loc[g_f["pct_contacted"].round(2) == 0.10, "pct_loans"].iloc[0]
    top_seg = (f.groupby("Segment", observed=True)[E.TARGET].agg(["mean", "sum", "size"]).sort_values("mean", ascending=False))
    ts_name = top_seg.index[0]

    section("The story in four questions")
    cards = [
        ("Descriptive · What happened?", BLUE, f"Only {pct(RATE)} of customers took the loan.",
         [f"{N - LOANS:,} of {N:,} contacts ({pct(1 - RATE)}) produced nothing.",
          f"Customers earning $100k+ are {pct(n_hi / N, 0)} of the base but bring {pct(share_hi_loans, 0)} of the loans."]),
        ("Diagnostic · Why?", ORANGE, "Income, education, family size and a CD account drive the decision.",
         [f"$100k+ income: {pct(r_hi)} accept vs {pct(r_lo)} below $100k.",
          f"CD holders: {pct(r_cd)} vs {pct(r_nocd)}. Graduates+: {pct(r_g)} vs undergrads {pct(r_u)}."]),
        ("Predictive · Who next?", VIOLET, f"The {BEST} model ranks customers by their chance of saying yes.",
         [f"AUC {models['metrics'].loc[0, 'AUC']:.3f} on customers it never saw (1.0 = perfect).",
          f"The top 10% it picks contain {pct(top10, 0)} of all loan takers."]),
        ("Prescriptive · What to do?", GREEN, f"Contact the top {pct(b95, 0)} by score, not everyone.",
         [f"Keeps {pct(sim95['capture'], 0)} of loans at {pct(b95, 0)} of the spend; cost per loan {money(sim95['new_cpa'])} vs {money(sim95['base_cpa'])}.",
          f"Half budget (your goal): {pct(sim50['capture'], 0)} of loans, {pct(sim50['new_rate'])} conversion."]),
    ]
    cols = st.columns(4)
    for col, (h, a, head, bullets) in zip(cols, cards):
        col.markdown(nomath(f'<div class="card" style="--accent:{a}"><h4>{h}</h4><div class="head">{head}</div><ul>'
                            + "".join(f"<li>{b}</li>" for b in bullets) + "</ul></div>"), unsafe_allow_html=True)

    section("Last campaign vs the targeted plan", "Back-test on the last campaign: what if we had contacted only the customers the model ranks highest?")
    plans = [("Everyone", 1.0), ("50% goal", 0.5), (f"{pct(b95, 0)} plan", b95)]
    sims = [E.simulate_budget(s_f, y_f, b, cost, value) for _, b in plans]
    fig = make_subplots(1, 4, subplot_titles=["Customers contacted", "Loans won", "Conversion rate", "Cost per loan"],
                        horizontal_spacing=0.06)
    cols3 = [DEC, BLUE, GREEN]
    names = [p[0] for p in plans]
    for i, (vals, fmt) in enumerate([([s["k"] for s in sims], "{:,.0f}"), ([s["new_loans"] for s in sims], "{:,.0f}"),
                                     ([s["new_rate"] for s in sims], "pct"), ([s["new_cpa"] for s in sims], "money")]):
        txt = [pct(v) if fmt == "pct" else money(v) if fmt == "money" else fmt.format(v) for v in vals]
        fig.add_trace(go.Bar(x=names, y=vals, marker_color=cols3, text=txt, textposition="auto",
                             textfont=dict(size=13, color="white")), 1, i + 1)
        fig.update_yaxes(showticklabels=False, row=1, col=i + 1)
        fig.update_xaxes(tickangle=0, tickfont=dict(color=INK2, size=12), row=1, col=i + 1)
    fig.update_annotations(font=dict(color=INK, size=13))
    show(fig, 340, False)
    st.caption(f"Grey = last campaign (contacted everyone) · Blue = your goal (top 50% by score) · Green = recommended (top {pct(b95, 0)} by score, keeps 95% of loans)")
    insight(f"Last time we paid to contact <b>{N:,}</b> customers to win <b>{LOANS:,}</b> loans (<b>{money(sims[0]['new_cpa'])}</b> per loan). "
            f"Contacting only the top <b>{pct(b95, 0)}</b> by model score would have won <b>{sims[2]['new_loans']:,}</b> loans "
            f"({pct(sims[2]['capture'], 0)}) at <b>{money(sims[2]['new_cpa'])}</b> per loan. Your 50% goal is met with room to spare.", "good")

    c1, c2 = st.columns([1.15, 1])
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random contact (no model)",
                                 line=dict(color=MUTED, dash="dash")))
        fig.add_trace(go.Scatter(x=g_f["pct_contacted"], y=g_f["pct_loans"], mode="lines+markers", name=f"{BEST} ranking",
                                 line=dict(color=BLUE, width=3), fill="tonexty", fillcolor="rgba(57,135,229,.12)"))
        for p, ax, ay in ((0.1, 70, 60), (0.2, 80, 110), (0.5, 20, 60)):
            v = g_f.loc[g_f["pct_contacted"].round(2) == p, "pct_loans"].iloc[0]
            fig.add_annotation(x=p, y=v, text=f"Top {pct(p, 0)} → {pct(v, 0)} of loans", showarrow=True, arrowhead=2,
                               ax=ax, ay=ay, font=dict(color=INK, size=11), bgcolor="#0d1117", bordercolor=AXIS, arrowcolor=INK2)
        fig.update_xaxes(tickformat=".0%", title="Share of customers contacted (ranked by model)")
        fig.update_yaxes(tickformat=".0%", title="Share of all loan takers reached", range=[0, 1.05])
        show(fig, 380, True, "Gains curve: how fast the model finds loan takers")
    with c2:
        sg = f.groupby("Segment", observed=True)[E.TARGET].agg(customers="size", loans="sum").reset_index()
        sg["Share of customers"], sg["Share of loans"] = sg["customers"] / N, sg["loans"] / LOANS
        fig = go.Figure()
        for colname, colr in [("Share of customers", DEC), ("Share of loans", BLUE)]:
            fig.add_trace(go.Bar(y=sg["Segment"].astype(str), x=sg[colname], orientation="h", name=colname, marker_color=colr,
                                 text=[pct(v, 0) for v in sg[colname]], textposition="auto"))
        fig.update_layout(barmode="group")
        fig.update_xaxes(tickformat=".0%")
        fig.update_yaxes(autorange="reversed")
        show(fig, 380, True, "Where the loans come from (by segment)")
    r_ts = top_seg.loc[ts_name]
    insight(f"<b>{ts_name}</b> is {pct(r_ts['size'] / N, 0)} of customers but brings <b>{pct(r_ts['sum'] / LOANS, 0)}</b> of loans "
            f"(acceptance {pct(r_ts['mean'])}). That is where the budget should go first. See the 🎯 Prescriptive tab for the full playbook.")

# ====================================================================================== 1. DESCRIPTIVE
with tabs[1]:
    section("What happened in the last campaign?", "A plain count of who we contacted and who said yes. All numbers follow the sidebar filters.")
    c = st.columns(5)
    kpi(c[0], "Customers", f"{N:,}", "contacted with the loan offer", BLUE)
    kpi(c[1], "Accepted", f"{LOANS:,}", f"{pct(RATE)} acceptance rate", AQUA)
    kpi(c[2], "Median income", f"${f['Income'].median():,.0f}k", f"acceptors: ${f.loc[f[E.TARGET] == 1, 'Income'].median():,.0f}k", YELLOW)
    kpi(c[3], "Use online banking", pct(f["Online"].mean(), 0), "of customers", VIOLET)
    kpi(c[4], "Avg products held", f"{f['Products Held'].mean():.1f}", "of 4 (Securities, CD, Online, Card)", ORANGE)

    c1, c2 = st.columns([0.8, 1.2])
    with c1:
        fig = go.Figure(go.Pie(labels=["Accepted", "Declined"], values=[LOANS, N - LOANS], hole=0.58,
                               marker=dict(colors=[ACC, DEC], line=dict(color=SURFACE, width=2)),
                               texttemplate="<b>%{label}</b><br>%{percent:.1%}<br>%{value:,}", textposition="outside", sort=False))
        fig.add_annotation(text=f"<b>{pct(RATE)}</b><br>said yes", showarrow=False, font=dict(size=18, color=INK))
        show(fig, 360, False, "Loan offer outcome")
    with c2:
        items = ["Online", "CreditCard", "Has Mortgage", "Securities Account", "CD Account", E.TARGET]
        own = pd.DataFrame({"item": [E.PRODUCT_LABEL[i] for i in items], "share": [f[i].mean() for i in items],
                            "n": [int(f[i].sum()) for i in items]})
        fig = go.Figure(go.Bar(x=own["item"], y=own["share"], text=[f"<b>{pct(s)}</b><br>{n:,}" for s, n in zip(own["share"], own["n"])],
                               textposition="auto", marker_color=[VIOLET, ORANGE, YELLOW, AQUA, GREEN, BLUE]))
        fig.update_yaxes(tickformat=".0%", title="% of customers")
        show(fig, 360, False, "Product ownership: what share of customers holds each product")
    insight(f"Online banking is the most common product (<b>{pct(f['Online'].mean(), 0)}</b>), so digital channels reach most people cheaply. "
            f"CD accounts are rare (<b>{pct(f['CD Account'].mean())}</b>) but, as the Diagnostic tab shows, CD holders are the keenest loan takers.")

    section("Acceptance rate by any attribute, with drill-down",
            "Pick an attribute. <b>Click any bar</b> to drill into that group in the right-hand chart. The yellow dotted line is the average.")
    cc = st.columns([1, 1, 2])
    d1 = cc[0].selectbox("Break down by", list(DIMS), 0, key="d1")
    d2 = cc[1].selectbox("Drill down by", [d for d in DIMS if d != d1], 1, key="d2")
    t1 = E.rate_table(f, DIMS[d1])
    c1, c2 = st.columns(2)
    with c1:
        ev = rate_bars(t1, DIMS[d1], f"Acceptance rate by {d1.lower()}", RATE, key=f"drill_{d1}", select=True)
    pts = (ev or {}).get("selection", {}).get("points", []) if ev else []
    clicked = str(pts[0]["x"]) if pts else None
    groups = t1[DIMS[d1]].astype(str).tolist()
    pick = clicked if clicked in groups else t1.sort_values("rate", ascending=False)[DIMS[d1]].astype(str).iloc[0]
    sub = f[f[DIMS[d1]].astype(str) == pick]
    t2 = E.rate_table(sub, DIMS[d2])
    with c2:
        rate_bars(t2, DIMS[d2], f"Drill-down: <b>{pick}</b> split by {d2.lower()}", sub[E.TARGET].mean(), color=AQUA, key="drill2")
    best, worst = t1.sort_values("rate").iloc[-1], t1.sort_values("rate").iloc[0]
    bsub, wsub = t2.sort_values("rate").iloc[-1], t2.sort_values("rate").iloc[0]
    insight(f"By {d1.lower()}, <b>{best[DIMS[d1]]}</b> converts best at <b>{pct(best['rate'])}</b> ({times(best['rate'], RATE)} the average), "
            f"while <b>{worst[DIMS[d1]]}</b> converts at only <b>{pct(worst['rate'])}</b>. "
            f"Inside <b>{pick}</b>{' (you clicked)' if clicked else ' (top group, click another bar to change)'}, "
            f"<b>{bsub[DIMS[d2]]}</b> is best at {pct(bsub['rate'])} and <b>{wsub[DIMS[d2]]}</b> worst at {pct(wsub['rate'])}.")

    section("How acceptors differ from decliners", "Each bar shows what % of each group falls in a band, so the two groups compare like for like.")
    cc = st.columns([1, 3])
    dv = cc[0].selectbox("Compare on", ["Income band", "Card spend / month", "Education", "Family size", "Age band", "Mortgage",
                                        "No. of products held"], key="dv")
    dist = f.groupby(["Loan Status", DIMS[dv]], observed=True).size().reset_index(name="n")
    dist["share"] = dist["n"] / dist.groupby("Loan Status")["n"].transform("sum")
    fig = go.Figure()
    for status, colr in [("Accepted", ACC), ("Declined", DEC)]:
        dd = dist[dist["Loan Status"] == status]
        fig.add_trace(go.Bar(x=dd[DIMS[dv]].astype(str), y=dd["share"], name=status, marker_color=colr,
                             text=[pct(v, 0) for v in dd["share"]], textposition="auto"))
    fig.update_layout(barmode="group")
    fig.update_yaxes(tickformat=".0%", title="% of the group")
    fig.update_xaxes(type="category")
    show(fig, 360, True, f"{dv}: acceptors vs decliners")
    da = dist[dist["Loan Status"] == "Accepted"].set_index(DIMS[dv])["share"]
    dd_ = dist[dist["Loan Status"] == "Declined"].set_index(DIMS[dv])["share"]
    gap = (da - dd_.reindex(da.index).fillna(0)).sort_values()
    insight(f"The biggest over-representation among acceptors is <b>{gap.index[-1]}</b>: {pct(da[gap.index[-1]], 0)} of acceptors "
            f"vs {pct(dd_.get(gap.index[-1], 0), 0)} of decliners. The biggest under-representation is <b>{gap.index[0]}</b>.")

# ====================================================================================== 2. DIAGNOSTIC
with tabs[2]:
    section("Why did customers say yes?", "Combinations of two attributes. Each cell shows the acceptance rate and how many customers sit in it.")
    cc = st.columns([1, 1, 1, 1])
    hr = cc[0].selectbox("Rows", list(DIMS), 0, key="hr")
    hc = cc[1].selectbox("Columns", [d for d in DIMS if d != hr], 0, key="hc")
    hv = cc[2].radio("Colour by", ["Acceptance %", "Share of loans"], horizontal=True, key="hv")
    t = E.rate_table(f, [DIMS[hr], DIMS[hc]])
    zcol = "rate" if hv == "Acceptance %" else "share_loans"
    pz = t.pivot(index=DIMS[hr], columns=DIMS[hc], values=zcol)
    pn = t.pivot(index=DIMS[hr], columns=DIMS[hc], values="customers").reindex_like(pz)
    pl = t.pivot(index=DIMS[hr], columns=DIMS[hc], values="loans").reindex_like(pz)
    txt = [[("" if pd.isna(z) else f"<b>{pct(z, 0)}</b><br>{int(l)}/{int(n)}") for z, n, l in zip(zr, nr, lr)]
           for zr, nr, lr in zip(pz.values, pn.values, pl.values)]
    heat(pz.values, pz.columns.astype(str), pz.index.astype(str), txt,
         f"{hv} by {hr.lower()} × {hc.lower()} (label: rate · loans/customers)", 440)
    tt = t[t["customers"] >= 20].sort_values("rate")
    if len(tt):
        top_c, low_c = tt.iloc[-1], tt.iloc[0]
        insight(f"The hottest combination with at least 20 customers is <b>{top_c[DIMS[hr]]} + {top_c[DIMS[hc]]}</b>: "
                f"<b>{pct(top_c['rate'])}</b> accepted ({int(top_c['loans'])} of {int(top_c['customers'])}), "
                f"{times(top_c['rate'], RATE)} the average. The coldest is <b>{low_c[DIMS[hr]]} + {low_c[DIMS[hc]]}</b> at {pct(low_c['rate'])}. "
                f"Darker cells are where marketing money is wasted.")

    section("Stacking the signals: how acceptance climbs as we narrow the audience",
            "Waterfall chart. Pick conditions in the order to apply them. Each step keeps only customers who also meet the next condition.")
    CONDS = {"Income $100k+": lambda d: d["Income"] >= 100, "Graduate or higher": lambda d: d["Education"] >= 2,
             "Card spend $2.5k+/mo": lambda d: d["CCAvg"] >= 2.5, "Has CD account": lambda d: d["CD Account"] == 1,
             "Family of 3+": lambda d: d["Family"] >= 3, "Has mortgage": lambda d: d["Mortgage"] > 0,
             "Uses online banking": lambda d: d["Online"] == 1, "Has bank credit card": lambda d: d["CreditCard"] == 1,
             "Age 40+": lambda d: d["Age"] >= 40}
    steps = st.multiselect("Conditions (in order)", list(CONDS), ["Income $100k+", "Graduate or higher", "Card spend $2.5k+/mo", "Has CD account"],
                           max_selections=5, key="wf")
    d, prev = f, RATE
    xs, ys, ms, tx = ["All customers"], [RATE], ["absolute"], [f"<b>{pct(RATE)}</b><br>{N:,} cust."]
    for s in steps:
        d = d[CONDS[s](d)]
        if len(d) == 0:
            break
        r = d[E.TARGET].mean()
        xs.append(f"+ {s}")
        ys.append(r - prev)
        ms.append("relative")
        tx.append(f"{'+' if r >= prev else ''}{(r - prev) * 100:.1f} pts<br><b>{pct(r)}</b> · {len(d):,} cust.")
        prev = r
    xs.append("Final audience")
    ys.append(prev)
    ms.append("total")
    tx.append(f"<b>{pct(prev)}</b><br>{len(d):,} cust. · {int(d[E.TARGET].sum()):,} loans")
    fig = go.Figure(go.Waterfall(x=xs, y=ys, measure=ms, text=tx, textposition="outside",
                                 connector=dict(line=dict(color=AXIS, dash="dot")),
                                 increasing=dict(marker=dict(color=GREEN)), decreasing=dict(marker=dict(color=RED)),
                                 totals=dict(marker=dict(color=BLUE)), hoverinfo="skip"))
    fig.update_yaxes(tickformat=".0%", title="Acceptance rate", range=[0, min(1.2, max(prev, RATE) * 1.3 + 0.1)],
                     tickvals=[v / 10 for v in range(0, 11, 2)])
    show(fig, 430, False, "Acceptance rate waterfall")
    insight(f"Starting from <b>{pct(RATE)}</b>, applying {', then '.join(f'<b>{s}</b>' for s in steps) or 'no conditions'} lifts acceptance to "
            f"<b>{pct(prev)}</b> ({times(prev, RATE)}), but the audience shrinks to <b>{len(d):,}</b> customers holding "
            f"<b>{pct(d[E.TARGET].sum() / LOANS, 0)}</b> of all loans. Red steps mean that condition hurts. The trade-off is precision vs reach.")

    c1, c2 = st.columns([1.25, 1])
    with c1:
        section("Drill-down explorer", "Click a slice to zoom in, click the centre to zoom out. Colour = acceptance rate, size = number of customers.")
        path_lbl = st.multiselect("Hierarchy (in order, up to 3)", list(DIMS), ["Income band", "Education", "Family size"],
                                  max_selections=3, key="sb")
        if path_lbl:
            drill_chart(hierarchy(f, [DIMS[p] for p in path_lbl]), "sunburst", None, 560)
    with c2:
        section("Acceptors vs decliners at a glance", "Average of each attribute for both groups.")
        acc, dec = f[f[E.TARGET] == 1], f[f[E.TARGET] == 0]
        rows = [("Income", "Annual income", lambda v: f"${v:,.0f}k"), ("CCAvg", "Card spend / month", lambda v: f"${v:,.2f}k"),
                ("CD Account", "Has CD account", pct), ("Education", "Education (1-3)", lambda v: f"{v:.2f}"),
                ("Family", "Family size", lambda v: f"{v:.2f}"), ("Mortgage", "Mortgage", lambda v: f"${v:,.0f}k"),
                ("Securities Account", "Has securities", pct), ("Online", "Online banking", pct),
                ("CreditCard", "Bank credit card", pct), ("Age", "Age", lambda v: f"{v:.1f}")]
        comp = pd.DataFrame([{"label": lab, "ratio": (acc[c].mean() / dec[c].mean()) if dec[c].mean() else np.nan,
                              "txt": f"{times(acc[c].mean(), dec[c].mean())}  ({fm(acc[c].mean())} vs {fm(dec[c].mean())})"}
                             for c, lab, fm in rows]).sort_values("ratio")
        fig = go.Figure(go.Bar(y=comp["label"], x=comp["ratio"], orientation="h", text=comp["txt"], textposition="auto",
                               marker_color=[BLUE if v > 1.05 else RED if v < 0.95 else DEC for v in comp["ratio"]]))
        fig.add_vline(x=1, line_dash="dot", line_color=YELLOW, annotation_text="1× = same as decliners", annotation_font_color=YELLOW,
                      annotation_position="bottom right")
        fig.update_xaxes(title="Acceptors' average ÷ decliners' average")
        show(fig, 560, False)
    top3 = comp.sort_values("ratio", ascending=False).head(3)
    insight("Compared with decliners, acceptors have " + ", ".join(f"<b>{r.label}</b> {r.txt.split('  ')[0]}" for r in top3.itertuples())
            + " higher. Age, online banking and holding a bank credit card barely differ, so they make poor targeting levers on their own.")

# ====================================================================================== 3. SEGMENTS
with tabs[3]:
    section(f"Customer segments ({method}, k = {k})",
            "Clustering groups customers who look alike on income, card spend, family size, education and mortgage. "
            "Products and the loan outcome are left out on purpose, so we can then ask what each segment owns and buys.")
    fs = f.groupby("Segment", observed=True)
    prof = fs.agg(customers=(E.TARGET, "size"), loans=(E.TARGET, "sum"), rate=(E.TARGET, "mean"), Income=("Income", "mean"),
                  CCAvg=("CCAvg", "mean"), Family=("Family", "mean"), Education=("Education", "mean"),
                  Mortgage=("Mortgage", "mean"), Age=("Age", "mean")).reset_index()
    prof["share_c"], prof["share_l"] = prof["customers"] / N, prof["loans"] / LOANS
    cols = st.columns(len(prof))
    for col, r in zip(cols, prof.itertuples()):
        col.markdown(f'<div class="seg" style="--accent:{SEG_COL[r.Segment]}"><div class="n">{r.Segment}</div>'
                     f'<div class="r">{pct(r.rate)}</div><div class="s">acceptance rate</div>'
                     f'<div class="s" style="margin-top:6px">{r.customers:,} customers ({pct(r.share_c, 0)})<br>'
                     f'{r.loans:,} loans ({pct(r.share_l, 0)} of all)</div></div>', unsafe_allow_html=True)

    with st.expander("How many segments? Elbow and silhouette check (K-Means)", expanded=False):
        kd = get_kdiag()
        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure(go.Scatter(x=kd["k"], y=kd["inertia"], mode="lines+markers+text", text=[f"{v:,.0f}" for v in kd["inertia"]],
                                       textposition="top right", line=dict(color=BLUE, width=3)))
            fig.add_vline(x=k, line_dash="dot", line_color=YELLOW, annotation_text=f"chosen k={k}", annotation_font_color=YELLOW)
            fig.update_xaxes(title="Number of segments (k)", dtick=1)
            fig.update_yaxes(title="Within-segment spread (lower = tighter)")
            show(fig, 320, False, "Elbow: where extra segments stop helping much")
        with c2:
            fig = go.Figure(go.Bar(x=kd["k"], y=kd["silhouette"], text=[f"{v:.3f}" for v in kd["silhouette"]], textposition="auto",
                                   marker_color=[YELLOW if kk == k else BLUE for kk in kd["k"]]))
            fig.update_xaxes(title="Number of segments (k)", dtick=1)
            fig.update_yaxes(title="Silhouette (higher = better separated)")
            show(fig, 320, False, "Silhouette score")
        st.markdown("The elbow bends around **k = 3-4** and silhouette scores are close for k = 3 to 8, so we use **4** "
                    "segments: the fewest that still separate homeowners, families and singles in the mass market. "
                    "Gaussian Mixture is a latent-profile model (the continuous-data form of Latent Class Analysis). "
                    "It tends to find a small, very high-converting niche.")

    c1, c2 = st.columns([1, 1])
    with c1:
        fig = go.Figure()
        for colname, lab, colr in [("share_c", "Share of customers", DEC), ("share_l", "Share of loans", BLUE)]:
            fig.add_trace(go.Bar(x=prof["Segment"].map(wrap), y=prof[colname], name=lab, marker_color=colr,
                                 text=[pct(v, 0) for v in prof[colname]], textposition="auto"))
        fig.add_trace(go.Scatter(x=prof["Segment"].map(wrap), y=prof["rate"], name="Acceptance rate", mode="lines+markers+text",
                                 text=[pct(v) for v in prof["rate"]], textposition="middle right", yaxis="y2",
                                 line=dict(color=YELLOW, width=2), textfont=dict(color=YELLOW)))
        fig.update_layout(barmode="group", yaxis2=dict(overlaying="y", side="right", tickformat=".0%", showgrid=False,
                                                       range=[0, max(prof["rate"].max() * 1.35, 0.05)], title="Acceptance rate"))
        fig.update_yaxes(tickformat=".0%", title="Share of total")
        fig.update_xaxes(tickangle=0)
        show(fig, 420, True, "Segment size vs loans delivered")
    with c2:
        feats = [("Income", "Income ($k)", "{:,.0f}"), ("CCAvg", "Card spend ($k/mo)", "{:,.2f}"), ("Family", "Family size", "{:.1f}"),
                 ("Education", "Education (1-3)", "{:.1f}"), ("Mortgage", "Mortgage ($k)", "{:,.0f}"), ("Age", "Age", "{:.0f}")]
        overall = f[[c for c, _, _ in feats]].mean()
        z = np.array([[r[c] / overall[c] * 100 if overall[c] else 100 for _, r in prof.iterrows()] for c, _, _ in feats])
        txt = [[f"<b>{fm.format(r[c])}</b><br>idx {z[j, i]:.0f}" for i, (_, r) in enumerate(prof.iterrows())] for j, (c, _, fm) in enumerate(feats)]
        heat(np.log(np.clip(z, 1, None) / 100), prof["Segment"].map(wrap), [lab for _, lab, _ in feats], txt,
             "Segment profile (value · index, 100 = average)", 420, DIV, ".1f", 0, "log index")
    top = prof.sort_values("rate", ascending=False).iloc[0]
    insight(f"<b>{top.Segment}</b> is only <b>{pct(top.share_c, 0)}</b> of customers but delivers <b>{pct(top.share_l, 0)}</b> of loans, "
            f"converting at <b>{pct(top.rate)}</b> ({times(top.rate, RATE)} average). Their average income is ${top.Income:,.0f}k and card spend "
            f"${top.CCAvg:,.1f}k a month. The other segments convert at {', '.join(pct(v) for v in prof.sort_values('rate', ascending=False)['rate'].iloc[1:])}: "
            f"mass mailing them is the main source of waste.")

    section("What to sell to each segment", "Left: % of each segment holding each product (idx = index vs the average customer, 100 = average). "
            "Right: loan acceptance among holders of each product inside the segment, the bridge to association rules.")
    items = ["Online", "CreditCard", "Securities Account", "CD Account", "Has Mortgage", E.TARGET]
    labs = [E.PRODUCT_LABEL[i] for i in items]
    pen = fs[items].mean()
    ov = f[items].mean()
    txt = [[f"<b>{pct(pen.loc[s, i], 0)}</b><br>idx {pen.loc[s, i] / ov[i] * 100 if ov[i] else 0:.0f}" for s in pen.index] for i in items]
    c1, c2 = st.columns(2)
    with c1:
        heat(pen.values.T, [wrap(s) for s in pen.index], labs, txt, "Product penetration by segment (% holding · index)", 420)
    with c2:
        rows, txt2 = [], []
        for s in pen.index:
            ss = f[f["Segment"] == s]
            rr, tt_ = [], []
            for i in items[:-1]:
                h = ss[ss[i] == 1]
                r = h[E.TARGET].mean() if len(h) >= 10 else np.nan
                rr.append(r)
                tt_.append("n<10" if pd.isna(r) else f"<b>{pct(r, 0)}</b><br>{int(h[E.TARGET].sum())}/{len(h)}")
            rows.append(rr)
            txt2.append(tt_)
        heat(np.array(rows, dtype=float).T, [wrap(s) for s in pen.index], labs[:-1], [list(r) for r in zip(*txt2)],
             "Loan acceptance among product holders (rate · loans/holders)", 420)
    best_cell = max(((s, labs[j], rows[i][j]) for i, s in enumerate(pen.index) for j in range(len(labs) - 1)
                     if not pd.isna(rows[i][j])), key=lambda x: x[2])
    insight(f"Across all segments, <b>CD account</b> holders are the most loan-ready. The single best pocket is "
            f"<b>{best_cell[1]} holders in {best_cell[0]}</b> at <b>{pct(best_cell[2])}</b> acceptance. Practical use: offer the "
            f"personal loan <b>as a bundle with a CD</b> to affluent customers, and use the CD base as a warm list.")

    section("Segment drill-down", "Treemap: click a box to drill down, use the bar at the top to go back. Colour = acceptance rate.")
    tm_path = st.multiselect("Drill path", list(DIMS), ["Segment (cluster)", "Education", "Family size"], max_selections=3, key="tm")
    if tm_path:
        drill_chart(hierarchy(f, [DIMS[p] for p in tm_path]), "treemap", None, 560)
    if "Affluent" in str(top.Segment):
        at = f[f["Segment"] == top.Segment]
        u1 = at[(at["Education"] == 1) & (at["Family"] <= 2)]
        g3 = at[(at["Education"] >= 2)]
        if len(u1) and len(g3):
            insight(f"Even inside <b>{top.Segment}</b>, behaviour splits sharply: graduates and above accept at <b>{pct(g3[E.TARGET].mean())}</b>, "
                    f"while undergrads in families of 1-2 accept at only <b>{pct(u1[E.TARGET].mean())}</b> ({len(u1):,} customers). "
                    f"Segment first, then use the model to pick individuals inside the segment.", "warn")

# ====================================================================================== 4. ASSOCIATION RULES
with tabs[4]:
    section("Product affinity: association rule mining (Apriori)",
            "Finds combinations that show up together, for example customers with a CD account and $100k+ income who also took a loan.")
    insight("<b>How to read a rule</b>, e.g. <i>IF CD Account + Income $100k+ THEN Personal Loan</i>. "
            "<b>Support</b> = % of all customers with the whole combination (how common). "
            "<b>Confidence</b> = of customers with the IF part, % who also have the THEN part (the conversion rate). "
            "<b>Lift</b> = confidence ÷ the normal rate of the THEN part. Lift 3 means three times more likely than average; lift above 1 is a real signal.")
    cc = st.columns([1.4, 1, 1, 1, 1.2])
    focus = cc[0].radio("Rule focus", ["Who takes a personal loan", "Cross-sell between products"], key="focus")
    loan_only = focus.startswith("Who")
    sup = cc[1].slider("Min support", 0.005, 0.10, 0.01, 0.005, format="%.3f", key="sup",
                       help="Minimum share of customers a combination must cover")
    conf = cc[2].slider("Min confidence", 0.05, 0.95, 0.30 if loan_only else 0.20, 0.05, key="conf")
    mlen = cc[3].slider("Max items in rule", 2, 4, 4, key="mlen")
    seg_pick = cc[4].selectbox("Segment", ["All segments"] + SEGS, key="armseg")
    prof_items = st.checkbox("Include profile flags (Income \\$100k+, Graduate+, Family 3+, Card spend \\$2.5k+) as items", True, key="pi")
    adf = f if seg_pick == "All segments" else f[f["Segment"] == seg_pick]
    rule_cols = E.PRODUCTS + ["Has Mortgage", E.TARGET, "Income", "Education", "Family", "CCAvg"]
    rules = get_rules(adf[rule_cols], sup, conf, prof_items, loan_only, mlen) if len(adf) >= 30 else pd.DataFrame()
    if rules.empty:
        st.info("No rules meet these thresholds for this selection. Lower the minimum support or confidence.")
    else:
        base_then = adf[E.TARGET].mean() if loan_only else None
        c = st.columns(4)
        kpi(c[0], "Rules found", f"{len(rules):,}", f"in {len(adf):,} customers", BLUE)
        kpi(c[1], "Highest lift", f"{rules['lift'].max():.1f}×", "vs. average likelihood", GREEN)
        kpi(c[2], "Highest confidence", pct(rules["confidence"].max(), 0), "conversion of the best combo", AQUA)
        kpi(c[3], "Baseline loan rate" if loan_only else "Avg. lift", pct(base_then) if loan_only else f"{rules['lift'].mean():.1f}×",
            "for this selection" if loan_only else "across rules", YELLOW)

        top_r = rules.sort_values(["confidence", "lift"], ascending=False).head(15).iloc[::-1]
        c1, c2 = st.columns([1.25, 1])
        with c1:
            lbl = [f"{r.IF}  →  {r.THEN}" for r in top_r.itertuples()]
            fig = go.Figure(go.Bar(y=lbl, x=top_r["confidence"], orientation="h",
                                   marker=dict(color=top_r["lift"], colorscale=SEQ, cmin=1,
                                               colorbar=dict(title="Lift", thickness=12, outlinewidth=0)),
                                   text=[f"<b>{pct(r.confidence, 0)}</b> · lift {r.lift:.1f}× · {r.customers:,} cust." for r in top_r.itertuples()],
                                   textposition="auto", insidetextanchor="end"))
            fig.update_xaxes(tickformat=".0%", title="Confidence", range=[0, 1.02])
            fig.update_yaxes(tickfont=dict(size=10.5, color=INK2))
            show(fig, 560, False, "Top 15 rules by confidence")
        with c2:
            fig = go.Figure(go.Scatter(x=rules["support"], y=rules["confidence"], mode="markers",
                                       marker=dict(size=np.clip(rules["lift"] * 3.5, 6, 40), color=rules["lift"], colorscale=SEQ,
                                                   line=dict(color=SURFACE, width=1), showscale=False, opacity=0.85),
                                       customdata=np.stack([rules["IF"], rules["THEN"], rules["lift"], rules["customers"]], axis=1),
                                       hovertemplate="IF %{customdata[0]}<br>THEN %{customdata[1]}<br>Support %{x:.1%} · Confidence %{y:.0%}"
                                                     "<br>Lift %{customdata[2]:.1f}× · %{customdata[3]} customers<extra></extra>"))
            for r in rules.head(4).itertuples():
                fig.add_annotation(x=r.support, y=r.confidence, text=f"{r.lift:.1f}×", showarrow=False, yshift=14,
                                   font=dict(color=INK, size=11))
            fig.update_xaxes(tickformat=".1%", title="Support (how common)")
            fig.update_yaxes(tickformat=".0%", title="Confidence (conversion)")
            show(fig, 560, False, "All rules: common vs reliable (bubble size = lift)")
        r0 = rules.sort_values(["lift", "confidence"], ascending=False).iloc[0]
        insight(f"Strongest signal: customers with <b>{r0.IF}</b> → <b>{pct(r0.confidence, 0)}</b> have <b>{r0.THEN}</b>, "
                f"<b>{r0.lift:.1f}×</b> the normal rate ({r0.customers:,} customers, {pct(r0.support)} of the selection). "
                + ("Use rules like this as ready-made target lists: they are simple enough for a branch manager or CRM filter."
                   if loan_only else "Use these pairs for bundles and next-best-offer prompts in online banking."), "good")

    section("Product-to-product affinity", "Calculated on the current selection. Left: lift (1 = no relationship). Right: chance of the column product given the row product.")
    lift, cnd = E.lift_matrix(adf)
    lv, cv = lift.to_numpy(dtype=float, copy=True), cnd.to_numpy(dtype=float, copy=True)
    np.fill_diagonal(lv, np.nan)
    np.fill_diagonal(cv, np.nan)
    c1, c2 = st.columns(2)
    with c1:
        txt = [["" if pd.isna(v) else f"<b>{v:.2f}×</b>" for v in r] for r in lv]
        with np.errstate(divide="ignore"):
            lz = np.log(np.where(lv > 0, lv, np.nan))
        heat(lz, lift.columns, lift.index, txt, "Lift between products", 400, DIV, ".1f", 0, "log lift")
    with c2:
        txt = [["" if pd.isna(v) else f"<b>{pct(v, 0)}</b>" for v in r] for r in cv]
        heat(cv, cnd.columns, cnd.index, txt, "P(has column product | has row product)", 400)
    insight(f"A <b>CD account</b> is the hub product: CD holders are <b>{lift.loc['CD Account', 'Personal Loan']:.1f}×</b> more likely to have a personal loan "
            f"and <b>{lift.loc['CD Account', 'Securities']:.1f}×</b> more likely to hold securities. Online banking and the credit card are "
            f"almost neutral (lift ≈ 1), so on their own they say little about loan interest, but they are cheap channels to deliver the offer.")

    if not rules.empty:
        with st.expander(f"📋 All {len(rules):,} rules in plain English (download)", expanded=False):
            tbl = rules.copy()
            tbl["In plain English"] = [f"Of customers with {r.IF}, {pct(r.confidence, 0)} have {r.THEN} ({r.lift:.1f}× normal)"
                                       for r in tbl.itertuples()]
            tbl = tbl[["In plain English", "IF", "THEN", "support", "confidence", "lift", "customers"]]
            st.dataframe(tbl, width="stretch", hide_index=True, height=380,
                         column_config={"support": st.column_config.NumberColumn("Support", format="percent"),
                                        "confidence": st.column_config.ProgressColumn("Confidence", format="percent", min_value=0, max_value=1),
                                        "lift": st.column_config.NumberColumn("Lift", format="%.2f×")})
            st.download_button("⬇️ Download rules (CSV)", tbl.to_csv(index=False).encode(), "association_rules.csv", "text/csv")

# ====================================================================================== 5. PREDICTIVE
with tabs[5]:
    section("Predicting who will accept: classification",
            f"Three models learn from last campaign's yes/no outcome (the <i>Personal Loan</i> column). They are scored on a "
            f"held-out {models['test_size']:,} customers they never saw. Experience (≈ Age, r = 0.99) and ZIP code are left out.")
    met = models["metrics"]
    c1, c2 = st.columns([1.3, 1])
    with c1:
        fig = go.Figure()
        for i, r in met.iterrows():
            ms_ = ["AUC", "Accuracy", "Precision", "Recall", "F1"]
            fig.add_trace(go.Bar(x=ms_, y=[r[m] for m in ms_], name=r["Model"], marker_color=[BLUE, AQUA, ORANGE][i],
                                 text=[pct(r[m]) for m in ms_], textposition="auto", textangle=-90, textfont=dict(size=11)))
        fig.update_layout(barmode="group")
        fig.update_yaxes(tickformat=".0%", range=[0.5, 1.02], title="Score on unseen customers")
        show(fig, 400, True, "Model comparison (higher is better)")
    with c2:
        fig = go.Figure(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=MUTED, dash="dash"), name="Coin flip (AUC 0.50)"))
        for i, r in met.iterrows():
            fpr, tpr = models["roc"][r["Model"]]
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{r['Model']} (AUC {r['AUC']:.3f})",
                                     line=dict(color=[BLUE, AQUA, ORANGE][i], width=3 if i == 0 else 2)))
        fig.update_xaxes(title="False alarms (decliners flagged)", tickformat=".0%")
        fig.update_yaxes(title="Loan takers caught", tickformat=".0%")
        show(fig, 400, "inside", "ROC curve")
    b = met.iloc[0]
    insight(f"<b>{b['Model']}</b> wins: of the customers it flags as 'will accept', <b>{pct(b['Precision'], 0)}</b> really did (precision), and it "
            f"catches <b>{pct(b['Recall'], 0)}</b> of all real acceptors (recall). Logistic Regression is weaker ({pct(met.set_index('Model').loc['Logistic Regression', 'Recall'], 0)} recall) "
            f"because the pattern is not a straight line: income only matters <i>combined</i> with education or family size.")

    c1, c2 = st.columns([1, 1.25])
    with c1:
        mpick = st.selectbox("Confusion matrix for", met["Model"].tolist(), key="cm")
        cm = models["cm"][mpick]
        tot = cm.sum()
        lab = [["Correctly ignored", "False alarm"], ["Missed acceptor", "Correctly found"]]
        txt = [[f"<b>{lab[i][j]}</b><br>{cm[i, j]:,}<br>{pct(cm[i, j] / cm[i].sum())} of row" for j in range(2)] for i in range(2)]
        fig = go.Figure(go.Heatmap(z=cm / cm.sum(axis=1, keepdims=True), x=["Predicted: decline", "Predicted: accept"],
                                   y=["Actually declined", "Actually accepted"], text=txt, texttemplate="%{text}",
                                   colorscale=SEQ, showscale=False, xgap=3, ygap=3))
        fig.update_yaxes(autorange="reversed")
        show(fig, 360, False)
        insight(f"Out of <b>{cm[1].sum()}</b> real acceptors in the test set, {mpick} found <b>{cm[1, 1]}</b> and missed {cm[1, 0]}. "
                f"It raised only <b>{cm[0, 1]}</b> false alarms among {cm[0].sum():,} decliners.")
    with c2:
        imp = models["importance"].iloc[::-1]
        fig = go.Figure(go.Bar(y=imp["label"], x=imp["importance"], orientation="h", text=[pct(v) for v in imp["importance"]],
                               textposition="auto", marker=dict(color=imp["importance"], colorscale=SEQ)))
        fig.update_xaxes(tickformat=".0%", title="Share of the model's decision power")
        show(fig, 430, False, "What drives the prediction (Random Forest feature importance)")
        i3 = models["importance"].head(4)
        insight(f"<b>{i3.iloc[0].label}</b> ({pct(i3.iloc[0].importance, 0)}), <b>{i3.iloc[1].label.lower()}</b> ({pct(i3.iloc[1].importance, 0)}), "
                f"<b>{i3.iloc[2].label.lower()}</b> and <b>{i3.iloc[3].label.lower()}</b> carry {pct(i3.importance.sum(), 0)} of the model's decisions. "
                f"Product flags such as online banking or credit card add almost nothing.")

    section("The model's rules in plain English (Decision Tree, depth 4)",
            "Each row is one group of customers the tree carves out, with what actually happened to them. Sorted from hottest to coldest.")
    tr = models["tree_rules"].copy()
    tr["Verdict"] = np.select([tr["rate"] >= 0.5, tr["rate"] >= 0.15, tr["rate"] > 0], ["🔥 Target", "🎯 Test", "🌱 Low"], "⛔ Skip")
    tr["If the customer has…"] = tr["rule"].str.replace(" AND ", " · ")
    st.dataframe(tr[["Verdict", "If the customer has…", "customers", "loans", "rate", "share_loans"]], width="stretch", hide_index=True,
                 column_config={"customers": st.column_config.NumberColumn("Customers", format="%d"),
                                "loans": st.column_config.NumberColumn("Loans", format="%d"),
                                "rate": st.column_config.ProgressColumn("Acceptance", format="percent", min_value=0, max_value=1),
                                "share_loans": st.column_config.ProgressColumn("Share of all loans", format="percent", min_value=0, max_value=1)})
    hot = tr[tr["rate"] >= 0.5]
    skip = tr[tr["rate"] == 0]
    insight(f"Just <b>{len(hot)}</b> simple rules cover <b>{pct(hot['loans'].sum() / tr['loans'].sum(), 0)}</b> of all loans while holding only "
            f"<b>{pct(hot['customers'].sum() / tr['customers'].sum(), 0)}</b> of customers. At the other end, <b>{skip['customers'].sum():,}</b> customers "
            f"({pct(skip['customers'].sum() / tr['customers'].sum(), 0)}) sit in groups where <b>nobody</b> took the loan: stop paying to contact them.", "good")

    section("Lift by score decile", "Customers ranked by model score and cut into 10 equal groups. Bars = acceptance rate per group, line = cumulative share of loans.")
    dec_ = pd.DataFrame({"s": s_f, "y": y_f})
    dec_["Decile"] = pd.qcut(dec_["s"].rank(method="first", ascending=False), 10, labels=[f"D{i}" for i in range(1, 11)])
    dt = dec_.groupby("Decile", observed=True)["y"].agg(["mean", "sum", "size"]).reset_index()
    dt["cum"] = dt["sum"].cumsum() / dt["sum"].sum()
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=dt["Decile"], y=dt["mean"], name="Acceptance rate in decile", marker_color=BLUE,
                         text=[f"<b>{pct(m)}</b><br>{int(s)}/{int(n)}" for m, s, n in zip(dt["mean"], dt["sum"], dt["size"])], textposition="auto"))
    fig.add_trace(go.Scatter(x=dt["Decile"], y=dt["cum"], name="Cumulative % of loans", mode="lines+markers+text",
                             text=[pct(v, 0) for v in dt["cum"]], textposition="top center", line=dict(color=YELLOW, width=2),
                             textfont=dict(color=YELLOW)), secondary_y=True)
    fig.update_yaxes(tickformat=".0%", title="Acceptance rate", secondary_y=False)
    fig.update_yaxes(tickformat=".0%", range=[0, 1.15], tickvals=[0, 0.25, 0.5, 0.75, 1], showgrid=False,
                     title="Cumulative loans", secondary_y=True)
    fig.update_xaxes(title="Score decile (D1 = highest scores)")
    show(fig, 400, True)
    insight(f"Decile 1 converts at <b>{pct(dt['mean'].iloc[0])}</b> vs {pct(RATE)} on average (<b>{times(dt['mean'].iloc[0], RATE)}</b> lift). "
            f"The top 2 deciles hold <b>{pct(dt['cum'].iloc[1], 0)}</b> of all loans; deciles 4-10 together hold {pct(1 - dt['cum'].iloc[2], 1)}.")

    section("What-if scorer: score any customer", "Change the profile; all three models re-score live.")
    with st.form("whatif"):
        cc = st.columns(5)
        w_inc = cc[0].slider("Income ($000)", 8, 224, 120)
        w_cc = cc[1].slider("Card spend ($000/mo)", 0.0, 10.0, 2.5, 0.1)
        w_fam = cc[2].select_slider("Family size", [1, 2, 3, 4], 3)
        w_edu = cc[3].select_slider("Education", [1, 2, 3], 2, format_func=E.EDU_LABEL.get)
        w_age = cc[4].slider("Age", 23, 67, 40)
        cc = st.columns(5)
        w_mort = cc[0].slider("Mortgage ($000)", 0, 635, 0, 5)
        w_cd = cc[1].checkbox("Has CD account")
        w_sec = cc[2].checkbox("Has securities account")
        w_onl = cc[3].checkbox("Uses online banking", True)
        w_crd = cc[4].checkbox("Has bank credit card")
        st.form_submit_button("Score this customer", type="primary")
    row = pd.DataFrame([{"Income": w_inc, "CCAvg": w_cc, "Family": w_fam, "Education": w_edu, "CD Account": int(w_cd),
                         "Mortgage": w_mort, "Age": w_age, "Securities Account": int(w_sec), "Online": int(w_onl),
                         "CreditCard": int(w_crd)}])[E.MODEL_FEATURES]
    probs = {n: float(m.predict_proba(row)[0, 1]) for n, m in models["full"].items()}
    c1, c2 = st.columns([1, 1.3])
    with c1:
        p = probs[BEST]
        fig = go.Figure(go.Indicator(mode="gauge+number", value=p * 100, number=dict(suffix="%", font=dict(size=44, color=INK)),
                                     title=dict(text=f"{BEST}: chance of accepting", font=dict(size=14, color=INK2)),
                                     gauge=dict(axis=dict(range=[0, 100], ticksuffix="%"), bar=dict(color=BLUE),
                                                steps=[dict(range=[0, 5], color="#2b2b29"), dict(range=[5, 20], color="#3a3320"),
                                                       dict(range=[20, 50], color="#4a3220"), dict(range=[50, 100], color="#4a2224")],
                                                threshold=dict(line=dict(color=YELLOW, width=3), value=RATE * 100))))
        show(fig, 300, False)
    with c2:
        fig = go.Figure(go.Bar(x=list(probs), y=list(probs.values()), text=[pct(v) for v in probs.values()], textposition="auto",
                               marker_color=[BLUE if n == BEST else DEC for n in probs]))
        avg_line(fig, RATE, "Average customer")
        fig.update_yaxes(tickformat=".0%", range=[0, 1.05], title="Predicted chance")
        show(fig, 300, False, "All three models")
    tname = E.tier([p])[0]
    insight(f"This customer is <b>{tname}</b>: {pct(p)} chance of accepting, <b>{times(p, RATE)}</b> the average customer. "
            + ("Put them on the priority list." if p >= 0.5 else "Include them if budget allows." if p >= 0.2
               else "Low priority: low-cost digital nudge only." if p >= 0.05 else "Do not spend on contacting them."))

# ====================================================================================== 6. PRESCRIPTIVE
with tabs[6]:
    section("Action plan: same loans, a fraction of the spend",
            "Move the budget slider. The plan contacts customers in order of model score until the budget runs out. Back-tested on the last campaign.")
    cc = st.columns([2, 1])
    bshare = cc[0].slider("Budget as % of the last campaign", 5, 100, 50, 5, format="%d%%", key="bud") / 100
    sim = E.simulate_budget(s_f, y_f, bshare, cost, value)
    cc[1].markdown(nomath(f"<div class='secsub' style='margin-top:28px'>Cost per contact <b>{money(cost)}</b>, value per loan "
                          f"<b>{money(value)}</b> (change in sidebar)</div>"), unsafe_allow_html=True)
    c = st.columns(5)
    kpi(c[0], "Campaign spend", money(sim["new_cost"]), f"vs {money(sim['base_cost'])} · saves {money(sim['saving'])}", BLUE)
    kpi(c[1], "Loans kept", f"{sim['new_loans']:,} / {sim['base_loans']:,}", f"{pct(sim['capture'])} of last campaign's loans",
        GREEN if sim["capture"] >= 0.95 else YELLOW if sim["capture"] >= 0.8 else RED)
    kpi(c[2], "Conversion rate", pct(sim["new_rate"]), f"vs {pct(sim['base_rate'])} · {times(sim['new_rate'], sim['base_rate'])}", AQUA)
    kpi(c[3], "Cost per loan", money(sim["new_cpa"]), f"vs {money(sim['base_cpa'])}", VIOLET)
    kpi(c[4], "Net profit", money(sim["new_profit"]), f"vs {money(sim['base_profit'])} ({'+' if sim['new_profit'] >= sim['base_profit'] else ''}"
        f"{money(sim['new_profit'] - sim['base_profit'])})", ORANGE)

    c1, c2 = st.columns(2)
    skipped = sim["n"] - sim["k"]
    with c1:
        fig = go.Figure(go.Waterfall(
            x=["Last campaign<br>(contact all)", f"Skip {skipped:,} lowest-<br>scored customers", "New campaign<br>spend"],
            y=[sim["base_cost"], -sim["saving"], sim["new_cost"]], measure=["absolute", "relative", "total"],
            text=[f"<b>{money(sim['base_cost'])}</b><br>100%", f"<b>-{money(sim['saving'])}</b><br>-{pct(1 - bshare, 0)}",
                  f"<b>{money(sim['new_cost'])}</b><br>{pct(bshare, 0)}"], textposition="auto",
            decreasing=dict(marker=dict(color=GREEN)), totals=dict(marker=dict(color=BLUE)),
            increasing=dict(marker=dict(color=DEC)), connector=dict(line=dict(color=AXIS, dash="dot"))))
        fig.update_yaxes(title="Spend ($)", tickprefix="$")
        show(fig, 380, False, "Budget waterfall: where the money goes")
    with c2:
        fig = go.Figure(go.Waterfall(
            x=["Loans last<br>campaign", "Loans lost by<br>not contacting", "Loans with<br>targeted plan"],
            y=[sim["base_loans"], -sim["lost_loans"], sim["new_loans"]], measure=["absolute", "relative", "total"],
            text=[f"<b>{sim['base_loans']:,}</b><br>100%", f"<b>-{sim['lost_loans']:,}</b><br>-{pct(1 - sim['capture'])}",
                  f"<b>{sim['new_loans']:,}</b><br>{pct(sim['capture'])}"], textposition="auto",
            decreasing=dict(marker=dict(color=RED)), totals=dict(marker=dict(color=GREEN)),
            increasing=dict(marker=dict(color=DEC)), connector=dict(line=dict(color=AXIS, dash="dot"))))
        fig.update_yaxes(title="Personal loans")
        show(fig, 380, False, "Loans waterfall: what we give up")
    kind = "good" if sim["capture"] >= 0.95 else "warn" if sim["capture"] < 0.8 else ""
    insight(f"At <b>{pct(bshare, 0)}</b> of the old budget we skip the {skipped:,} customers least likely to say yes. Spend drops by "
            f"<b>{money(sim['saving'])}</b> and we still win <b>{sim['new_loans']:,}</b> of {sim['base_loans']:,} loans ({pct(sim['capture'])}). "
            f"Conversion rises from {pct(sim['base_rate'])} to <b>{pct(sim['new_rate'])}</b>. "
            + ("Your goal of halving spend is met." if bshare <= 0.5 and sim["capture"] >= 0.95 else
               "Below this level loans start to fall quickly. See the curve below." if sim["capture"] < 0.95 else ""), kind)

    c1, c2 = st.columns([1.2, 1])
    with c1:
        g = E.gains_table(s_f, y_f, 40)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=g["pct_contacted"], y=g["pct_loans"], name="% of loans kept", mode="lines", line=dict(color=GREEN, width=3)))
        fig.add_trace(go.Scatter(x=g["pct_contacted"], y=g["conv_rate"], name="Conversion rate", mode="lines", line=dict(color=BLUE, width=3, dash="dot")))
        marks = [(bshare, f"Your budget ({pct(bshare, 0)})", YELLOW, 0.62), (b95, f"95% of loans ({pct(b95, 0)})", GREEN, 0.42)]
        if abs(bshare - 0.5) > 0.01:
            marks.append((0.5, "Goal: 50%", INK2, 0.22))
        for xv, lab, colr, yy in marks:
            fig.add_vline(x=xv, line_dash="dash", line_color=colr)
            fig.add_annotation(x=xv, y=yy, text=lab, showarrow=False, xanchor="left", xshift=4, font=dict(color=colr, size=11),
                               bgcolor="rgba(13,17,23,.85)")
        fig.update_xaxes(tickformat=".0%", title="Budget (% of customers contacted, best first)")
        fig.update_yaxes(tickformat=".0%", range=[0, 1.05])
        show(fig, 380, True, "Budget vs results curve")
    with c2:
        picked = f.assign(_r=f["Propensity"].rank(ascending=False, method="first")).nsmallest(sim["k"], "_r")
        alloc = picked.groupby("Segment", observed=True).agg(contacts=(E.TARGET, "size"), loans=(E.TARGET, "sum")).reindex(SEGS).fillna(0).reset_index()
        alloc = alloc[alloc["Segment"].isin(f["Segment"].unique())]
        alloc["share"] = alloc["contacts"] / max(alloc["contacts"].sum(), 1)
        fig = go.Figure(go.Bar(y=alloc["Segment"].astype(str), x=alloc["contacts"], orientation="h",
                               marker_color=[SEG_COL[s] for s in alloc["Segment"]],
                               text=[f"<b>{int(c):,}</b> ({pct(s, 0)}) · {int(l)} loans" for c, s, l in zip(alloc["contacts"], alloc["share"], alloc["loans"])],
                               textposition="auto"))
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_yaxes(autorange="reversed")
        fig.update_xaxes(title="Customers contacted", range=[0, max(alloc["contacts"].max(), 1) * 1.75])
        show(fig, 380, False, f"Where the {pct(bshare, 0)} budget goes, by segment")

    section("Segment playbook: who to target, what to sell, how to reach them")
    rows = []
    for s in SEGS:
        ss = f[f["Segment"] == s]
        if len(ss) == 0:
            continue
        r = ss[E.TARGET].mean()
        nb = ss[ss[E.TARGET] == 0]
        hotw = int((nb["Propensity"] >= 0.2).sum())
        lifts = []
        for p_ in ["CD Account", "Securities Account", "CreditCard", "Online", "Has Mortgage"]:
            h = ss[ss[p_] == 1]
            if len(h) >= 15 and r > 0:
                lifts.append((h[E.TARGET].mean() / r, E.PRODUCT_LABEL[p_]))
        lead = max(lifts)[1] if lifts else "–"
        if r >= 2 * RATE:
            pri = "🔥 Priority: contact every model-picked customer"
        elif r >= 0.4 * RATE:
            pri = "🎯 Selective: contact only Hot/Warm scores"
        else:
            pri = "🌱 Nurture: no mass mailing, digital nudges only"
        msg = ("Pre-approved, high-limit loan; consolidate card balances at a lower rate" if "Big Spender" in s else
               "Home improvement / renovation loan, tied to the mortgage relationship" if "Homeowner" in s else
               "Family milestones: education, car, medical; flexible EMIs" if "Families" in s else
               "Small-ticket instant loan for travel, lifestyle and upskilling" if "Singles" in s else
               "Career investment loan; preferential rate for advanced degrees" if "Professional" in s else
               "Fast, small, digital-only loan; build credit history" if "Undergrad" in s else
               "Tailored offer based on top model drivers")
        rows.append({"Segment": s, "Customers": len(ss), "Acceptance": r, "Share of loans": ss[E.TARGET].sum() / LOANS,
                     "Look-alike prospects": hotw, "Priority": pri, "Bundle with": lead,
                     "Channel": f"Online banking + email ({pct(ss['Online'].mean(), 0)} online)" if ss["Online"].mean() >= 0.5 else "Branch / phone",
                     "Message": msg})
    pb = pd.DataFrame(rows)
    pcols = st.columns(min(len(pb), 3))
    for i, r in pb.iterrows():
        pcols[i % len(pcols)].markdown(nomath(
            f'<div class="seg" style="--accent:{SEG_COL.get(r.Segment, BLUE)}; margin-bottom:12px">'
            f'<div class="n">{r.Segment}</div><div class="s" style="margin:4px 0 8px"><b style="color:{INK}">{r.Priority}</b></div>'
            f'<div class="s">Acceptance <b style="color:{INK}">{pct(r.Acceptance)}</b> · {pct(r["Share of loans"], 0)} of loans · '
            f'{r.Customers:,} customers · <b style="color:{INK}">{r["Look-alike prospects"]}</b> look-alike prospects</div>'
            f'<div class="s" style="margin-top:8px">🎁 <b style="color:{INK}">Bundle with:</b> {r["Bundle with"]}<br>'
            f'📣 <b style="color:{INK}">Channel:</b> {r.Channel}<br>💬 <b style="color:{INK}">Message:</b> {r.Message}</div></div>'),
            unsafe_allow_html=True)
    with st.expander("Playbook as a table"):
        st.dataframe(pb, width="stretch", hide_index=True,
                 column_config={"Acceptance": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
                                "Share of loans": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
                                "Look-alike prospects": st.column_config.NumberColumn(help="Customers who declined last time but score 20%+ now"),
                                "Message": st.column_config.TextColumn(width="large")})
    st.caption("“Bundle with” = the product whose holders, inside that segment, accept the loan most often (from the association analysis). "
               "“Look-alike prospects” = customers who declined last time but whose profile scores 20%+, the warmest names for a re-try.")

    section("Next-campaign target list", "Customers who do <b>not</b> have a personal loan yet, ranked by model score. Download and load into the CRM.")
    nb = f[f[E.TARGET] == 0].sort_values("Propensity", ascending=False)
    td = nb["Tier"].value_counts().reindex(E.TIERS).fillna(0)
    c1, c2 = st.columns([1, 1.6])
    with c1:
        fig = go.Figure(go.Bar(x=[t.replace(" ", "<br>") for t in td.index], y=td.values, marker_color=[TIER_COL[t] for t in td.index],
                               text=[f"<b>{int(v):,}</b><br>{pct(v / max(td.sum(), 1))}" for v in td.values], textposition="auto"))
        fig.update_yaxes(title="Customers (log scale)", type="log", tickvals=[1, 10, 100, 1000, 10000],
                         ticktext=["1", "10", "100", "1,000", "10,000"])
        fig.update_xaxes(tickangle=0)
        show(fig, 360, False, "Prospects by score tier (log scale)")
    with c2:
        show_n = st.select_slider("Show top", [25, 50, 100, 250, 500], 50, key="topn")
        out = nb.head(show_n)[["ID", "Segment", "Propensity", "Tier", "Income", "Education Level", "Family", "CCAvg", "CD Account",
                               "Mortgage", "Online", "Region"]]
        st.dataframe(out, width="stretch", hide_index=True, height=300,
                     column_config={"Propensity": st.column_config.ProgressColumn("Score", format="percent", min_value=0, max_value=1),
                                    "Income": st.column_config.NumberColumn("Income ($k)"),
                                    "CCAvg": st.column_config.NumberColumn("Card $k/mo", format="%.1f"),
                                    "Mortgage": st.column_config.NumberColumn("Mortgage ($k)")})
        exp = nb[nb["Propensity"] >= 0.05][["ID", "Segment", "Propensity", "Tier", "Income", "Education Level", "Family", "CCAvg",
                                            "CD Account", "Mortgage", "Online", "CreditCard", "Region", "ZIPCode"]]
        st.download_button(f"⬇️ Download Hot + Warm + Cool prospects ({len(exp):,} customers, CSV)", exp.to_csv(index=False).encode(),
                           "personal_loan_target_list.csv", "text/csv", type="primary")
    hw = int(td.iloc[0] + td.iloc[1])
    insight(f"<b>{hw:,}</b> non-borrowers score Hot or Warm. They look like past borrowers but said no last time, which may be bad timing or the wrong offer. "
            f"Re-approach them first with the segment-specific message above. <b>{int(td.iloc[3]):,}</b> customers are Cold (under 5% chance): drop them from paid "
            f"campaigns and reach them only through free online-banking banners.")

    section("Recommended strategy")
    st.markdown(md(f"""
1. **Stop mass-mailing.** Score every customer with the model before each campaign and contact only the top **{pct(b95, 0)}-{pct(max(b95, 0.25), 0)}**.
   The back-test shows this keeps **{pct(sim95['capture'], 0)}** of loans at **{pct(b95, 0)}** of the cost. Your target was 50%, so there is room to spare.
2. **Lead with the affluent, educated segment.** Income **$100k+** combined with **Graduate+** education or a **family of 3+** converts at 75-100%.
   Undergrad singles on high incomes almost never accept: leave them out even though they look rich.
3. **Use the CD account as the door-opener.** CD holders accept at **{pct(r_cd, 0)}** vs {pct(r_nocd, 0)}. Bundle a loan offer with CD renewals
   and maturity reminders. That moment is a natural, low-cost contact.
4. **Deliver digitally.** {pct(f['Online'].mean(), 0)} use online banking. Push pre-approved offers in-app or by email first and keep calls or branch visits for Hot scores.
5. **Keep a control group.** Hold out 5-10% of Hot/Warm customers uncontacted in every campaign so the real uplift can be measured and the model retrained.
6. **Retrain each quarter.** The model trains in seconds. Refresh it with each campaign's results so the ranking stays current.
"""))

# ====================================================================================== 7. METHOD & DATA
with tabs[7]:
    q = get_quality()
    section("Data quality and preparation")
    c = st.columns(5)
    kpi(c[0], "Rows", f"{q['rows']:,}", f"{q['columns']} columns", BLUE)
    kpi(c[1], "Missing values", f"{q['missing']}", "none to impute", GREEN)
    kpi(c[2], "Duplicates", f"{q['duplicates']}", "none removed", GREEN)
    kpi(c[3], "Negative experience", f"{q['negative_experience']}", "set to 0 (data-entry error)", YELLOW)
    kpi(c[4], "Age vs experience", f"r = {q['age_experience_corr']:.2f}", "Experience dropped from models", ORANGE)
    st.markdown(md(f"""
**Cleaning and feature preparation**
* `Experience` had {q['negative_experience']} negative values: clipped to 0. Because it moves almost exactly with `Age` (r = {q['age_experience_corr']:.2f}), it is left out of the models to avoid double-counting.
* `ZIP Code`: mapped to 6 California regions for filtering ({q['bad_zip']} malformed ZIP marked *Unknown*). Not used as a model input, because ZIP is an identifier, not a behaviour.
* Business bands were added for readability: income, age, card spend and mortgage.

**Methods, and why they are light enough to productise**

| Question | Method | Why this one | Compute |
|---|---|---|---|
| Who will accept? | Logistic Regression, Decision Tree (depth 4), **Random Forest** | LR = transparent baseline; tree = plain-English rules; RF = best accuracy | ~4 s on 5,000 rows |
| Honest scores for everyone | 5-fold out-of-fold predictions | every customer is scored by a model that never saw them | included above |
| Who looks alike? | **K-Means** (default) or Gaussian Mixture (LCA-style latent profiles) | K-Means is fast and stable; GMM finds small, sharp niches | < 1 s |
| What goes together? | **Apriori** association rules | transparent IF → THEN rules a CRM can apply directly | < 0.1 s |

**Suggestions before building the product**
* **Latent Class Analysis proper** (e.g. `stepmix`) only pays off if more categorical survey-style variables are added. With today's mostly numeric data, Gaussian Mixture gives the same latent-profile idea at a fraction of the effort.
* **Skip deep learning and heavy boosting.** Random Forest already reaches AUC {models['metrics'].loc[0, 'AUC']:.3f}; extra complexity would add cost, not conversions.
* **Add campaign metadata** (channel, offer rate, contact date) in the next campaign so the model can also learn *how* and *when* to approach each customer.
* **A/B control group**: the back-test assumes future customers behave like the last campaign's. A small hold-out proves the real uplift.

**Glossary**: *Acceptance / conversion rate* = loans ÷ customers contacted. *AUC* = chance the model ranks a random acceptor above a random decliner (0.5 = coin flip, 1 = perfect).
*Precision* = of those flagged, % who accepted. *Recall* = of all acceptors, % flagged. *Lift* = how many times better than average.
"""))
