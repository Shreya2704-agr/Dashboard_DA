"""
Campaign Intelligence Dashboard  -  Streamlit app
Run:  streamlit run app.py
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import analysis as A

st.set_page_config(page_title="Campaign Intelligence", page_icon="🎯", layout="wide", initial_sidebar_state="expanded")

# ============================================================== design tokens
SURFACE, PAGE, CARD = "#161b22", "#0d1117", "#161b22"
INK, INK2, MUTED, GRID, AXIS = "#f0efec", "#c3c2b7", "#898781", "#2c2c2a", "#383835"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = "#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"
CAT = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED]
SEG_COL = {"High CLTV": BLUE, "Medium CLTV": ORANGE, "Low CLTV": AQUA}
FAT_COL = {"Engaged": "#0ca30c", "Low Fatigue": "#fab219", "Moderate Fatigue": "#ec835a", "High Fatigue": "#d03b3b"}
FAT_ICON = {"Engaged": "🟢", "Low Fatigue": "🟡", "Moderate Fatigue": "🟠", "High Fatigue": "🔴"}
SEQ = [[0, "#0d366b"], [0.25, "#184f95"], [0.5, "#2a78d6"], [0.75, "#6da7ec"], [1, "#cde2fb"]]
DIV = [[0, RED], [0.5, AXIS], [1, BLUE]]

st.markdown(f"""
<style>
  .block-container {{padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1500px;}}
  .hero {{padding: 26px 30px; border-radius: 18px; margin-bottom: 18px;
          background: radial-gradient(1200px 300px at 0% 0%, rgba(57,135,229,.28), transparent 60%),
                      radial-gradient(900px 300px at 100% 0%, rgba(144,133,233,.22), transparent 60%), {CARD};
          border: 1px solid rgba(255,255,255,.08);}}
  .hero h1 {{margin: 0; font-size: 2.1rem; font-weight: 800; letter-spacing: -.02em;
             background: linear-gradient(90deg, #f0efec, #86b6ef 55%, #9085e9); -webkit-background-clip: text; -webkit-text-fill-color: transparent;}}
  .hero p {{margin: 6px 0 0; color: {INK2}; font-size: 1rem;}}
  .kpi {{background: {CARD}; border: 1px solid rgba(255,255,255,.08); border-radius: 14px; padding: 16px 18px; height: 100%;
         position: relative; overflow: hidden; transition: transform .15s ease, border-color .15s ease;}}
  .kpi:hover {{transform: translateY(-2px); border-color: rgba(57,135,229,.55);}}
  .kpi::before {{content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 4px; background: var(--accent, {BLUE});}}
  .kpi .lbl {{color: {MUTED}; font-size: .78rem; text-transform: uppercase; letter-spacing: .06em; font-weight: 600;}}
  .kpi .val {{color: {INK}; font-size: 1.85rem; font-weight: 750; margin-top: 4px; line-height: 1.1;}}
  .kpi .sub {{color: {INK2}; font-size: .82rem; margin-top: 4px;}}
  .insight {{background: linear-gradient(135deg, rgba(57,135,229,.10), rgba(144,133,233,.06)); border: 1px solid rgba(57,135,229,.25);
             border-radius: 12px; padding: 14px 16px; margin: 6px 0 10px; color: {INK}; font-size: .93rem; line-height: 1.5;}}
  .insight b {{color: #86b6ef;}}
  .warn {{background: rgba(208,59,59,.10); border-color: rgba(208,59,59,.35);}}
  .warn b {{color: #e66767;}}
  .good {{background: rgba(12,163,12,.10); border-color: rgba(12,163,12,.35);}}
  .good b {{color: #4fd14f;}}
  .sec {{margin: 22px 0 4px; font-size: 1.2rem; font-weight: 700; color: {INK};}}
  .secsub {{color: {MUTED}; font-size: .88rem; margin-bottom: 8px;}}
  .chip {{display:inline-block; padding: 2px 10px; border-radius: 999px; font-size: .78rem; font-weight: 600; margin-right: 6px;
          border: 1px solid rgba(255,255,255,.12); color: {INK};}}
  div[data-testid="stTabs"] button[role="tab"] {{font-size: .98rem; font-weight: 600; padding: 10px 14px;}}
  div[data-testid="stTabs"] button[aria-selected="true"] {{color: #86b6ef;}}
  section[data-testid="stSidebar"] {{border-right: 1px solid rgba(255,255,255,.06);}}
</style>
""", unsafe_allow_html=True)


# ============================================================== helpers
def style(fig, h=380, legend=True, title=None):
    fig.update_layout(
        height=h, template="plotly_dark", paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif", color=INK2, size=12),
        margin=dict(l=10, r=10, t=50 if title else 20, b=10), barcornerradius=4, bargap=0.25,
        title=dict(text=title, font=dict(size=15, color=INK), x=0.01) if title else None,
        hoverlabel=dict(bgcolor="#0d1117", bordercolor="#383835", font=dict(color=INK, size=12)),
        legend=dict(orientation="h", yanchor="top", y=-0.2, x=0, bgcolor="rgba(0,0,0,0)", font=dict(color=INK2)) if legend else None,
        showlegend=legend,
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=AXIS, linecolor=AXIS, tickfont=dict(color=MUTED))
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=AXIS, linecolor=AXIS, tickfont=dict(color=MUTED))
    return fig


def show(fig, h=380, legend=True, title=None):
    st.plotly_chart(style(fig, h, legend, title), use_container_width=True, config={"displaylogo": False, "scrollZoom": False})


def kpi(col, label, value, sub="", accent=BLUE):
    col.markdown(f'<div class="kpi" style="--accent:{accent}"><div class="lbl">{label}</div>'
                 f'<div class="val">{value}</div><div class="sub">{sub}</div></div>', unsafe_allow_html=True)


def insight(html, kind=""):
    st.markdown(f'<div class="insight {kind}">{html}</div>', unsafe_allow_html=True)


def section(title, sub=""):
    st.markdown(f'<div class="sec">{title}</div>' + (f'<div class="secsub">{sub}</div>' if sub else ""), unsafe_allow_html=True)


def money(v):
    s = "-" if v < 0 else ""
    v = abs(v)
    return f"{s}${v:,.0f}" if v < 1e6 else f"{s}${v/1e6:,.2f}M"


# ============================================================== data (cached)
@st.cache_data(show_spinner="Crunching RFM, CLTV & fatigue …")
def get_data(margin, discount):
    return A.build_all(margin, discount)


@st.cache_resource(show_spinner="Training propensity models (5-fold CV) …")
def get_models():
    return A.train_models(A.load_data())


@st.cache_data(show_spinner="Mining association rules …")
def get_rules(min_support, min_conf):
    return A.association_rules_(A.load_data(), min_support, min_conf)


# ============================================================== sidebar
with st.sidebar:
    st.markdown("## 🎛️ Control Panel")
    st.caption("Filters apply to every tab. Models are trained on the full customer base.")
    with st.expander("💰 Business assumptions", expanded=False):
        cost = st.number_input("Cost per contact", 0.5, 50.0, 3.0, 0.5, help="Z_CostContact in the data = 3")
        revenue = st.number_input("Revenue per response", 1.0, 200.0, 11.0, 1.0, help="Z_Revenue in the data = 11")
        margin = st.slider("Gross margin for CLTV", 0.05, 0.60, 0.25, 0.05)
        discount = st.slider("Annual discount rate", 0.02, 0.30, 0.10, 0.01)

base, km_diag = get_data(margin, discount)
models = get_models()
best = models["best"]
base["Propensity"] = models["scores"][best]
plain = lambda d: d.assign(Segment=d["Segment"].astype(str), Fatigue=d["Fatigue"].astype(str))  # px prefers plain strings

with st.sidebar:
    st.markdown("### 🔎 Filters")
    seg_f = st.multiselect("CLTV segment", A.SEGMENT_ORDER, A.SEGMENT_ORDER)
    fat_f = st.multiselect("Fatigue level", A.FATIGUE_ORDER, A.FATIGUE_ORDER)
    edu_f = st.multiselect("Education", sorted(base["Education"].unique()), sorted(base["Education"].unique()))
    mar_f = st.multiselect("Marital status", sorted(base["Marital_Status"].unique()), sorted(base["Marital_Status"].unique()))
    inc = st.slider("Income", int(base.Income.min()), int(base.Income.max()), (int(base.Income.min()), int(base.Income.max())), 1000)
    age = st.slider("Age", int(base.Age.min()), int(base.Age.max()), (int(base.Age.min()), int(base.Age.max())))
    kids = st.multiselect("Children at home", sorted(base["Children"].unique()), sorted(base["Children"].unique()))
    st.markdown("---")
    st.caption(f"Data: {base.attrs.get('raw_n', 2240):,} rows → {len(base):,} after cleaning · 6 campaigns")

df = base[base["Segment"].isin(seg_f) & base["Fatigue"].isin(fat_f) & base["Education"].isin(edu_f)
          & base["Marital_Status"].isin(mar_f) & base["Income"].between(*inc) & base["Age"].between(*age)
          & base["Children"].isin(kids)].copy()
if df.empty:
    st.warning("No customers match these filters - widen the selection in the sidebar.")
    st.stop()

camp = A.campaign_summary(df, cost, revenue)
be = camp.attrs["break_even"]

# ============================================================== header
st.markdown(f"""<div class="hero"><h1>🎯 Campaign Intelligence Studio</h1>
<p>Why customers stop responding — and exactly whom to target next. RFM · CLTV · Fatigue · Machine Learning</p>
<div style="margin-top:10px"><span class="chip">👥 {len(df):,} customers in view</span>
<span class="chip">📣 6 campaigns</span><span class="chip">🤖 Best model: {best}</span>
<span class="chip">⚖️ Break-even response: {be:.1%}</span></div></div>""", unsafe_allow_html=True)

tabs = st.tabs(["🏠 Executive Overview", "📣 Campaign Performance", "💎 RFM & CLTV", "😴 Campaign Fatigue",
                "🤖 AI Predictive Lab", "🎯 Strategy & Targeting", "🔍 Customer Explorer"])

# ============================================================== 1. OVERVIEW
with tabs[0]:
    c = st.columns(6)
    never = (df["Total_Accepts"] == 0).mean()
    gt_all = A.gains_table(models["y"], models["scores"][best], cost, revenue)
    best_row = gt_all.loc[gt_all["Profit"].idxmax()]
    kpi(c[0], "Customers", f"{len(df):,}", f"Avg income {money(df.Income.mean())}")
    kpi(c[1], "Avg response rate", f"{camp['Response Rate'].mean():.1%}", f"vs {be:.1%} break-even", RED)
    kpi(c[2], "Never responded", f"{never:.0%}", "ignored all 6 campaigns", "#d03b3b")
    kpi(c[3], "Total campaign P&L", money(camp["Profit"].sum()), "6 mass campaigns", ORANGE)
    kpi(c[4], "Best campaign", camp.loc[camp["Response Rate"].idxmax(), "Campaign"], f"{camp['Response Rate'].max():.1%} response", AQUA)
    kpi(c[5], "AI-targeted profit", money(best_row["Profit"]), f"contact top {int(best_row['Target %'])}% only", "#0ca30c")

    l, r = st.columns([1.35, 1])
    with l:
        fig = go.Figure(go.Bar(x=camp["Campaign"], y=camp["Response Rate"], marker_color=BLUE,
                               text=[f"{v:.1%}" for v in camp["Response Rate"]], textposition="outside", textfont=dict(color=INK2),
                               customdata=camp[["Responders", "Profit"]],
                               hovertemplate="<b>%{x}</b><br>Response %{y:.2%}<br>Responders %{customdata[0]:,}<br>P&L $%{customdata[1]:,.0f}<extra></extra>"))
        fig.add_hline(y=be, line_dash="dash", line_color=RED, annotation_text=f"Break-even {be:.1%}",
                      annotation_font_color=RED, annotation_position="top left")
        fig.update_yaxes(tickformat=".0%", range=[0, max(be, camp["Response Rate"].max()) * 1.25])
        show(fig, 360, False, "Response rate per campaign vs break-even")
    with r:
        dist = df["Total_Accepts"].value_counts().sort_index()
        ramp = ["#d03b3b", "#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]
        fig = go.Figure(go.Pie(labels=[f"{i} accepted" for i in dist.index], values=dist.values, hole=.62, sort=False,
                               marker=dict(colors=ramp[:len(dist)], line=dict(color=SURFACE, width=2)),
                               textinfo="label+percent", textposition="outside", texttemplate="%{label}<br>%{percent:.1%}",
                               textfont=dict(color=INK2), hovertemplate="%{label}: %{value:,} customers (%{percent})<extra></extra>"))
        fig.add_annotation(text=f"<b>{never:.0%}</b><br><span style='font-size:11px'>never said yes</span>", showarrow=False, font=dict(size=22, color=INK))
        show(fig, 360, False, "How many of the 6 campaigns did each customer accept?")

    l, r = st.columns(2)
    with l:
        fig = go.Figure(go.Bar(x=camp["Campaign"], y=camp["Profit"], marker_color=[RED if v < 0 else "#0ca30c" for v in camp["Profit"]],
                               text=[money(v) for v in camp["Profit"]], textposition="outside", textfont=dict(color=INK2),
                               hovertemplate="<b>%{x}</b><br>P&L $%{y:,.0f}<extra></extra>"))
        show(fig, 330, False, "Campaign P&L (revenue − contact cost)")
    with r:
        sc = pd.DataFrame({"Strategy": ["Mass (everyone)", "High-CLTV only", f"AI top {int(best_row['Target %'])}%"]})
        hi = base[base.Segment == "High CLTV"]
        sc["Profit"] = [gt_all.iloc[-1]["Profit"], hi["Response"].sum() * revenue - len(hi) * cost, best_row["Profit"]]
        sc["Rate"] = [base.Response.mean(), hi.Response.mean(), best_row["Response Rate"]]
        fig = go.Figure(go.Bar(y=sc["Strategy"], x=sc["Profit"], orientation="h",
                               marker_color=[RED if v < 0 else "#0ca30c" for v in sc["Profit"]],
                               text=[f"{money(p)} · {r:.0%} resp." for p, r in zip(sc.Profit, sc.Rate)], textposition="auto",
                               textfont=dict(color="#ffffff"), hovertemplate="%{y}: $%{x:,.0f}<extra></extra>"))
        show(fig, 330, False, "Campaign 6 replayed: same offer, three targeting strategies")

    section("🧠 Headline insights")
    a, b = st.columns(2)
    with a:
        insight(f"<b>No campaign has ever paid back.</b> Each contact costs {cost:g} and a response earns {revenue:g}, so a campaign needs "
                f"<b>{be:.1%}</b> response to break even. The best (Cmp 6) reached {camp['Response Rate'].iloc[-1]:.1%}; "
                f"Cmp 2 reached only {camp['Response Rate'].iloc[1]:.1%}.", "warn")
        insight(f"<b>{never:.0%} of customers have ignored every campaign.</b> Repeatedly mailing them is the main source of loss "
                "and the clearest sign of campaign fatigue.", "warn")
    with b:
        insight(f"<b>Response is concentrated.</b> Using the {best} propensity score, the top {int(best_row['Target %'])}% of customers hold "
                f"<b>{best_row['Capture %']:.0f}%</b> of all responders, at a <b>{best_row['Response Rate']:.0%}</b> response rate.", "good")
        insight("<b>Past acceptance and recency predict the next response.</b> Customers who bought recently, are long-tenured, spend "
                "heavily on meat/wine and are single respond far more often. Customers with kids and deal-seekers respond less.")

# ============================================================== 2. CAMPAIGNS
with tabs[1]:
    section("Who did each campaign attract?", "Index vs. customer-base average (100 = average). Blue = over-indexed, red = under-indexed.")
    prof = A.acceptor_profile(df).round(0)
    fig = go.Figure(go.Heatmap(z=prof.values, x=prof.columns, y=prof.index, colorscale=DIV, zmid=100, zmin=0, zmax=250,
                               text=prof.values.astype(int), texttemplate="%{text}",                                xgap=2, ygap=2, colorbar=dict(title="Index", tickfont=dict(color=MUTED)),
                               hovertemplate="%{x} acceptors<br>%{y}: index %{z:.0f}<extra></extra>"))
    show(fig, 440, False)
    insight("<b>Cmp 1, 2 and 5 reached affluent, big-spending, childless wine and meat buyers</b> (spend index 216–266). "
            "<b>Cmp 3 reached a different group</b>: average income, deal-driven, more children and more web visits. "
            "<b>Cmp 6 appealed to long-tenured customers</b> (tenure index 126). One message did not fit everyone. The campaigns that "
            "matched a clear persona did well, and generic ones did poorly.")

    l, r = st.columns(2)
    with l:
        co = A.co_acceptance(df)
        z = co.values.copy(); np.fill_diagonal(z, np.nan)
        fig = go.Figure(go.Heatmap(z=z, x=co.columns, y=co.index, colorscale=SEQ, zmin=0, zmax=0.8, xgap=2, ygap=2,
                                   text=np.where(np.isnan(z), "", np.char.add(np.round(np.nan_to_num(z) * 100).astype(int).astype(str), "%")),
                                   texttemplate="%{text}",                                    hovertemplate="Accepted %{y} → also accepted %{x}: %{z:.0%}<extra></extra>",
                                   colorbar=dict(tickformat=".0%", tickfont=dict(color=MUTED))))
        fig.update_yaxes(autorange="reversed", title="Accepted …")
        fig.update_xaxes(title="… also accepted")
        show(fig, 420, False, "Cross-campaign loyalty: P(accept column | accepted row)")
    with r:
        e = (df[["AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3"]].sum(axis=1) > 0).map({True: "Early: Yes (Cmp1-3)", False: "Early: No (Cmp1-3)"})
        m = (df[["AcceptedCmp4", "AcceptedCmp5"]].sum(axis=1) > 0).map({True: "Mid: Yes (Cmp4-5)", False: "Mid: No (Cmp4-5)"})
        f = df["Response"].map({1: "Cmp 6: Yes", 0: "Cmp 6: No"})
        nodes = ["Early: Yes (Cmp1-3)", "Early: No (Cmp1-3)", "Mid: Yes (Cmp4-5)", "Mid: No (Cmp4-5)", "Cmp 6: Yes", "Cmp 6: No"]
        idx = {n: i for i, n in enumerate(nodes)}
        l1 = pd.crosstab(e, m).stack().reset_index(); l1.columns = ["s", "t", "v"]
        l2 = pd.crosstab(m, f).stack().reset_index(); l2.columns = ["s", "t", "v"]
        links = pd.concat([l1, l2])
        lc = ["rgba(57,135,229,.45)" if "Yes" in s else "rgba(137,135,129,.25)" for s in links.s]
        fig = go.Figure(go.Sankey(
            node=dict(label=nodes, pad=18, thickness=16, line=dict(color=SURFACE, width=1),
                      color=[BLUE, MUTED, BLUE, MUTED, "#0ca30c", "#d03b3b"]),
            link=dict(source=links.s.map(idx), target=links.t.map(idx), value=links.v, color=lc,
                      hovertemplate="%{source.label} → %{target.label}<br>%{value:,} customers<extra></extra>")))
        show(fig, 420, False, "Response journey across campaign waves")
    insight(f"<b>Past responders keep responding.</b> {co.loc['Cmp 2','Cmp 6']:.0%} of Cmp 2 acceptors and {co.loc['Cmp 1','Cmp 6']:.0%} of Cmp 1 "
            f"acceptors also accepted Cmp 6, against a {df.Response.mean():.0%} base rate. The Sankey shows most Cmp 6 'No' answers come from "
            "customers who never said yes before. Acceptance history is the strongest free targeting signal you have.")

    section("Response by customer profile")
    l, r = st.columns([1, 1.2])
    with l:
        dim = st.selectbox("Break down Cmp 6 response by", ["Income_Band", "Education", "Marital_Status", "Children", "Age_Band", "Segment", "Fatigue"])
        g = df.groupby(dim, observed=True)["Response"].agg(["mean", "size"]).reset_index()
        fig = go.Figure(go.Bar(x=g[dim].astype(str), y=g["mean"], marker_color=BLUE, customdata=g["size"],
                               text=[f"{v:.0%}" for v in g["mean"]], textposition="outside", textfont=dict(color=INK2),
                               hovertemplate="%{x}<br>Response %{y:.1%}<br>%{customdata:,} customers<extra></extra>"))
        fig.add_hline(y=df.Response.mean(), line_dash="dot", line_color=MUTED, annotation_text="average", annotation_font_color=MUTED)
        fig.update_yaxes(tickformat=".0%")
        show(fig, 400, False)
    with r:
        sb = df.groupby(["Education", "Marital_Status"]).agg(n=("ID", "size"), rate=("Response", "mean")).reset_index()
        fig = px.sunburst(sb, path=["Education", "Marital_Status"], values="n", color="rate", color_continuous_scale=SEQ,
                          hover_data={"rate": ":.1%"})
        fig.update_traces(marker=dict(line=dict(color=SURFACE, width=2)))
        fig.update_layout(coloraxis_colorbar=dict(title="Resp.", tickformat=".0%"))
        show(fig, 400, False, "Education → Marital status (size = customers, color = Cmp 6 response)")

    section("Campaign acceptance trajectory by CLTV segment")
    tr = (df.groupby("Segment", observed=True)[A.CAMPAIGNS].mean() * 100).T
    tr.index = A.CAMPAIGN_LABELS
    fig = go.Figure()
    for s in tr.columns:
        fig.add_trace(go.Scatter(x=tr.index, y=tr[s], name=s, mode="lines+markers", line=dict(color=SEG_COL[s], width=2),
                                 marker=dict(size=9, line=dict(color=SURFACE, width=2)),
                                 hovertemplate=f"{s}<br>%{{x}}: %{{y:.1f}}%<extra></extra>"))
    fig.update_yaxes(title="Acceptance %")
    show(fig, 340)

# ============================================================== 3. RFM & CLTV
with tabs[2]:
    seg = df.groupby("Segment", observed=True).agg(Customers=("ID", "size"), Avg_CLTV=("CLTV", "mean"), Total_CLTV=("CLTV", "sum"),
                                                   Recency=("Recency", "mean"), Frequency=("Frequency", "mean"), Monetary=("Monetary", "mean"),
                                                   Income=("Income", "mean"), Response=("Response", "mean"), Prior=("Prior_Accepts", "mean"),
                                                   Deal_Share=("Deal_Share", "mean"), Children=("Children", "mean")).reset_index()
    cols = st.columns(len(seg))
    for c_, (_, row) in zip(cols, seg.iterrows()):
        kpi(c_, row["Segment"], money(row["Avg_CLTV"]),
            f"{row['Customers']:,} customers · {row['Total_CLTV']/df.CLTV.sum():.0%} of value · {row['Response']:.0%} Cmp 6 response",
            SEG_COL[row["Segment"]])

    with st.expander("📐 How CLTV & segments are calculated (plain English)"):
        st.markdown(f"""
1. **R, F and M scores (1–5).** Each customer is ranked into fifths on **Recency** (days since last purchase, lower is better), **Frequency** (web + catalog + store purchases) and **Monetary** (total spend across the 6 categories).
2. **CLTV per customer.** `annual value = spend ÷ years as a customer`, then `retention = 50% … 90%` depending on the R-score, then
   `CLTV = annual value × margin ({margin:.0%}) × r / (1 + discount ({discount:.0%}) − r)`. This is the standard retention-and-discount lifetime value formula.
3. **K-Means clustering (k = 3)** groups customers with similar recency, frequency, spend and CLTV. The clusters are then ranked by average CLTV and named **High / Medium / Low**. The elbow and silhouette charts below check that 3 is a sensible number of groups.
""")

    l, r = st.columns([1.3, 1])
    with l:
        smp = plain(df.sample(min(len(df), 1500), random_state=1))
        fig = px.scatter_3d(smp, x="Recency", y="Frequency", z="Monetary", color="Segment", color_discrete_map=SEG_COL,
                            category_orders={"Segment": A.SEGMENT_ORDER}, opacity=0.8,
                            hover_data={"ID": True, "CLTV": ":$,.0f", "Income": ":$,.0f"})
        fig.update_traces(marker=dict(size=3.5, line=dict(width=0)))
        fig.update_layout(scene_camera=dict(eye=dict(x=1.6, y=1.6, z=0.9)), scene=dict(xaxis=dict(backgroundcolor=SURFACE, gridcolor=GRID), yaxis=dict(backgroundcolor=SURFACE, gridcolor=GRID),
                                     zaxis=dict(backgroundcolor=SURFACE, gridcolor=GRID)))
        show(fig, 520, True, "3-D RFM space · drag to rotate")
    with r:
        tm = df.groupby(["Segment", "Fatigue"], observed=True).agg(n=("ID", "size"), cltv=("CLTV", "sum")).reset_index()
        tm[["Segment", "Fatigue"]] = tm[["Segment", "Fatigue"]].astype(str)
        fig = px.treemap(tm, path=[px.Constant("All customers"), "Segment", "Fatigue"], values="cltv", color="Segment",
                         color_discrete_map={**SEG_COL, "(?)": AXIS}, hover_data={"n": ":,"})
        fig.update_traces(marker=dict(line=dict(color=SURFACE, width=2)), texttemplate="<b>%{label}</b><br>%{value:$,.0f}",
                          hovertemplate="<b>%{label}</b><br>CLTV $%{value:,.0f}<br>%{customdata[0]:,} customers<extra></extra>",
                          root_color=AXIS)
        show(fig, 520, False, "Where the lifetime value sits (size = total CLTV)")

    l, r = st.columns(2)
    with l:
        hm = df.pivot_table(index="R_Score", columns="F_Score", values="Monetary", aggfunc="mean").sort_index(ascending=False)
        fig = go.Figure(go.Heatmap(z=hm.values, x=[f"F{c}" for c in hm.columns], y=[f"R{i}" for i in hm.index], colorscale=SEQ,
                                   xgap=2, ygap=2, text=np.round(hm.values).astype(int), texttemplate="$%{text}",                                    hovertemplate="%{y} · %{x}<br>Avg spend $%{z:,.0f}<extra></extra>", colorbar=dict(tickprefix="$")))
        show(fig, 400, False, "RFM grid: average spend by Recency × Frequency score")
    with r:
        fig = go.Figure()
        for s in A.SEGMENT_ORDER:
            d = df[df.Segment == s]
            if len(d):
                fig.add_trace(go.Violin(x=[s] * len(d), y=d["CLTV"].clip(lower=1), name=s, line_color=SEG_COL[s], fillcolor=SEG_COL[s],
                                        opacity=0.75, box_visible=True, meanline_visible=True, points=False, spanmode="hard"))
        fig.update_yaxes(type="log", dtick=1, title="CLTV ($, log scale)")
        show(fig, 400, False, "CLTV distribution per segment")

    l, r = st.columns(2)
    with l:
        met = {"Recency (inv.)": -seg["Recency"], "Frequency": seg["Frequency"], "Monetary": seg["Monetary"], "Income": seg["Income"],
               "Past accepts": seg["Prior"], "Cmp 6 response": seg["Response"], "Deal share": seg["Deal_Share"], "Children": seg["Children"]}
        nm = pd.DataFrame(met)
        nm = (nm - nm.min()) / (nm.max() - nm.min()).replace(0, 1)
        fig = go.Figure()
        for i, s in enumerate(seg["Segment"]):
            vals = nm.iloc[i].tolist()
            fig.add_trace(go.Scatterpolar(r=vals + vals[:1], theta=list(nm.columns) + [nm.columns[0]], name=s, fill="toself",
                                          line=dict(color=SEG_COL[s], width=2), opacity=0.55))
        fig.update_layout(polar=dict(bgcolor=SURFACE, radialaxis=dict(visible=False), angularaxis=dict(gridcolor=GRID, linecolor=AXIS)))
        show(fig, 420, True, "Segment DNA (scaled 0–1 across segments)")
    with r:
        mix = df.groupby("Segment", observed=True)[A.PRODUCTS].sum()
        mix = mix.div(mix.sum(axis=1), axis=0) * 100
        fig = go.Figure()
        for i, (p, lab) in enumerate(zip(A.PRODUCTS, A.PRODUCT_LABELS)):
            fig.add_trace(go.Bar(y=mix.index.astype(str), x=mix[p], name=lab, orientation="h", marker=dict(color=CAT[i], line=dict(color=SURFACE, width=2)),
                                 hovertemplate=f"{lab}: %{{x:.1f}}%<extra>%{{y}}</extra>"))
        fig.update_layout(barmode="stack", barcornerradius=0)
        fig.update_xaxes(ticksuffix="%")
        show(fig, 420, True, "Wallet share by category")

    l, r = st.columns(2)
    with l:
        srt = np.sort(df["CLTV"].values)[::-1]
        cum = np.cumsum(srt) / srt.sum()
        x = np.arange(1, len(srt) + 1) / len(srt)
        top20 = cum[int(len(srt) * 0.2) - 1] if len(srt) >= 5 else 1
        fig = go.Figure(go.Scatter(x=x, y=cum, mode="lines", line=dict(color=BLUE, width=2), fill="tozeroy", fillcolor="rgba(57,135,229,.15)",
                                   hovertemplate="Top %{x:.0%} of customers → %{y:.0%} of CLTV<extra></extra>"))
        fig.add_vline(x=0.2, line_dash="dot", line_color=MUTED)
        fig.add_annotation(x=0.2, y=top20, text=f"Top 20% = {top20:.0%} of value", showarrow=True, arrowcolor=MUTED, font=dict(color=INK), ax=60, ay=30)
        fig.update_xaxes(tickformat=".0%", title="Customers (ranked by CLTV)")
        fig.update_yaxes(tickformat=".0%", title="Cumulative CLTV")
        show(fig, 360, False, "Pareto: concentration of lifetime value")
    with r:
        a1, a2 = st.columns(2)
        with a1:
            fig = go.Figure(go.Scatter(x=km_diag.k, y=km_diag.Inertia, mode="lines+markers", line=dict(color=BLUE, width=2), marker=dict(size=8)))
            fig.add_vline(x=3, line_dash="dot", line_color=ORANGE)
            show(fig, 360, False, "Elbow (inertia)")
        with a2:
            fig = go.Figure(go.Bar(x=km_diag.k, y=km_diag.Silhouette, marker_color=[ORANGE if k == 3 else BLUE for k in km_diag.k],
                                   hovertemplate="k=%{x}: %{y:.3f}<extra></extra>"))
            show(fig, 360, False, "Silhouette score")

    hi_r, md_r = seg.set_index("Segment").reindex(A.SEGMENT_ORDER)["Recency"].values[:2]
    insight(f"<b>High and Medium CLTV customers spend about the same; the difference is how recently they bought.</b> Both buy ~17 times and spend "
            f"~$950. High-CLTV customers last bought ~{hi_r:.0f} days ago, Medium-CLTV ~{md_r:.0f} days ago. "
            "<b>The Medium tier is your 'at-risk VIPs'</b>: one reactivation away from High.")
    insight(f"<b>Low-CLTV customers are a different group</b>: half the income, more children, about 40% of purchases on deals, "
            f"store-first, with a larger share of spend on gold and sweets. They answer price promotions, not premium campaigns. "
            f"The top 20% of customers hold <b>{top20:.0%}</b> of total CLTV.")
    st.dataframe(seg.style.format({"Avg_CLTV": "${:,.0f}", "Total_CLTV": "${:,.0f}", "Recency": "{:.0f}", "Frequency": "{:.1f}",
                                   "Monetary": "${:,.0f}", "Income": "${:,.0f}", "Response": "{:.1%}", "Prior": "{:.2f}",
                                   "Deal_Share": "{:.0%}", "Children": "{:.2f}"}), use_container_width=True, hide_index=True)

# ============================================================== 4. FATIGUE
with tabs[3]:
    with st.expander("📐 How fatigue is defined", expanded=False):
        st.markdown("""
Campaigns are treated as a timeline: **Cmp 1 → Cmp 6**. For every customer we measure the **ignore streak**, the number of consecutive campaigns ignored since their last "yes".

| Level | Rule | Meaning |
|---|---|---|
| 🟢 **Engaged** | accepted ≥ 2 campaigns, including one of the last 2 | Responds regularly |
| 🟡 **Low Fatigue** | last "yes" within the last 3 campaigns | Responds occasionally |
| 🟠 **Moderate Fatigue** | said yes before, then ignored ≥ 3 in a row | Lapsed responder: has stopped responding |
| 🔴 **High Fatigue** | ignored all 6 campaigns | Not reached by campaigns |

**Fatigue Index (0–100)** = 60% ignore streak + 25% lifetime non-acceptance + 15% purchase recency. Higher means more fatigued.
""")
    fc = df["Fatigue"].value_counts().reindex(A.FATIGUE_ORDER).fillna(0)
    cols = st.columns(4)
    for c_, lvl in zip(cols, A.FATIGUE_ORDER):
        d = df[df.Fatigue == lvl]
        kpi(c_, f"{FAT_ICON[lvl]} {lvl}", f"{int(fc[lvl]):,}", f"{fc[lvl]/len(df):.0%} of base · avg CLTV {money(d.CLTV.mean()) if len(d) else '-'}", FAT_COL[lvl])

    l, r = st.columns(2)
    with l:
        fig = go.Figure()
        for lvl in A.FATIGUE_ORDER:
            d = df[df.Fatigue == lvl]
            fig.add_trace(go.Histogram(x=d["Fatigue_Index"], name=f"{FAT_ICON[lvl]} {lvl}", marker=dict(color=FAT_COL[lvl], line=dict(color=SURFACE, width=1)),
                                       xbins=dict(start=0, end=100, size=4)))
        fig.update_layout(barmode="stack", barcornerradius=0, bargap=0.05)
        fig.update_xaxes(title="Fatigue index (0 = fresh, 100 = fully fatigued)")
        show(fig, 380, True, "Fatigue index distribution")
    with r:
        mt = st.radio("Heatmap metric", ["Customers", "Total CLTV at stake", "Avg Cmp-6 propensity"], horizontal=True)
        if mt == "Customers":
            pv = pd.crosstab(df.Segment, df.Fatigue); fmt = "%{z:,}"
        elif mt == "Total CLTV at stake":
            pv = df.pivot_table(index="Segment", columns="Fatigue", values="CLTV", aggfunc="sum", observed=False); fmt = "$%{z:,.0f}"
        else:
            pv = df.pivot_table(index="Segment", columns="Fatigue", values="Propensity", aggfunc="mean", observed=False); fmt = "%{z:.0%}"
        pv = pv.reindex(index=[s for s in A.SEGMENT_ORDER if s in pv.index], columns=[f for f in A.FATIGUE_ORDER if f in pv.columns])
        fig = go.Figure(go.Heatmap(z=pv.values, x=pv.columns, y=pv.index, colorscale=SEQ, xgap=2, ygap=2, texttemplate=fmt,
                                   hovertemplate="%{y} × %{x}<br>" + fmt + "<extra></extra>"))
        fig.update_yaxes(autorange="reversed")
        show(fig, 330, False, f"CLTV segment × fatigue level · {mt}")

    l, r = st.columns(2)
    with l:
        tr = (df.groupby("Fatigue", observed=True)[A.CAMPAIGNS].mean() * 100)
        tr.columns = A.CAMPAIGN_LABELS
        fig = go.Figure(go.Heatmap(z=tr.values, x=tr.columns, y=[f"{FAT_ICON[i]} {i}" for i in tr.index], colorscale=SEQ, xgap=2, ygap=2,
                                   text=np.round(tr.values).astype(int), texttemplate="%{text}%",                                    hovertemplate="%{y}<br>%{x}: %{z:.1f}% accepted<extra></extra>", colorbar=dict(ticksuffix="%")))
        fig.update_yaxes(autorange="reversed")
        show(fig, 360, False, "How each fatigue group responded over time")
    with r:
        smp = plain(df.sample(min(len(df), 1500), random_state=2))
        fig = px.scatter(smp, x="Fatigue_Index", y="CLTV", color="Fatigue", color_discrete_map=FAT_COL, category_orders={"Fatigue": A.FATIGUE_ORDER},
                         log_y=True, opacity=0.75, hover_data={"ID": True, "Income": ":$,.0f", "Segment": True})
        fig.update_traces(marker=dict(size=8, line=dict(color=SURFACE, width=1)))
        hi_cltv = df["CLTV"].quantile(0.67)
        fig.update_yaxes(dtick=1)
        fig.add_hline(y=hi_cltv, line_dash="dot", line_color=MUTED)
        fig.add_vline(x=50, line_dash="dot", line_color=MUTED)
        fig.add_annotation(x=80, y=np.log10(df.CLTV.max()), text="<b>VALUE AT RISK</b><br>valuable but fatigued", showarrow=False, font=dict(color=RED))
        show(fig, 360, True, "Value-at-risk map: fatigue vs CLTV")

    risk = df[(df.Fatigue.isin(["Moderate Fatigue", "High Fatigue"])) & (df.Segment == "High CLTV")]
    silent = df[df.Silent_Loyal == 1]
    a, b = st.columns(2)
    with a:
        insight(f"<b>{fc['High Fatigue']/len(df):.0%} of customers are highly fatigued</b>: they have never answered a campaign. Their response to Cmp 6 is 0% by "
                f"definition, but their predicted propensity is low too ({df[df.Fatigue=='High Fatigue'].Propensity.mean():.0%}). "
                "Sending them the same message again is the main cost drain.", "warn")
        insight(f"<b>{len(risk):,} High-CLTV customers are moderately or highly fatigued</b>, holding <b>{money(risk.CLTV.sum())}</b> of lifetime value. "
                "Repeating the same mass campaign will not reach them. They need a different channel, offer or message.", "warn")
    with b:
        insight(f"<b>{len(silent):,} 'silent loyalists'</b>: highly fatigued toward campaigns, yet they bought recently and often. "
                "They are loyal to the store, not to the campaigns. <b>Do not discount them.</b> Remove them from mass campaigns "
                "and use in-store or loyalty recognition instead.", "good")
        insight(f"<b>Engaged customers convert at {df[df.Fatigue=='Engaged'].Response.mean():.0%}</b> and have the highest CLTV. "
                "Protect this group by limiting how often you contact them, so they do not tire of campaigns too.", "good")

# ============================================================== 5. AI LAB
with tabs[4]:
    with st.expander("🤖 What the algorithms do (plain English)", expanded=False):
        st.markdown("""
| Algorithm | Question it answers | Why it's used |
|---|---|---|
| **K-Means clustering** | Which customers look alike on R, F, M and CLTV? | Finds natural High / Medium / Low value groups without hand-set cut-offs |
| **Logistic Regression** | Baseline: how likely is each customer to say yes? | Simple and easy to interpret |
| **Random Forest** | Same question, using many decision trees | Captures interactions such as "rich *and* childless" |
| **XGBoost** (gradient boosting) | Same question, with trees that each correct the previous ones | Usually the most accurate on tabular data |
| **SHAP** | *Why* did the model give this customer a high score? | Shows each feature's push up or down on the prediction |
| **Gains / Lift analysis** | If I contact only the top X%, how many responders do I reach? | Turns the model into a targeting and ROI decision |
| **Apriori association rules** | Which trait combinations lead to "yes"? | Readable if–then rules for campaign design |

All models are evaluated with **5-fold cross-validation**: every customer is scored by a model that never saw them during training, so the accuracy figures are honest.
""")
    met = models["metrics"]
    l, r = st.columns([1, 1.4])
    with l:
        fig = go.Figure(go.Bar(y=met["Model"], x=met["ROC-AUC"], orientation="h", marker_color=[BLUE if m == best else AXIS for m in met["Model"]],
                               text=[f"AUC {a:.3f} · PR-AUC {p:.3f}" for a, p in zip(met["ROC-AUC"], met["PR-AUC"])], textposition="inside",
                               textfont=dict(color="#ffffff"), hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
        fig.update_xaxes(range=[0.5, 1])
        fig.update_yaxes(autorange="reversed")
        show(fig, 380, False, "Model leaderboard (cross-validated)")
    with r:
        fig = go.Figure()
        for i, (name, (fpr, tpr)) in enumerate(models["roc"].items()):
            fig.add_trace(go.Scatter(x=fpr, y=tpr, name=name, mode="lines", line=dict(color=CAT[i], width=2)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Random", mode="lines", line=dict(color=MUTED, dash="dot", width=1)))
        fig.update_xaxes(title="False-positive rate"); fig.update_yaxes(title="True-positive rate")
        show(fig, 380, True, "ROC curves")

    section("What drives a 'yes'? — SHAP explainability")
    pretty = lambda s: (s.replace("MntMeatProducts", "Meat spend").replace("MntFishProducts", "Fish spend").replace("MntWines", "Wine spend")
                        .replace("MntGoldProds", "Gold spend").replace("Num", "").replace("Purchases", " purchases").replace("WebVisitsMonth", "Web visits/month")
                        .replace("Tenure_Days", "Tenure (days)").replace("Deal_Share", "Deal share").replace("AcceptedCmp", "Accepted Cmp ")
                        .replace("Couple", "Couple (1=yes)").replace("Edu_", "Education: ").replace("AOV", "Avg order value"))
    shap_df, X = models["shap"].rename(columns=pretty), models["X"].rename(columns=pretty)
    imp = shap_df.abs().mean().sort_values(ascending=False)
    topn = st.slider("Features shown", 5, len(imp), 12)
    feats = imp.index[:topn][::-1]
    l, r = st.columns([1, 1.5])
    with l:
        fig = go.Figure(go.Bar(y=feats, x=imp[feats], orientation="h", marker_color=BLUE, hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
        fig.update_xaxes(title="mean |SHAP|")
        show(fig, 30 * topn + 120, False, "Global importance")
    with r:
        rng = np.random.default_rng(0)
        sidx = rng.choice(len(X), min(800, len(X)), replace=False)
        fig = go.Figure()
        for j, fn in enumerate(feats):
            v = X[fn].iloc[sidx]
            pct = v.rank(pct=True).values
            fig.add_trace(go.Scattergl(x=shap_df[fn].iloc[sidx], y=j + rng.uniform(-0.3, 0.3, len(sidx)), mode="markers", showlegend=False,
                                       marker=dict(size=5, color=pct, colorscale=[[0, "#184f95"], [0.5, "#86b6ef"], [1, "#eb6834"]], cmin=0, cmax=1,
                                                   colorbar=dict(title="Feature<br>value", tickvals=[0, 1], ticktext=["low", "high"]) if j == 0 else None,
                                                   showscale=j == 0),
                                       customdata=v.values, hovertemplate=f"{fn} = %{{customdata:,.0f}}<br>SHAP %{{x:.2f}}<extra></extra>"))
        fig.update_yaxes(tickvals=list(range(len(feats))), ticktext=list(feats))
        fig.add_vline(x=0, line_color=AXIS)
        fig.update_xaxes(title="← lowers response odds · SHAP value · raises response odds →")
        show(fig, 30 * topn + 120, False, "SHAP beeswarm (each dot = one customer)")
    insight("<b>Top drivers of responding to Cmp 6:</b> <b>Recency</b> (recent buyers say yes), <b>tenure</b> (older relationships respond more), "
            "<b>meat and wine spend</b>, <b>being single</b> and <b>accepting earlier campaigns</b>. Heavy store-only shoppers and customers with "
            "many web visits but few purchases respond less. These are the traits to target and the inputs for writing offers.")

    section("From model to money — gains, lift & ROI simulator")
    mdl_pick = st.radio("Score customers with", list(models["scores"].keys()), index=list(models["scores"].keys()).index(best), horizontal=True)
    gt = A.gains_table(models["y"], models["scores"][mdl_pick], cost, revenue)
    pct = st.slider("Contact the top X% of customers by propensity", 5, 100, int(gt.loc[gt.Profit.idxmax(), "Target %"]), 5)
    row = gt[gt["Target %"] == pct].iloc[0]
    c = st.columns(5)
    kpi(c[0], "Customers contacted", f"{int(row.Contacted):,}", f"{pct}% of base")
    kpi(c[1], "Responders captured", f"{row['Capture %']:.0f}%", f"{int(row.Responders):,} of {int(models['y'].sum()):,}", AQUA)
    kpi(c[2], "Response rate", f"{row['Response Rate']:.1%}", f"{row['Response Rate']/be:.1f}× break-even", BLUE)
    kpi(c[3], "Lift", f"{row['Lift']:.1f}×", "vs random targeting", VIOLET)
    kpi(c[4], "Campaign profit", money(row["Profit"]), f"mass mailing: {money(gt.iloc[-1].Profit)}", "#0ca30c" if row.Profit > 0 else RED)

    l, m_, r = st.columns(3)
    with l:
        fig = go.Figure()
        xs = [0] + gt["Target %"].tolist()
        fig.add_trace(go.Scatter(x=xs, y=xs, name="Random", line=dict(color=MUTED, dash="dot", width=1)))
        fig.add_trace(go.Scatter(x=xs, y=[0] + gt["Capture %"].tolist(), name="Model", line=dict(color=BLUE, width=2),
                                 fill="tonexty", fillcolor="rgba(57,135,229,.18)", hovertemplate="Top %{x}% → %{y:.0f}% of responders<extra></extra>"))
        fig.add_vline(x=pct, line_color=ORANGE, line_dash="dash")
        fig.update_xaxes(title="% customers contacted", ticksuffix="%"); fig.update_yaxes(title="% responders captured", ticksuffix="%")
        show(fig, 340, True, "Cumulative gains")
    with m_:
        fig = go.Figure(go.Bar(x=gt["Target %"], y=gt["Lift"], marker_color=[ORANGE if t == pct else BLUE for t in gt["Target %"]],
                               hovertemplate="Top %{x}%: lift %{y:.2f}×<extra></extra>"))
        fig.add_hline(y=1, line_dash="dot", line_color=MUTED)
        fig.update_xaxes(title="% customers contacted", ticksuffix="%")
        show(fig, 340, False, "Lift by targeting depth")
    with r:
        fig = go.Figure(go.Scatter(x=gt["Target %"], y=gt["Profit"], mode="lines+markers", line=dict(color=AQUA, width=2), marker=dict(size=8),
                                   hovertemplate="Top %{x}% → $%{y:,.0f}<extra></extra>"))
        opt = gt.loc[gt.Profit.idxmax()]
        fig.add_hline(y=0, line_color=AXIS)
        fig.add_annotation(x=opt["Target %"], y=opt.Profit, text=f"Optimum: top {int(opt['Target %'])}%<br>{money(opt.Profit)}", showarrow=True,
                           arrowcolor=INK2, font=dict(color=INK), ax=50, ay=50)
        fig.add_vline(x=pct, line_color=ORANGE, line_dash="dash")
        fig.update_xaxes(title="% customers contacted", ticksuffix="%")
        show(fig, 340, False, "Profit curve")

    section("Association rules (Apriori) — recipes for a 'yes'")
    a1, a2 = st.columns(2)
    ms = a1.slider("Min support", 0.01, 0.10, 0.03, 0.01, help="Share of all customers that match the rule")
    mc = a2.slider("Min confidence", 0.2, 0.8, 0.3, 0.05, help="Of customers matching the rule, share who said yes")
    rules = get_rules(ms, mc)
    if rules.empty:
        st.info("No rules at these thresholds — lower them.")
    else:
        l, r = st.columns([1.2, 1])
        with l:
            fig = px.scatter(rules.head(60), x="support", y="confidence", size="lift", color="lift", color_continuous_scale=SEQ, hover_name="Rule",
                             size_max=26, hover_data={"support": ":.1%", "confidence": ":.1%", "lift": ":.2f"})
            fig.update_traces(marker=dict(line=dict(color=SURFACE, width=1)))
            fig.add_hline(y=df.Response.mean(), line_dash="dot", line_color=MUTED, annotation_text="base response", annotation_font_color=MUTED)
            fig.update_xaxes(tickformat=".0%"); fig.update_yaxes(tickformat=".0%")
            show(fig, 420, False, "Rules → 'Responded Cmp 6' (bubble = lift)")
        with r:
            st.dataframe(rules.head(15).style.format({"support": "{:.1%}", "confidence": "{:.1%}", "lift": "{:.2f}×"}),
                         use_container_width=True, hide_index=True, height=420)
        insight(f"<b>Strongest recipe:</b> {rules.iloc[0]['Rule']} gives a {rules.iloc[0]['confidence']:.0%} response rate "
                f"(<b>{rules.iloc[0]['lift']:.1f}× the average</b>). Combining past acceptance with high income, no kids, catalog buying "
                "and meat/wine spend defines a responder persona that can be targeted directly.")

# ============================================================== 6. STRATEGY
with tabs[5]:
    section("Segment × Fatigue action matrix", "Customer counts by cell; hover for the recommended play.")
    cnt = pd.crosstab(df.Segment, df.Fatigue).reindex(index=A.SEGMENT_ORDER, columns=A.FATIGUE_ORDER).fillna(0)
    acts = [[A.recommend(s, f) for f in A.FATIGUE_ORDER] for s in A.SEGMENT_ORDER]
    prio = np.array([[5, 4, 4, 2], [4, 3, 3, 1], [2, 2, 1, 0]])
    fig = go.Figure(go.Heatmap(z=prio, x=[f"{FAT_ICON[f]} {f}" for f in A.FATIGUE_ORDER], y=A.SEGMENT_ORDER, colorscale=SEQ, showscale=False,
                               xgap=3, ygap=3, customdata=np.dstack([cnt.values, np.array(acts, dtype=object)]),
                               text=[[f"<b>{int(cnt.iloc[i, j]):,}</b> customers<br>{acts[i][j][:38]}…" for j in range(4)] for i in range(3)],
                               texttemplate="%{text}", textfont=dict(size=11),
                               hovertemplate="<b>%{y} · %{x}</b><br>%{customdata[0]:,} customers<br>▶ %{customdata[1]}<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    show(fig, 400, False)
    st.caption("Shade = investment priority (lighter = invest more, darker = suppress). Hover any cell for the full play.")
    with st.expander("📋 Full action matrix as a table"):
        st.dataframe(pd.DataFrame(acts, index=A.SEGMENT_ORDER, columns=A.FATIGUE_ORDER), use_container_width=True)

    section("🗺️ The 6-step response-rate playbook")
    plays = [
        ("1 · Stop mass mailing", f"Every mass campaign lost money. Replace 'send to all' with propensity-ranked lists; contacting the top ~{int(best_row['Target %'])}% "
                                  f"turns a {money(gt_all.iloc[-1].Profit)} loss into a {money(best_row.Profit)} profit on the same offer."),
        ("2 · Suppress the never-responders", f"{(df.Fatigue=='High Fatigue').mean():.0%} have ignored all 6 campaigns. Rest them for 2–3 cycles; reach "
                                              "silent loyalists through in-store and loyalty touchpoints instead."),
        ("3 · Win back the at-risk VIPs", "Medium-CLTV customers spend like High-CLTV ones but haven't bought for ~70 days. Send a time-bound, personalised "
                                          "wine/meat offer through catalog plus email."),
        ("4 · Persona-specific creative", "Affluent childless buyers respond to premium wine and meat (Cmp 1/5 style). Families and deal-seekers respond to "
                                          "value bundles (Cmp 3 style). Never send one message to both."),
        ("5 · Use acceptance history", "Past acceptors are 3–4× more likely to say yes. Build a 'responder club' with early access; treat a first 'yes' "
                                       "as an onboarding moment."),
        ("6 · Control frequency & test", "Cap contacts per customer by fatigue level, hold out 10% as a control group to measure true uplift, and "
                                         "re-train the model after every campaign."),
    ]
    cols = st.columns(3)
    for i, (t, d) in enumerate(plays):
        cols[i % 3].markdown(f'<div class="kpi" style="--accent:{CAT[i]}; margin-bottom:12px"><div class="val" style="font-size:1.05rem">{t}</div>'
                             f'<div class="sub" style="font-size:.88rem; line-height:1.5">{d}</div></div>', unsafe_allow_html=True)

    section("Channel mix by segment", "Where each segment actually buys: put campaign spend in the same place.")
    ch = df.groupby("Segment", observed=True)[A.CHANNELS + ["NumDealsPurchases"]].mean()
    fig = go.Figure()
    for i, (c_, lab) in enumerate(zip(A.CHANNELS + ["NumDealsPurchases"], A.CHANNEL_LABELS + ["Deals"])):
        fig.add_trace(go.Bar(x=ch.index.astype(str), y=ch[c_], name=lab, marker_color=CAT[i], hovertemplate=f"{lab}: %{{y:.1f}} purchases<extra>%{{x}}</extra>"))
    fig.update_layout(barmode="group", bargroupgap=0.08)
    show(fig, 340)

    section("🎯 Build the next campaign's target list")
    a1, a2, a3 = st.columns(3)
    thr = a1.slider("Min propensity score", 0.0, 1.0, 0.35, 0.05)
    tseg = a2.multiselect("Segments", A.SEGMENT_ORDER, A.SEGMENT_ORDER, key="tseg")
    excl = a3.checkbox("Exclude High-Fatigue customers", False,
                       help="For the NEXT campaign. Note: fatigue uses Cmp 6 too, so with this on the expected figures below are optimistic.")
    tl = df[(df.Propensity >= thr) & df.Segment.isin(tseg)]
    if excl:
        tl = tl[tl.Fatigue != "High Fatigue"]
    exp_resp = tl.Propensity.sum()
    c = st.columns(4)
    kpi(c[0], "List size", f"{len(tl):,}", f"{len(tl)/len(df):.0%} of filtered base")
    kpi(c[1], "Expected responders", f"{exp_resp:,.0f}", f"{(exp_resp/len(tl) if len(tl) else 0):.0%} expected rate", AQUA)
    kpi(c[2], "Expected profit", money(exp_resp * revenue - len(tl) * cost), f"at {cost:g}/contact, {revenue:g}/response", "#0ca30c")
    kpi(c[3], "CLTV covered", money(tl.CLTV.sum()), f"{(tl.CLTV.sum()/df.CLTV.sum() if df.CLTV.sum() else 0):.0%} of value", VIOLET)
    st.caption("Propensity = cross-validated XGBoost probability of accepting a campaign like Cmp 6. Expected responders = sum of propensities.")
    show_cols = ["ID", "Segment", "Fatigue", "Propensity", "CLTV", "Income", "Recency", "Frequency", "Monetary", "Prior_Accepts", "Action"]
    out = tl.sort_values("Propensity", ascending=False)[show_cols]
    st.dataframe(out.style.format({"Propensity": "{:.0%}", "CLTV": "${:,.0f}", "Income": "${:,.0f}", "Monetary": "${:,.0f}"})
                 .background_gradient(subset=["Propensity"], cmap="Blues"), use_container_width=True, hide_index=True, height=360)
    st.download_button("⬇️ Download target list (CSV)", out.to_csv(index=False).encode(), "target_list.csv", "text/csv", type="primary")

# ============================================================== 7. EXPLORER
with tabs[6]:
    section("Customer 360°", "Pick any customer to see their scores, fatigue and recommended action.")
    cid = st.selectbox("Customer ID", df.sort_values("CLTV", ascending=False)["ID"].tolist(),
                       format_func=lambda i: f"#{i} · {df.loc[df.ID == i, 'Segment'].iloc[0]} · {df.loc[df.ID == i, 'Fatigue'].iloc[0]}")
    cu = df[df.ID == cid].iloc[0]
    c = st.columns(5)
    kpi(c[0], "Segment", cu.Segment, f"RFM cell {cu.RFM_Cell}", SEG_COL[cu.Segment])
    kpi(c[1], "CLTV", money(cu.CLTV), f"annual value {money(cu.Annual_Value)}", BLUE)
    kpi(c[2], f"{FAT_ICON[cu.Fatigue]} Fatigue", cu.Fatigue, f"index {cu.Fatigue_Index:.0f}/100 · last yes: {cu.Last_Accept}", FAT_COL[cu.Fatigue])
    kpi(c[3], "Income", money(cu.Income), f"{int(cu.Age)} yrs · {cu.Education} · {cu.Marital_Status}", VIOLET)
    kpi(c[4], "Children", f"{int(cu.Children)}", f"tenure {int(cu.Tenure_Days)} days", AQUA)
    l, m_, r = st.columns([1, 1.2, 1.2])
    with l:
        fig = go.Figure(go.Indicator(mode="gauge+number", value=cu.Propensity * 100, number=dict(suffix="%", font=dict(color=INK)),
                                     gauge=dict(axis=dict(range=[0, 100], tickcolor=MUTED), bar=dict(color=BLUE), bgcolor=AXIS, borderwidth=0,
                                                threshold=dict(line=dict(color=RED, width=3), value=be * 100))))
        show(fig, 300, False, "Propensity to respond (red = break-even)")
    with m_:
        fig = go.Figure(go.Bar(x=A.CAMPAIGN_LABELS, y=[1] * 6, marker_color=["#0ca30c" if cu[c_] else "#3a3a38" for c_ in A.CAMPAIGNS],
                               text=["YES" if cu[c_] else "no" for c_ in A.CAMPAIGNS], textposition="inside",
                               textfont=dict(color="#ffffff"), hoverinfo="skip"))
        fig.update_yaxes(visible=False)
        show(fig, 300, False, "Campaign history")
    with r:
        vals = [cu[p] for p in A.PRODUCTS]
        avg = [df[p].mean() for p in A.PRODUCTS]
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=avg + avg[:1], theta=A.PRODUCT_LABELS + A.PRODUCT_LABELS[:1], name="Average", line=dict(color=MUTED, dash="dot")))
        fig.add_trace(go.Scatterpolar(r=vals + vals[:1], theta=A.PRODUCT_LABELS + A.PRODUCT_LABELS[:1], name=f"#{cid}", fill="toself",
                                      line=dict(color=BLUE, width=2), opacity=0.7))
        fig.update_layout(polar=dict(bgcolor=SURFACE, radialaxis=dict(gridcolor=GRID, tickfont=dict(color=MUTED)), angularaxis=dict(gridcolor=GRID)))
        show(fig, 300, True, "Spend by category vs average ($)")
    insight(f"<b>Recommended play:</b> {cu.Action}")

    section("Full customer table")
    st.dataframe(df[["ID", "Segment", "Fatigue", "Fatigue_Index", "Propensity", "CLTV", "R_Score", "F_Score", "M_Score", "Income", "Age",
                     "Education", "Marital_Status", "Children", "Recency", "Frequency", "Monetary", "Total_Accepts"]]
                 .style.format({"Propensity": "{:.0%}", "CLTV": "${:,.0f}", "Income": "${:,.0f}", "Monetary": "${:,.0f}"}),
                 use_container_width=True, hide_index=True, height=420)

st.markdown(f"<div style='text-align:center;color:{MUTED};font-size:.8rem;margin-top:30px'>Campaign Intelligence Studio · "
            "Streamlit · Plotly · scikit-learn · XGBoost · SHAP · mlxtend</div>", unsafe_allow_html=True)
