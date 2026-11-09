"""
Business idea validation dashboard (Individual PBL, Data Analytics).
Run:  streamlit run app.py

Layout: Home, four analysis sections (Descriptive, Diagnostic, Predictive, Prescriptive) and a
Survey & data page. Every chart is followed by an Insight and a Decision computed from the data.
"""
import ast
import html
import os
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import streamlit as st

import analytics as A
from pipeline import clean, slug
from survey_design import HEEL_TYPES, MULTI_SELECT, QUESTIONS, HEEL_FREQ, PAINS, SITUATIONS, FEATURES, CONCERNS, CARRY_FREQ, OCCUPATIONS, COMMUTE, CITIES

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))

# ------------------------------------------------------------------ names, sourced facts, planning constants
PRODUCT_NAME = "Convertto"             # brand name: change it here only
TAGLINE = "Detach, switch, impress"
FILE_SLUG = re.sub(r"[^a-z0-9]+", "-", PRODUCT_NAME.lower()).strip("-")   # used in downloaded file names
MARKET_PRICES = "Convertible-heel brands abroad sell for roughly &#36;100 to €330 a pair"
GST_THRESHOLD = 2500                   # ₹ per pair
GST_NOTE = ("GST on footwear is 5% up to ₹2,500 per pair and 18% above it (GST 2.0, effective 22 Sep 2025; "
            "confirm the rate with a CA)")
ENTRY_PRICE = 2499                     # planning assumption: entry kit price (default in the break-even simulator)
K_CHOSEN = 4                           # number of K-Means segments used for personas and targeting
HOT, WARM = 0.6, 0.3                   # lead-scoring thresholds on P(Yes)
PREMIUM_WTP, ENTRY_WTP = 3800, 2600     # segment median WTP cut-offs for the premium / entry offer

st.set_page_config(page_title=f"{PRODUCT_NAME} - Idea Validation", page_icon="👠", layout="wide")

# ------------------------------------------------------------------ brand palette
# WCAG AA contrast (text needs 4.5:1) was checked for every text colour used:
#   navy on cream 17.7, navy on white 19.3, navy on camel 8.2, navy on greige 12.8, white on espresso 6.9,
#   espresso on cream 6.3, MUTED on cream 6.7, DANGER on cream 6.0, every SEG colour on white >= 5.8.
#   Camel (2.2:1) and greige (1.4-2.0:1) fail as text, so they are used only for fills, borders and markers.
CREAM, WHITE, INK = "#FFF3E7", "#FFFFFF", "#0B0E17"
GREIGE, GREIGE_DARK = "#D8D2C7", "#BFB6AA"     # borders, dividers, gridlines, neutral fills
MUTED = "#5E554A"                              # secondary text: darkened greige (greige itself fails as text)
CAMEL, ESPRESSO = "#C9A27E", "#6D5641"
ACCENT = CAMEL                                 # UI accent: buttons, active tab, card / KPI accent borders
ACCENT_TEXT = ESPRESSO                         # text that should read as accent (labels, eyebrows)
MARK, MARK2 = ESPRESSO, CAMEL                  # main and secondary chart series
LINE = GREIGE
YES, MAYBE, NO = ESPRESSO, CAMEL, GREIGE       # intent colours, used everywhere intent appears
INTENT_COL = {"Yes": YES, "Maybe": MAYBE, "No": NO}
INTENT_TXT = {"Yes": WHITE, "Maybe": INK, "No": INK}
INTENT_EDGE = {"Yes": WHITE, "Maybe": WHITE, "No": INK}      # No gets a thin navy outline
INTENT_LW = {"Yes": 1.5, "Maybe": 1.5, "No": 0.6}
NEUTRAL, NEUTRAL_LIGHT = GREIGE_DARK, GREIGE
RISK, DANGER = "#D9692B", "#B42318"            # only for concerns, warnings and failures
SEG = ["#3E5C76", "#5B6B2F", "#7A4E6D", "#2F6B66", "#23395B", "#57534E", "#6B5B95", INK]   # segment colours


CORR_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list("corr", ["#3E5C76", WHITE, ESPRESSO])
SEQ_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list("seq", ["#FFFBF6", CAMEL, ESPRESSO])
GREY_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list("grey", ["#FFFBF6", GREIGE_DARK, MUTED])


def seg_col(c):
    return SEG[int(c) % len(SEG)]


plt.rcParams.update({
    "figure.dpi": 110, "figure.facecolor": WHITE, "axes.facecolor": WHITE, "savefig.facecolor": WHITE,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "axes.edgecolor": GREIGE_DARK, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.grid": True, "grid.color": "#ECE7DF", "grid.linewidth": 0.7, "axes.axisbelow": True,
    "font.size": 10, "axes.titlesize": 11.5, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "axes.titlelocation": "left", "axes.titlepad": 10, "legend.frameon": False,
})

CSS = """<style>
.block-container{padding-top:4.8rem;padding-bottom:3rem;max-width:1240px}
[class*="st-key-nav_"] button{padding:.4rem .35rem;min-height:2.6rem}
[class*="st-key-nav_"] button p, [class*="st-key-home_"] button p{font-size:.9rem;line-height:1.2;white-space:normal;overflow:visible;text-overflow:clip}
.hdr{background:%CREAM%;border-bottom:1px solid %GREIGE%;padding:14px 2px 16px 2px;margin:4px 0 18px 0;display:flex;justify-content:space-between;align-items:center;gap:14px;flex-wrap:wrap}
.hdr .hd{flex:1;min-width:240px}
.chip{display:inline-block;border:1px solid var(--c);color:%INK%;border-radius:999px;padding:1px 9px;margin:2px 4px 2px 0;font-size:.78rem}
.fam .big{font-size:1.7rem;font-weight:800;color:%INK%;line-height:1.1}
.hdr .eb{color:%ACCENT_TEXT%;font-weight:700;font-size:.78rem;letter-spacing:.06em;text-transform:uppercase}
.hdr .brand{color:%INK%;font-size:2.6rem;font-weight:900;letter-spacing:-.01em;line-height:1.1;margin-top:4px}
.hdr .tag{color:%ACCENT_TEXT%;font-size:1.05rem;font-weight:600;font-style:italic;margin-top:2px}
.hdr .t{color:%INK%;font-size:1.9rem;font-weight:800;line-height:1.2;margin-top:4px}
.hdr .s{color:%MUTED%;font-size:1rem;margin-top:6px}
.kpi{background:%WHITE%;border:1px solid %GREIGE%;border-left:5px solid var(--c);border-radius:10px;padding:12px 16px;height:100%}
.kpi .v{font-size:1.65rem;font-weight:800;color:var(--vc,%INK%);line-height:1.2;overflow-wrap:anywhere}
.kpi .l{color:%MUTED%;font-size:.84rem;margin-top:2px}
.tile{background:%WHITE%;border:1px solid %GREIGE%;border-top:4px solid %ACCENT%;border-radius:10px;padding:16px 18px;min-height:140px;margin-bottom:8px}
.tile .t{font-size:1.1rem;font-weight:800;color:%INK%}
.tile .q{color:%ACCENT_TEXT%;font-weight:700;font-size:.9rem;margin-top:2px}
.tile .d{color:%MUTED%;font-size:.88rem;margin-top:8px;line-height:1.45}
.card{background:%WHITE%;border:1px solid %GREIGE%;border-top:4px solid var(--c);border-radius:10px;padding:14px 16px;margin-bottom:12px;height:100%}
.card .h{font-weight:800;color:%INK%;font-size:1rem}
.card .sub{color:%MUTED%;font-size:.84rem;margin-bottom:6px}
.card .lbl{color:%ACCENT_TEXT%;font-weight:700;font-size:.74rem;text-transform:uppercase;letter-spacing:.05em;margin-top:9px}
.card p{margin:2px 0 0 0;font-size:.88rem;color:%INK%;line-height:1.45}
.card ul{margin:4px 0 0 18px;padding:0;font-size:.88rem;color:%INK%;line-height:1.5}
.badge{display:inline-block;border:1px solid %GREIGE_DARK%;color:%MUTED%;border-radius:999px;padding:1px 9px;font-size:.74rem;font-weight:700;margin-bottom:8px}
.bar{background:#F3EDE4;border-radius:999px;height:8px;margin:3px 0 8px 0;overflow:hidden}
.bar>div{height:8px;border-radius:999px;background:var(--c)}
.row2{display:flex;justify-content:space-between;gap:8px;font-size:.85rem;color:%INK%}
.insight{background:%WHITE%;border:1px solid %GREIGE%;border-left:4px solid %ACCENT%;border-radius:6px;padding:10px 14px;margin:0 0 24px 0;font-size:.9rem;line-height:1.5;color:%INK%}
.insight .k{font-weight:700;color:%ACCENT_TEXT%;margin-right:6px}
.insight details{margin-top:4px;color:%MUTED%;font-size:.84rem}
.insight summary{cursor:pointer}
.step{display:inline-block;border:1px solid %GREIGE%;background:%WHITE%;border-radius:8px;padding:5px 11px;margin:3px 2px;font-size:.85rem;color:%INK%}
.arrow{color:%MUTED%;margin:0 2px}
.sbox{background:%WHITE%;border:1px solid %GREIGE%;border-left:4px solid %ACCENT%;border-radius:8px;padding:8px 12px;font-size:.88rem;color:%INK%}
div[data-testid="stImage"] img{border:1px solid %GREIGE%;border-radius:10px}
hr{border-color:%GREIGE% !important}
/* Streamlit widgets drawn in the camel primary colour: never white or camel text on them */
.stButton>button{border-radius:8px;font-weight:600}
[data-testid="stBaseButton-primary"], [data-testid="stBaseButton-primary"] *{color:%INK% !important}
[data-testid="stButtonGroup"] button[aria-checked="true"], [data-testid="stButtonGroup"] button[aria-checked="true"] *{color:%INK% !important}
[data-testid="stButtonGroup"] button[aria-checked="true"]{background:#F6E9DC !important;border-color:%ACCENT% !important;font-weight:700}
[role="tab"][aria-selected="true"], [role="tab"][aria-selected="true"] *{color:%INK% !important;font-weight:700}
[data-baseweb="tag"], [data-testid="stMultiSelectTagsContainer"] [role="button"]{background-color:%ACCENT% !important}
[data-baseweb="tag"], [data-baseweb="tag"] *, [data-testid="stMultiSelectTagsContainer"] span{color:%INK% !important}
[data-testid="stMultiSelectTagsContainer"] svg{fill:%INK% !important;color:%INK% !important}
[data-testid="stSliderThumbValue"], [data-testid="stSliderThumbValue"] *{color:%INK% !important}
@media (max-width:640px){.hdr .brand{font-size:2rem}.hdr .t{font-size:1.45rem}.kpi .v{font-size:1.35rem}}
</style>"""
for _k, _v in {"%GREIGE_DARK%": GREIGE_DARK, "%GREIGE%": GREIGE, "%ACCENT_TEXT%": ACCENT_TEXT, "%ACCENT%": ACCENT,
               "%INK%": INK, "%MUTED%": MUTED, "%CREAM%": CREAM, "%WHITE%": WHITE}.items():
    CSS = CSS.replace(_k, _v)
st.markdown(CSS, unsafe_allow_html=True)


# ------------------------------------------------------------------ small helpers
def pct(x):
    return f"{x:.0%}"


def inr(x):
    return f"₹{x:,.0f}"


def _md(t):
    """Escape text for HTML, then turn **bold** into <b>."""
    t = html.escape(str(t), quote=False).replace("&amp;#36;", "&#36;")
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)


def explain(what, insight, decision):
    """Insight + Decision always visible directly under a chart; 'how to read' is one click away."""
    st.markdown(
        f"<div class='insight'><div><span class='k'>Insight</span>{_md(insight)}</div>"
        f"<div><span class='k'>Decision</span>{_md(decision)}</div>"
        f"<details><summary>How to read this chart</summary>{_md(what)}</details></div>",
        unsafe_allow_html=True)


def kpis(items):
    """items: (value, label) or (value, label, colour)"""
    cols = st.columns(len(items))
    for c, it in zip(cols, items):
        val, label = it[0], it[1]
        color = it[2] if len(it) > 2 else ACCENT
        vcolor = DANGER if color == DANGER else INK
        c.markdown(f"<div class='kpi' style='--c:{color};--vc:{vcolor}'><div class='v'>{_md(val)}</div>"
                   f"<div class='l'>{_md(label)}</div></div>", unsafe_allow_html=True)
    st.write("")


def header(eyebrow, title, sub):
    st.markdown(f"<div class='hdr'><div class='hd'><div class='eb'>{_md(eyebrow)}</div><div class='t'>{_md(title)}</div>"
                f"<div class='s'>{_md(sub)}</div></div></div>", unsafe_allow_html=True)


def bar_html(label, value, color, fmt="{:.0%}"):
    return (f"<div class='row2'><span>{_md(label)}</span><b>{fmt.format(value)}</b></div>"
            f"<div class='bar' style='--c:{color}'><div style='width:{min(max(value, 0), 1) * 100:.0f}%'></div></div>")


def rationale(cards):
    """Algorithm rationale cards, two per row. Each card: name, why, business, strengths, weakness."""
    for i in range(0, len(cards), 2):
        cols = st.columns(2)
        for col, c in zip(cols, cards[i:i + 2]):
            col.markdown(
                f"<div class='card' style='--c:{ACCENT}'><div class='h'>{_md(c['name'])}</div>"
                f"<div class='lbl'>Why this algorithm</div><p>{_md(c['why'])}</p>"
                f"<div class='lbl'>What it tells the business</div><p>{_md(c['business'])}</p>"
                f"<div class='lbl'>Strengths</div><p>{_md(c['strengths'])}</p>"
                f"<div class='lbl'>Weaknesses in this dataset</div><p>{_md(c['weakness'])}</p></div>",
                unsafe_allow_html=True)


def fig(w=6.4, h=3.6):
    return plt.subplots(figsize=(w, h))


def show(f):
    f.tight_layout()
    st.pyplot(f, clear_figure=True)
    plt.close(f)


def pretty(col):
    """Readable column name with a capital first letter (also fixes analytics.pretty turning pain_severity into
    'Pain: severity')."""
    label = "pain severity" if col == "pain_severity" else A.pretty(col)
    return label[:1].upper() + label[1:]


def short(name):
    return name.split(" (")[0]


def seg_names(prof):
    """Short segment names keyed by cluster id; the cluster id is appended only if two names collide."""
    s = prof["Name"].map(short)
    dup = s.duplicated(keep=False)
    return {c: (f"{v} [{c}]" if dup[c] else v) for c, v in s.items()}


def hbar(series, title, color=MARK, fmt="{:.0%}", xlabel="", w=6.4, h=None):
    """Horizontal bars, sorted, direct labels; the top bar is full colour, the rest muted."""
    s = series.sort_values()
    f, ax = fig(w, h or max(2.4, 0.38 * len(s) + 0.8))
    bars = ax.barh(s.index.astype(str), s.values, color=color, height=0.62)
    for b in bars[:-1]:
        b.set_alpha(0.45)
    for i, v in enumerate(s.values):
        ax.text(v, i, " " + fmt.format(v), va="center", fontsize=9, color=INK,
                fontweight="bold" if i == len(s) - 1 else "normal")
    ax.set_title(title); ax.set_xlabel(xlabel); ax.grid(False); ax.set_xticks([])
    ax.set_xlim(0, s.max() * 1.22 if s.max() > 0 else 1)
    show(f)


def donut(values, colors, labels, center, sub="", w=6.4, h=4.2, title=None, legend_cols=None):
    f, ax = plt.subplots(figsize=(w, h))
    wedges, _ = ax.pie(values, colors=colors, startangle=90, counterclock=False,
                       wedgeprops=dict(width=0.34, edgecolor="white", linewidth=3))
    for wdg, lab in zip(wedges, labels or []):
        if lab == "No":                      # No = greige with a thin navy outline
            wdg.set_edgecolor(INK); wdg.set_linewidth(0.8)
    ax.text(0, 0.08, center, ha="center", va="center", fontsize=22, fontweight="bold", color=INK)
    ax.text(0, -0.2, sub, ha="center", va="center", fontsize=9.5, color=MUTED)
    if labels:
        tot = float(np.sum(values))
        ax.legend([f"{l}  {v / tot:.0%}" for l, v in zip(labels, values)],
                  loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=legend_cols or len(labels), fontsize=9, handlelength=1)
    if title:
        ax.set_title(title)
    ax.set_aspect("equal"); ax.grid(False)
    show(f)


def yes_rate_bar(data, col, title, order=None, w=6.4):
    g = data.groupby(col)["purchase_intent"].value_counts(normalize=True).unstack().fillna(0)
    g = g.reindex(columns=["Yes", "Maybe", "No"], fill_value=0)
    if order is not None:
        g = g.reindex([o for o in order if o in g.index])
    f, ax = fig(w, max(2.6, 0.42 * len(g) + 1))
    left = np.zeros(len(g))
    for c in ["Yes", "Maybe", "No"]:
        ax.barh(g.index.astype(str), g[c], left=left, color=INTENT_COL[c], label=c, height=0.62,
                edgecolor=INTENT_EDGE[c], linewidth=INTENT_LW[c])
        for i, (l, v) in enumerate(zip(left, g[c])):
            if v > 0.07:
                ax.text(l + v / 2, i, f"{v:.0%}", ha="center", va="center", color=INTENT_TXT[c], fontsize=8.5,
                        fontweight="bold")
        left += g[c].values
    ax.invert_yaxis(); ax.set_xlim(0, 1); ax.grid(False); ax.set_xticks([])
    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=8.5)
    ax.set_title(title, pad=24)
    show(f)
    return g


def trend_word(values):
    v = np.asarray(values, dtype=float)
    d = np.diff(v)
    if (d >= 0).all():
        return "rises at every step"
    if (d <= 0).all():
        return "falls at every step"
    return "is not monotonic"


def silhouette_quality(s):
    # common rule of thumb (Kaufman & Rousseeuw): < 0.25 weak, 0.25-0.5 reasonable, > 0.5 strong structure
    return "weak" if s < 0.25 else ("reasonable" if s <= 0.5 else "strong")


# ------------------------------------------------------------------- data & cached models
@st.cache_data(show_spinner=False)
def load_default():
    raw = pd.read_csv(os.path.join(HERE, "data", "raw_survey_responses.csv"))
    c, log = clean(raw)
    return raw, c, log


@st.cache_data(show_spinner="Cleaning uploaded data...")
def load_uploaded(b):
    import io
    raw = pd.read_csv(io.BytesIO(b))
    c, log = clean(raw)
    return raw, c, log


@st.cache_data(show_spinner=False)
def generator_facts():
    """Read persona names, sample size and seed from the generator source without running it."""
    path = os.path.join(HERE, "01_generate_synthetic_data.py")
    out = {"personas": [], "n": None, "seed": None}
    try:
        tree = ast.parse(open(path, encoding="utf-8").read())
    except (OSError, SyntaxError):
        return out
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name == "PERSONAS" and isinstance(node.value, ast.Dict):
                out["personas"] = [k.value for k in node.value.keys if isinstance(k, ast.Constant)]
            elif name == "N" and isinstance(node.value, ast.Constant):
                out["n"] = node.value.value
            elif name == "SEED" and isinstance(node.value, ast.Constant):
                out["seed"] = node.value.value
    return out


@st.cache_data(show_spinner="Training 4 classification models...")
def cached_classification(df):
    r = A.run_classification(df)
    r.pop("models")
    return r


@st.cache_data(show_spinner="Training Yes-vs-not model...")
def cached_binary(df, model):
    r = A.run_binary(df, model)
    r.pop("model")
    return r


@st.cache_resource(show_spinner=False)
def cached_binary_model(df_hash, _df, model):
    return A.run_binary(_df, model)["model"]


@st.cache_data(show_spinner="Scanning k = 2..8...")
def cached_scan(df):
    return A.cluster_scan(df)


@st.cache_data(show_spinner="Running K-Means...")
def cached_kmeans(df, k):
    r = A.run_kmeans(df, k)
    r.pop("model"); r.pop("scaler")
    return r


@st.cache_resource(show_spinner=False)
def cached_kmeans_model(df_hash, _df, k):
    r = A.run_kmeans(_df, k)
    return r["model"], r["scaler"], r["profile"]


@st.cache_data(show_spinner="Running Latent Class Analysis (EM)...")
def cached_lca(df, k):
    r = A.run_lca(df, k)
    return {kk: r[kk] for kk in ("labels", "pi", "theta", "levels", "vars", "bic", "entropy", "ll")}


@st.cache_data(show_spinner="Comparing 2-6 latent classes...")
def cached_lca_scan(df):
    return A.lca_bic_scan(df)


@st.cache_data(show_spinner="Fitting Linear / Ridge / Lasso...")
def cached_regression(df):
    r = A.run_regression(df)
    r.pop("models"); r.pop("scaler")
    return r


@st.cache_resource(show_spinner=False)
def cached_regression_models(df_hash, _df):
    r = A.run_regression(_df)
    return r["models"], r["scaler"], r["columns"]


def regression_extra(columns):
    """Extra inputs the regression adds on top of the classification matrix (derived, not typed)."""
    base = set(A.model_matrix(df.head(2)).columns)
    return [c for c in columns if c not in base and c in df.columns]


# ---------------------------------------------------------------- sidebar / data
st.sidebar.markdown(f"<div style='font-size:1.45rem;font-weight:900;color:{INK};line-height:1.1'>{_md(PRODUCT_NAME)}</div>"
                    f"<div style='font-style:italic;color:{MUTED};font-size:.85rem'>{_md(TAGLINE)}</div>",
                    unsafe_allow_html=True)
st.sidebar.caption("Convertible heels · India launch")
up = st.sidebar.file_uploader("Real survey CSV (same columns) - optional", type="csv",
                              help="When the real survey comes back, upload the export here. "
                                   "It runs through the same cleaning pipeline and every model re-trains.")
if up is not None:
    raw, df, log = load_uploaded(up.getvalue())
    st.sidebar.markdown(f"<div class='sbox'>Using uploaded data: {len(df):,} respondents</div>", unsafe_allow_html=True)
else:
    raw, df, log = load_default()
    st.sidebar.markdown(f"<div class='sbox'>Using synthetic survey: {len(df):,} respondents</div>", unsafe_allow_html=True)
DF_HASH = int(pd.util.hash_pandas_object(df, index=False).sum())

n = len(df)
intent_share = df["purchase_intent"].value_counts(normalize=True).reindex(["Yes", "Maybe", "No"]).fillna(0)
yes_rate, maybe_rate, no_rate = intent_share["Yes"], intent_share["Maybe"], intent_share["No"]
interested = df[df.purchase_intent != "No"]
med_wtp = interested["wtp_base_inr"].median()
tier1_share = (df.city_tier == "Tier 1").mean()

# ---------------------------------------------------------------- navigation
SECTIONS = {
    "Descriptive": ("What happened?", "Who answered, their pains, needs, budgets and the customer segments."),
    "Diagnostic": ("Why did it happen?", "What separates a Yes from a No: drivers, significance tests and the barriers holding back Maybe."),
    "Predictive": ("What will happen?", "The models: who will buy, how much she will pay, which segments exist and which features go together."),
    "Prescriptive": ("What should we do?", "Launch features, price, target segments, break-even, limitations and next steps."),
}
PAGES = ["Home"] + list(SECTIONS) + ["Survey & data"]
if "section" not in st.session_state:
    st.session_state.section = "Home"


def go(s):
    st.session_state.section = s


def top_nav():
    cols = st.columns(len(PAGES))
    for c, k in zip(cols, PAGES):
        c.button(k, key=f"nav_{k}", on_click=go, args=(k,), width="stretch",
                 type="primary" if st.session_state.section == k else "secondary")


def subnav(key, options):
    v = st.segmented_control("View", options, default=options[0], key=f"view_{key}", label_visibility="collapsed")
    st.write("")
    return v or options[0]


# =================================================================== HOME
def page_home():
    st.markdown(f"<div class='hdr'><div class='hd'><div class='eb'>Business idea validation · India</div>"
                f"<div class='brand'>{_md(PRODUCT_NAME)}</div><div class='tag'>{_md(TAGLINE)}</div>"
                "<div class='s'>One sandal body, four click-on heels (kitten, stiletto, block, wedge), flat in seconds. "
                "Is it a sustainable business in India?</div></div></div>", unsafe_allow_html=True)
    kpis([(f"{n:,}", "Survey respondents"), (pct(yes_rate), "Would buy (Yes)", YES),
          (pct(maybe_rate), "Undecided (Maybe)", MAYBE), (inr(med_wtp), "Median willingness to pay (Yes + Maybe)")])

    st.markdown("#### Choose an analysis")
    cols = st.columns(4)
    detail = {"Descriptive": "Profiles, pains, features, price, segments",
              "Diagnostic": "Drivers, correlations, chi-square, Maybe barriers",
              "Predictive": "Classification, regression, clustering, association rules, lead scoring",
              "Prescriptive": "Launch bundle, price, targeting, break-even, next steps"}
    for c, (k, (q, _)) in zip(cols, SECTIONS.items()):
        with c:
            st.markdown(f"<div class='tile'><div class='t'>{k}</div><div class='q'>{q}</div>"
                        f"<div class='d'>{detail[k]}</div></div>", unsafe_allow_html=True)
            st.button(f"Open {k}", key=f"home_{k}", on_click=go, args=(k,), width="stretch", type="primary")
    st.button("Survey design, raw data and cleaning", key="home_data", on_click=go, args=("Survey & data",))

    st.markdown("#### Which of the four heels do women want?")
    mods = ["Kitten", "Stiletto", "Block", "Wedge"]
    want = {m: df[f"feat_{slug(m + ' heel module')}"].mean() for m in mods}
    for c, m in zip(st.columns(4), mods):
        c.markdown(f"<div class='card' style='--c:{ACCENT}'><div class='h' style='margin-bottom:6px'>{m} heel</div>"
                   + bar_html("Interested (Q23)", want[m], MARK) + "</div>", unsafe_allow_html=True)
    top_m, low_m = max(want, key=want.get), min(want, key=want.get)
    explain("Share of all respondents who ticked each heel module in Q23; the bar length is that share.",
            f"The **{top_m}** heel is the most wanted module ({pct(want[top_m])}) and the **{low_m}** heel the least "
            f"({pct(want[low_m])}).",
            f"Develop the {top_m} heel mould first; the {low_m} heel can wait for a later phase.")

    st.write("")
    c1, c2 = st.columns([1, 1.4])
    with c1:
        donut(intent_share.values, [INTENT_COL[k] for k in intent_share.index], list(intent_share.index),
              pct(yes_rate + maybe_rate), "Yes + Maybe", w=5.2, h=4.4, title="Q29: Would you buy it in 6 months?")
        explain("Share of all respondents giving each answer to Q29.",
                f"{pct(yes_rate)} say Yes, {pct(maybe_rate)} Maybe and {pct(no_rate)} No; "
                f"{pct(yes_rate + maybe_rate)} are at least open to buying.",
                "Enough stated demand to continue the analysis; the Predictive and Prescriptive sections test whether it "
                "converts into a business.")
    with c2:
        st.markdown("#### The idea in three cards")
        a, b, c = st.columns(3)
        a.markdown(f"<div class='card' style='--c:{ACCENT}'><div class='h'>Product</div>"
                   "<p>Sandal body plus click-on kitten, stiletto, block and wedge heels; becomes a flat in seconds.</p></div>",
                   unsafe_allow_html=True)
        b.markdown(f"<div class='card' style='--c:{ACCENT}'><div class='h'>Pain</div>"
                   f"<p>{pct((df['pain_count'] > 0).mean())} of respondents report at least one pain after heels; "
                   f"{pct((df['carry_extra_pair'] >= 4).mean())} often or always carry an extra pair.</p></div>",
                   unsafe_allow_html=True)
        c.markdown(f"<div class='card' style='--c:{ACCENT}'><div class='h'>Market</div>"
                   f"<p>{MARKET_PRICES}; only {pct(df['aware_flag'].mean())} of respondents had heard of the concept.</p></div>",
                   unsafe_allow_html=True)
        st.markdown("#### How the analysis flows")
        steps = ["Survey (30 Qs)", "Synthetic data", "Cleaning", "Descriptive", "Diagnostic", "Predictive", "Prescriptive"]
        st.markdown("<span class='arrow'>→</span>".join(f"<span class='step'>{s}</span>" for s in steps),
                    unsafe_allow_html=True)
        st.caption("Real survey answers can be uploaded from the sidebar; every number and chart then updates.")

    synthetic_panel()
    with st.expander("Business questions and the technique that answers each"):
        st.table(pd.DataFrame([
            ["Will a given woman buy it?", "Classification (KNN, Decision Tree, Random Forest, Gradient Boosting)", "Predictive"],
            ["Who are our customers and which segment converts best?", "Clustering (K-Means + Latent Class Analysis)", "Descriptive / Predictive"],
            ["Why do some women say Yes?", "Correlation, chi-square, ANOVA", "Diagnostic"],
            ["How much will she pay?", "Regression (Linear, Ridge, Lasso)", "Predictive"],
            ["Which features do we launch first?", "Association rule mining (Apriori)", "Predictive / Prescriptive"],
            ["What price, and when do we break even?", "Demand curve + break-even simulation", "Prescriptive"],
        ], columns=["Business question", "Technique", "Analytics level"]))


def log_count(prefix, contains=None):
    """Sum of records_affected for cleaning-log steps starting with `prefix` (optionally containing text)."""
    lg = load_default()[2]
    m = lg["step"].str.startswith(prefix)
    if contains:
        m &= lg["action"].str.contains(contains, regex=False)
    return int(lg.loc[m, "records_affected"].sum())


def synthetic_panel():
    g = generator_facts()
    d_raw, d_clean, _ = load_default()
    personas = g["personas"]
    r_link = d_clean["heel_hours_per_wear"].corr(d_clean["pain_count"])
    r_link2 = d_clean["pain_count"].corr(d_clean["feat_orthopaedic_cushioned_insole"])
    with st.expander("How the survey data was created", expanded=False):
        st.markdown(
            f"<div class='card' style='--c:{ACCENT}'>"
            f"<div class='lbl'>Simulated, not collected</div><p>The built-in dataset is synthetic: "
            f"{g['n'] or len(d_clean):,} respondents generated by <code>01_generate_synthetic_data.py</code> "
            f"(random seed {g['seed']}) so the whole pipeline is ready before the real survey returns.</p>"
            f"<div class='lbl'>{len(personas)} consumer types</div><p>Each respondent is drawn from one of {len(personas)} hidden "
            f"types: {_md(', '.join(personas))}. The type column is not saved, so clustering has to rediscover it.</p>"
            f"<div class='lbl'>Answers are linked</div><p>Answers depend on each other the way they do in real life, e.g. "
            f"more hours in heels leads to more pain (r = {r_link:.2f} between hours per wear and number of pains in the "
            f"clean data) and more pain leads to more interest in the orthopaedic insole (r = {r_link2:.2f}).</p>"
            f"<div class='lbl'>Intent = score + noise</div><p>Purchase intent comes from a scoring formula (heel frequency, "
            f"carrying an extra pair, pains, driving / office-to-party needs, trust, influencers, features wanted, "
            f"awareness, safety concerns, income) plus random noise, then cut into Yes / Maybe / No, so no model can be "
            f"100% accurate.</p>"
            f"<div class='lbl'>Noise added on purpose (counts from the cleaning log)</div><ul>"
            f"<li>{log_count('1.'):,} duplicate submissions removed</li>"
            f"<li>{log_count('2a.'):,} city typos / spelling variants and {log_count('2b.'):,} occupation case errors fixed</li>"
            f"<li>{log_count('2c.'):,} purchase-intent label variants (Y, Nope, Not sure...) mapped</li>"
            f"<li>{log_count('4.'):,} impossible values (ages, hours, negative spend, joke prices, off-scale Likert) set to blank</li>"
            f"<li>{log_count('5.'):,} extreme outliers capped at Q3 + 3×IQR</li>"
            f"<li>{log_count('6.'):,} blank answers filled (original blanks plus the impossible values above)</li></ul>"
            f"<p style='margin-top:8px'>Raw file: {len(d_raw):,} rows → clean file: {len(d_clean):,} rows.</p></div>",
            unsafe_allow_html=True)


# =================================================================== SURVEY & DATA
def before_after(col, label, money=True):
    raw_d = raw.drop_duplicates(subset=["respondent_id"]) if "respondent_id" in raw else raw
    r_all = pd.to_numeric(raw_d[col], errors="coerce")
    r = r_all.dropna()
    c = df[col]
    lo, hi = c.min(), c.max()
    flagged = r[(r > hi) | (r < lo)]
    long = pd.concat([pd.DataFrame({"value": r[r > 0].values, "data": "Raw"}),
                      pd.DataFrame({"value": c[c > 0].values, "data": "Cleaned"})], ignore_index=True)
    f, ax = fig(6.4, 3.0)
    sns.boxplot(data=long, x="value", y="data", order=["Raw", "Cleaned"], hue="data", legend=False,
                palette={"Raw": NEUTRAL_LIGHT, "Cleaned": CAMEL}, width=0.5, fliersize=3, log_scale=True, ax=ax, linecolor=MUTED,
                boxprops=dict(alpha=0.6))
    ax.set_ylabel("")
    fp = flagged[flagged > 0]
    if len(fp):
        ax.scatter(fp, np.zeros(len(fp)), color=RISK, s=30, zorder=5,
                   label=f"{len(flagged)} raw values outside the cleaned range")
        ax.legend(loc="upper left", fontsize=8.5, bbox_to_anchor=(0, 1.18), ncol=1)
    ax.set_xlabel("")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"₹{v:,.0f}" if money else f"{v:,.0f}"))
    ax.set_title(f"{label}: raw vs cleaned (log scale)", pad=26 if len(fp) else 10); ax.grid(axis="y", visible=False)
    show(f)
    return r_all, r, c, flagged, hi


def page_data():
    header(f"{PRODUCT_NAME} · Survey & data", "Survey design, raw data and cleaning",
           "What was asked, what was wrong with the raw answers and what the cleaning changed.")
    tab1, tab2, tab3, tab4 = st.tabs(["Questionnaire", "Raw data quality", "Cleaning: before vs after", "Cleaning log & clean data"])
    with tab1:
        q = pd.DataFrame(QUESTIONS, columns=["Q", "Column", "Question", "Type", "Options / range", "Used for"])
        used = q["Used for"].str.replace(r"\s*\(.*?\)", "", regex=True)      # "REG (target)" counts as REG
        tags = used.str.split(", ").explode()
        tag_name = {"CLS": "Classification", "CLU": "Clustering", "ARM": "Association rules", "REG": "Regression",
                    "EDA": "Descriptive & diagnostic", "Strategy": "Strategy & pricing",
                    "Text analytics": "Open-text analysis (later)"}
        tag_col = {"CLS": FAMILY_COL["Classification"], "CLU": FAMILY_COL["K-Means"], "ARM": FAMILY_COL["Apriori"],
                   "REG": FAMILY_COL["Regression"], "EDA": GREIGE_DARK, "Strategy": GREIGE}
        counts = tags.value_counts()
        qtypes = q["Type"].value_counts()
        multi_use = int((used.str.count(",") > 0).sum())
        kpis([(f"{len(q)}", "Questions in the survey"), (f"{counts.get('CLS', 0)}", "questions used for classification"),
              (f"{counts.get('CLU', 0)}", "questions used for clustering"), (f"{multi_use}", "questions used by more than one analysis")])
        c1, c2 = st.columns(2)
        with c1:
            cs = counts.sort_values()
            f, ax = fig(6.4, max(2.8, 0.5 * len(cs) + 1))
            ax.barh([tag_name.get(t, t) for t in cs.index], cs.values, color=[tag_col.get(t, NEUTRAL) for t in cs.index], height=0.62)
            for i, v in enumerate(cs.values):
                ax.text(v, i, f" {v}", va="center", fontsize=9, fontweight="bold" if i == len(cs) - 1 else "normal")
            ax.set_title("How many questions feed each analysis"); ax.grid(False); ax.set_xticks([]); ax.set_xlim(0, cs.max() * 1.18)
            show(f)
            explain("Each question is tagged with the analyses it is used for (as an input or as the target); a question can be used "
                    "by several, so bars add up to more than the number of questions.",
                    f"**{tag_name.get(counts.index[0], counts.index[0])}** uses the most questions ({counts.iloc[0]} of {len(q)}) and "
                    f"**{tag_name.get(counts.index[-1], counts.index[-1])}** the fewest ({counts.iloc[-1]}). {multi_use} of {len(q)} "
                    "questions are used by more than one analysis.",
                    "Keep the questions that feed several analyses in any shortened survey: dropping one weakens several results at once.")
        with c2:
            pal = [ESPRESSO, CAMEL, "#3E5C76", "#5B6B2F", "#7A4E6D", GREIGE_DARK, "#2F6B66", GREIGE]
            donut(qtypes.values, pal[:len(qtypes)], list(qtypes.index), f"{len(q)}", "questions", w=6.4, h=4.4,
                  title="Answer types in the questionnaire", legend_cols=3)
            explain("Share of questions by answer type.",
                    f"**{qtypes.index[0]}** is the most common type ({qtypes.iloc[0]} of {len(q)}); {int(qtypes.get('Multi-select', 0))} questions "
                    f"are multi-select and {int(qtypes.get('Numeric', 0))} are exact numbers.",
                    "Multi-select answers become 0/1 columns for association rules; ask numbers as exact values, not ranges, so models "
                    "work at individual level (bands are created later only for charts).")
        st.markdown("#### Full questionnaire")
        st.dataframe(q, width="stretch", hide_index=True, height=420)
    with tab2:
        single = [c for c in raw.columns if c not in MULTI_SELECT]      # a blank multi-select means "none ticked"
        kpis([(f"{len(raw):,}", "Raw rows"), (f"{int(raw.duplicated('respondent_id').sum())}", "Duplicate submissions", RISK),
              (f"{int(raw[single].isna().sum().sum()):,}", "Skipped answers (single-answer questions)", RISK)])
        miss = raw[single].isna().mean().sort_values(ascending=False)
        miss = miss[miss > 0]
        c1, c2 = st.columns([1.2, 1])
        with c1:
            if len(miss):
                hbar(miss, "Share of skipped answers per question (raw data)", color=RISK, fmt="{:.1%}")
                explain("The share of raw rows in which each single-answer question was left blank. Multi-select questions are "
                        "excluded: a blank there means nothing was ticked and is read as 'None'.",
                        f"The most-skipped question is **{miss.index[0]}** ({miss.iloc[0]:.1%}); the least-skipped of those "
                        f"with blanks is {miss.index[-1]} ({miss.iloc[-1]:.1%}). {len(miss)} of {len(single)} single-answer "
                        "columns have blanks.",
                        f"With at most {miss.iloc[0]:.1%} missing in any column, gaps are filled (median, or occupation median "
                        f"for income) rather than deleting respondents.")
        with c2:
            st.markdown("**Messy answers found in the raw data**")
            ex = pd.DataFrame({
                "Issue": ["City spelt differently", "Intent label variants", "Impossible age", "Joke WTP"],
                "Examples": [", ".join(sorted(set(raw["city"].dropna().astype(str)) - set(CITIES)))[:90],
                             ", ".join(sorted(set(raw["purchase_intent"].dropna().astype(str)) - {"Yes", "No", "Maybe"})),
                             ", ".join(map(str, sorted(raw.loc[(raw.age < 16) | (raw.age > 75), "age"].dropna().astype(int).unique()))),
                             ", ".join(map(str, sorted(raw.loc[(raw.wtp_base_inr < 300) | (raw.wtp_base_inr > 25000),
                                                               "wtp_base_inr"].dropna().astype(int).unique())))]})
            st.dataframe(ex, hide_index=True, width="stretch")
        st.dataframe(raw.head(50), width="stretch", height=260)
    with tab3:
        c1, c2 = st.columns(2)
        with c1:
            r_all, r, c, flagged, hi = before_after("monthly_income_inr", "Monthly income (Q4)")
            n_blank = int(r_all.isna().sum())
            explain("Box = middle 50% of answers, line = median, dots = outliers. Orange dots (warning colour) are raw values the cleaning "
                    "changed. The x-axis is logarithmic so extreme values stay visible.",
                    f"{len(flagged)} raw incomes were above {inr(hi)} (Q3 + 3×IQR) and were capped there, not deleted; "
                    f"{n_blank} blanks were filled with the median income of the same occupation. The median moved from "
                    f"{inr(r.median())} to {inr(c.median())} and the mean from {inr(r.mean())} to {inr(c.mean())}.",
                    f"Keep all {n:,} respondents; models use the capped income (and log income), so the largest raw value "
                    f"({inr(r.max())}) cannot dominate the regression.")
        with c2:
            r_all, r, c, flagged, hi = before_after("wtp_base_inr", "Willingness to pay (Q27)")
            n_blank = int(r_all.isna().sum())
            explain("Same layout: raw answers vs cleaned answers on a log scale; orange dots are the values the cleaning removed.",
                    f"{len(flagged)} raw answers were outside the cleaned range ({inr(c.min())} to {inr(c.max())}), for example "
                    f"{inr(flagged.min()) if len(flagged) else '-'} and {inr(flagged.max()) if len(flagged) else '-'}; "
                    f"they were set to blank and, with {n_blank} original blanks, filled with the median ({inr(c.median())}). "
                    f"The mean moved from {inr(r.mean())} to {inr(c.mean())}.",
                    "Joke answers would distort the price model and the demand curve, so the cleaned column is the one "
                    "used for regression and pricing.")
    with tab4:
        kpis([(f"{n:,}", f"Clean respondents ({len(raw) - n} rows removed)"),
              (f"{df.shape[1]}", f"Columns after transformation (+{df.shape[1] - raw.shape[1]} engineered)")])
        st.dataframe(log, width="stretch", hide_index=True, height=360)
        st.dataframe(df.head(30), width="stretch", height=260)
        st.download_button("Download clean data (CSV)", df.to_csv(index=False).encode(), f"{FILE_SLUG}_clean_survey_data.csv")


# =================================================================== DESCRIPTIVE
def page_descriptive():
    view = subnav("desc", ["Who they are", "Pains & needs", "Money & channels", "Customer segments"])

    if view == "Who they are":
        frequent = df["heel_frequency"].isin(["Weekly", "Several times a week", "Daily"])
        kpis([(f"{df['age'].median():.0f}", "Median age"), (pct(tier1_share), "Live in Tier-1 cities"),
              (pct(frequent.mean()), "Wear heels weekly or more"), (pct(df["aware_flag"].mean()), "Had heard of convertible heels")])
        c1, c2 = st.columns(2)
        with c1:
            donut(intent_share.values, [INTENT_COL[k] for k in intent_share.index], list(intent_share.index),
                  pct(yes_rate), "say Yes", title="Q29: Would you buy it in the next 6 months?")
            bigger = "larger" if maybe_rate > yes_rate else "smaller"
            explain("Split of the classification label (Yes / Maybe / No).",
                    f"{pct(yes_rate)} say Yes, {pct(maybe_rate)} Maybe and {pct(no_rate)} No; the Maybe group is "
                    f"{bigger} than the Yes group.",
                    f"Converting Maybes (trust, safety proof, try-on) could add up to {pct(maybe_rate)} of the market on top "
                    f"of the {pct(yes_rate)} who already say Yes.")
        with c2:
            f, ax = fig(6.4, 4.2)
            for k in ["Yes", "Maybe", "No"]:
                ax.hist(df.loc[df.purchase_intent == k, "age"], bins=range(16, 68, 3), alpha=0.7, color=INTENT_COL[k], label=k,
                        edgecolor=INK if k == "No" else "none", linewidth=0.5)
            ax.set_title("Age distribution by purchase intent"); ax.set_xlabel("Age"); ax.set_ylabel("Respondents"); ax.legend()
            show(f)
            ay = df.groupby("age_group")["intent_yes"].mean()
            explain("Number of respondents at each age, coloured by their answer.",
                    f"The Yes rate is highest in the **{ay.idxmax()}** group ({pct(ay.max())}) and lowest in "
                    f"**{ay.idxmin()}** ({pct(ay.min())}).",
                    f"Aim launch creative at the {ay.idxmax()} age band; treat {ay.idxmin()} as a later, comfort-led audience.")
        c1, c2 = st.columns(2)
        with c1:
            g = df.groupby("occupation")["intent_yes"].mean()
            hbar(g, "Share answering Yes, by occupation")
            explain("Yes rate within each occupation.",
                    f"**{g.idxmax()}** convert best ({pct(g.max())}); **{g.idxmin()}** the least ({pct(g.min())}), a gap of "
                    f"{(g.max() - g.min()) * 100:.0f} points.",
                    f"Lead launch messaging with {g.idxmax()} use cases rather than a message for everyone.")
        with c2:
            vc = df["heel_frequency"].value_counts(normalize=True).reindex(HEEL_FREQ).fillna(0)
            f, ax = fig(6.4, 3.6)
            freq_levels = ("Weekly", "Several times a week", "Daily")
            ax.bar(range(6), vc.values, color=[MARK if h in freq_levels else NEUTRAL_LIGHT for h in HEEL_FREQ], width=0.62)
            ax.set_xticks(range(6), ["Never", "Rarely", "Monthly", "Weekly", "Several/wk", "Daily"])
            for i, v in enumerate(vc.values):
                ax.text(i, v, f"{v:.0%}", ha="center", va="bottom", fontsize=9)
            ax.set_title("Q8: How often do you wear heels?"); ax.grid(False); ax.set_yticks([])
            show(f)
            y_f, y_r = df.loc[frequent, "intent_yes"].mean(), df.loc[~frequent, "intent_yes"].mean()
            explain("Share of respondents at each heel-wearing frequency (espresso = weekly or more).",
                    f"{pct(frequent.mean())} wear heels weekly or more. Their Yes rate is {pct(y_f)} versus {pct(y_r)} for "
                    f"less frequent wearers.",
                    "Frequent wearers are the beachhead market." if y_f > y_r else
                    "Frequency does not lift intent here, so do not target on frequency alone.")
        c1, c2 = st.columns(2)
        ct = df["city"].value_counts()
        with c1:
            hbar(ct, "Respondents by city", color=NEUTRAL, fmt="{:.0f}")
            explain("Number of respondents per city.",
                    f"**{ct.index[0]}** has the most respondents ({ct.iloc[0]:,}, {pct(ct.iloc[0] / n)}); "
                    f"**{ct.index[-1]}** the fewest ({ct.iloc[-1]:,}).",
                    "City-level results are reliable only where the sample is large; pool small cities by tier.")
        with c2:
            donut([tier1_share, 1 - tier1_share], [MARK, NEUTRAL_LIGHT], ["Tier 1", "Tier 2"], pct(tier1_share),
                  "Tier-1 metros", h=max(2.4, 0.38 * len(ct) + 0.8), title="City tier mix")
            explain("Share of respondents living in Tier-1 versus Tier-2 cities.",
                    f"Tier-1 cities make up {pct(tier1_share)} of respondents and Tier-2 {pct(1 - tier1_share)}.",
                    "Results mainly describe metro women; validate demand in Tier-2 cities before expanding there."
                    if tier1_share > 0.5 else "The sample is balanced across tiers, so results apply to both.")

    elif view == "Pains & needs":
        p = df[A.PAIN_COLS].mean(); p.index = PAINS
        s = df[A.SIT_COLS].mean(); s.index = SITUATIONS
        ft = df[A.FEAT_COLS].mean(); ft.index = FEATURES
        cc = df[A.CONCERN_COLS].mean(); cc.index = CONCERNS
        ccs = cc.sort_values(ascending=False)
        kpis([(pct((df["pain_count"] > 0).mean()), "Report at least one pain"),
              (f"{df['pain_severity'].mean():.1f} / 10", "Average pain severity"),
              (pct(s.max()), f"Switch footwear: {s.idxmax()}"), (pct(cc.max()), f"Worry: {cc.idxmax()}", RISK)])
        c1, c2 = st.columns(2)
        with c1:
            hbar(p, "Q15: Pain experienced after wearing heels")
            explain("Share of respondents who reported each problem (multi-select).",
                    f"**{p.idxmax()}** is the most common ({pct(p.max())}); {pct((df['pain_count'] > 0).mean())} report at "
                    f"least one problem.",
                    "Use the most common pain in the comfort claim and keep the orthopaedic insole in the launch shortlist.")
        with c2:
            hbar(s, "Q18: When do you need to switch heels and flats?")
            s2 = s.drop(s.idxmax())
            explain("The situations in which women change footwear (multi-select).",
                    f"**{s.idxmax()}** ({pct(s.max())}) and **{s2.idxmax()}** ({pct(s2.max())}) lead.",
                    f"Use {s.idxmax()} and {s2.idxmax()} as the scenes in ads and product photos.")
        c1, c2 = st.columns(2)
        with c1:
            hbar(ft, "Q23: Features that interest respondents")
            explain("Share of respondents interested in each product feature.",
                    f"Top 3: **{', '.join(ft.sort_values(ascending=False).index[:3])}**. Lowest: {ft.idxmin()} ({pct(ft.min())}).",
                    "This is the launch shortlist; association rules (Predictive) and the roadmap (Prescriptive) pick the bundle.")
        with c2:
            hbar(cc, "Q24: What would worry you about buying it?", color=RISK)
            explain("Share of respondents who ticked each purchase barrier.",
                    f"**{ccs.index[0]}** worries {pct(ccs.iloc[0])}, {(ccs.iloc[0] - ccs.iloc[1]) * 100:.0f} points more than "
                    f"the next concern ({ccs.index[1]}, {pct(ccs.iloc[1])}).",
                    f"Answer '{ccs.index[0]}' with proof before launch, not claims.")
        heels = df[[f"heel_{slug(h)}" for h in HEEL_TYPES]].mean(); heels.index = HEEL_TYPES
        mods = ["Kitten", "Stiletto", "Block", "Wedge"]
        mcols = [f"feat_{slug(m + ' heel module')}" for m in mods]
        by_int = df.groupby("purchase_intent")[mcols].mean().T.reindex(columns=["Yes", "Maybe", "No"])
        by_int.index = mods
        c1, c2 = st.columns(2)
        with c1:
            hbar(heels, "Q14: Heel types respondents like wearing")
            explain("Share of respondents who like wearing each heel type (multi-select).",
                    f"**{heels.idxmax()}** is the most liked heel type ({pct(heels.max())}) and **{heels.idxmin()}** the least "
                    f"({pct(heels.min())}).",
                    f"Make sure the first heel set includes {heels.idxmax()}; {heels.idxmin()} is a later option.")
        with c2:
            f, ax = fig(6.4, 3.6)
            x = np.arange(len(mods)); wdt = 0.26
            for i, k in enumerate(["Yes", "Maybe", "No"]):
                ax.bar(x + (i - 1) * wdt, by_int[k], wdt - 0.02, color=INTENT_COL[k], label=k,
                       edgecolor=INK if k == "No" else "none", linewidth=0.6)
            ax.set_xticks(x, [f"{m} heel" for m in mods]); ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
            ax.legend(ncol=3, loc="upper right"); ax.set_title("Q23: Heel modules wanted, by purchase intent"); ax.grid(axis="x", visible=False)
            show(f)
            gap = (by_int["Yes"] - by_int["No"]).sort_values(ascending=False)
            order = list(by_int["Yes"].sort_values(ascending=False).index)
            explain("Share of each intent group that wants each heel module.",
                    f"Among Yes-buyers the **{order[0]}** heel is wanted most ({pct(by_int.loc[order[0], 'Yes'])}); the module where "
                    f"Yes-buyers lead No-buyers by the most is **{gap.index[0]}** ({gap.iloc[0] * 100:.0f} points).",
                    f"Order the heel moulds by Yes-buyer demand: {', '.join(order)}.")

    elif view == "Money & channels":
        w = interested["wtp_base_inr"]
        xh = interested["wtp_extra_heel_inr"]
        ch = df["preferred_channel"].value_counts(normalize=True)
        afford = (w >= ENTRY_PRICE).mean()
        kpis([(inr(w.median()), "Median WTP (Yes + Maybe)"), (inr(w.quantile(.75)), "Top 25% would pay above"),
              (inr(xh.median()), "Median WTP per extra heel"), (pct(ch.max()), f"Buy footwear on {ch.idxmax()}")])
        c1, c2 = st.columns(2)
        with c1:
            f, ax = fig(6.4, 3.9)
            ax.hist(w, bins=30, color=MARK, alpha=0.85, edgecolor="white")
            ax.axvline(w.median(), color=INK, ls="--", lw=1.2)
            ax.text(w.median(), ax.get_ylim()[1] * 0.92, f"  median {inr(w.median())}", fontsize=9, fontweight="bold")
            ax.axvline(ENTRY_PRICE, color=NEUTRAL, ls=":", lw=1.2)
            ax.text(ENTRY_PRICE, ax.get_ylim()[1] * 0.80, f" entry kit {inr(ENTRY_PRICE)}", fontsize=8.5, color=MUTED)
            ax.set_title("Q27: Willingness to pay (Yes + Maybe)"); ax.set_xlabel("₹ for sandal + 1 heel pair"); ax.set_yticks([])
            show(f)
            explain("How many interested respondents would pay each price for the sandal with one heel pair.",
                    f"Half would pay {inr(w.median())} or more and a quarter more than {inr(w.quantile(.75))}; "
                    f"{pct(afford)} would pay at least the planned entry price of {inr(ENTRY_PRICE)}.",
                    f"An entry kit at {inr(ENTRY_PRICE)} is within reach of {pct(afford)} of interested buyers; the Prescriptive "
                    "section finds the revenue-maximising price.")
        with c2:
            hbar(ch, "Q13: Where do you buy footwear?", h=3.9)
            bw = ch.get("Brand website / Instagram", 0)
            explain("Preferred shopping channel (share of respondents).",
                    f"**{ch.idxmax()}** leads with {pct(ch.max())}; brand website / Instagram gets {pct(bw)}.",
                    f"Use {ch.idxmax()} for reach and the brand website / Instagram for margin and storytelling.")
        with st.expander("Summary statistics (numeric questions)"):
            cols = ["age", "monthly_income_inr", "hours_on_feet", "heel_hours_per_wear", "heel_pairs_owned",
                    "annual_footwear_spend_inr", "pain_severity", "carry_extra_pair", "wtp_base_inr", "wtp_extra_heel_inr"]
            st.dataframe(df[cols].describe().T[["mean", "50%", "std", "min", "max"]].round(1), width="stretch")

    else:
        K = cached_kmeans(df, K_CHOSEN)
        prof = K["profile"].sort_values("Yes_pct", ascending=False)
        st.caption(f"{K_CHOSEN} segments from K-Means on behaviour and profile only (purchase intent is not an input). "
                   "How they were built and checked: Predictive → Clustering.")
        persona_cards(prof)
        f, ax = fig(9, max(2.6, 0.6 * len(prof) + 1))
        nm = seg_names(prof)
        labels = [nm[c] for c in prof.index][::-1]
        ys, ms_ = prof["Yes_pct"].values[::-1], prof["Maybe_pct"].values[::-1]
        ax.barh(labels, ys, color=YES, height=0.6, label="Yes")
        ax.barh(labels, ms_, left=ys, color=MAYBE, height=0.6, label="Maybe", edgecolor="white", linewidth=2)
        for i, (y, m) in enumerate(zip(ys, ms_)):
            if y > 0.06:
                ax.text(y / 2, i, f"{y:.0%}", ha="center", va="center", color="white", fontweight="bold", fontsize=9)
            if m > 0.06:
                ax.text(y + m / 2, i, f"{m:.0%}", ha="center", va="center", color=INTENT_TXT["Maybe"], fontsize=9)
        for t, c in zip(ax.get_yticklabels(), prof.index[::-1]):
            t.set_color(seg_col(c)); t.set_fontweight("bold")
        ax.set_xlim(0, 1); ax.set_xticks([]); ax.legend(loc="lower right", ncol=2)
        ax.set_title("Yes and Maybe share inside each segment"); ax.grid(False)
        show(f)
        top, bottom = prof.iloc[0], prof.iloc[-1]
        explain("Each bar is one segment; espresso = Yes, camel = Maybe. Segment names are coloured the same as on every other chart.",
                f"**{top['Name']}** ({pct(top['Share_pct'])} of respondents) has the highest Yes rate ({pct(top['Yes_pct'])}, "
                f"median WTP {inr(top['Median_WTP'])}); **{bottom['Name']}** the lowest ({pct(bottom['Yes_pct'])}).",
                f"Put the launch budget behind {short(top['Name'])}; reach {short(bottom['Name'])} organically later.")


def persona_cards(prof):
    for i in range(0, len(prof), 4):
        chunk = prof.iloc[i:i + 4]
        cols = st.columns(4)
        for col, (c, r) in zip(cols, chunk.iterrows()):
            name, occ = (r["Name"].split(" (") + [""])[:2]
            sc = seg_col(c)
            col.markdown(
                f"<div class='card' style='--c:{sc}'><div class='h' style='color:{sc}'>{_md(name)}</div>"
                f"<div class='sub'>Most common: {_md(occ.rstrip(')'))} · cluster {c}</div>"
                f"<span class='badge' style='--c:{MUTED}'>{r['Conversion_tier']} conversion</span>"
                + bar_html("Share of respondents", r["Share_pct"], sc)
                + bar_html("Say Yes", r["Yes_pct"], YES)
                + f"<div class='row2'><span>Median WTP</span><b>{inr(r['Median_WTP'])}</b></div>"
                f"<div class='row2'><span>Average age</span><b>{r['Avg_age']:.0f}</b></div>"
                f"<div class='row2'><span>Drives to work</span><b>{pct(r['Drives_pct'])}</b></div></div>",
                unsafe_allow_html=True)


# =================================================================== DIAGNOSTIC
def page_diagnostic():
    view = subnav("diag", ["Drivers of intent", "Behaviour vs intent", "Price & income", "Significance tests", "Why Maybe?"])
    num = ["intent_score", "wtp_base_inr", "age", "log_income", "heel_freq_score", "heel_hours_per_wear",
           "heel_pairs_owned", "pain_count", "pain_severity", "carry_extra_pair", "sit_count", "feat_count",
           "importance_comfort", "importance_style", "trust_d2c_brands", "influencer_influence", "drives_self",
           "concern_count", "hours_on_feet"]

    if view == "Drivers of intent":
        corr = df[num].corr()
        ci = corr["intent_score"].drop(["intent_score", "wtp_base_inr"]).sort_values()
        cw = corr["wtp_base_inr"].drop(["wtp_base_inr", "intent_score"]).sort_values()
        kpis([(pretty(ci.index[-1]), f"Strongest driver of intent (r = {ci.iloc[-1]:.2f})"),
              (pretty(ci.index[-2]), f"Second driver of intent (r = {ci.iloc[-2]:.2f})"),
              (pretty(cw.index[-1]), f"Strongest driver of price (r = {cw.iloc[-1]:.2f})")])
        c1, c2 = st.columns(2)
        with c1:
            f, ax = fig(6.4, 5.2)
            ax.barh([pretty(i) for i in ci.index], ci.values, color=[MARK if v > 0 else NEUTRAL for v in ci.values], height=0.62)
            ax.axvline(0, color=MUTED, lw=0.8); ax.set_title("Correlation with purchase intent"); ax.grid(axis="y", visible=False)
            show(f)
            neg = ci[ci < 0]
            explain("Pearson correlation of each answer with intent (No = 0, Maybe = 1, Yes = 2). Espresso = moves with intent, "
                    "greige = moves against it.",
                    f"Intent rises most with **{pretty(ci.index[-1])}** (r = {ci.iloc[-1]:.2f}) and **{pretty(ci.index[-2])}** "
                    f"(r = {ci.iloc[-2]:.2f}). " + (f"The most negative is {pretty(neg.index[0])} (r = {neg.iloc[0]:.2f})."
                                                     if len(neg) else "No answer is negatively correlated."),
                    f"Use {pretty(ci.index[-1])} and {pretty(ci.index[-2])} to qualify leads and choose ad audiences.")
        with c2:
            f, ax = fig(6.4, 5.2)
            ax.barh([pretty(i) for i in cw.index], cw.values, color=[MARK if v > 0 else NEUTRAL for v in cw.values], height=0.62)
            ax.axvline(0, color=MUTED, lw=0.8); ax.set_title("Correlation with willingness to pay"); ax.grid(axis="y", visible=False)
            show(f)
            r_inc_i, r_inc_w = corr.loc["intent_score", "log_income"], corr.loc["wtp_base_inr", "log_income"]
            explain("Pearson correlation of each answer with willingness to pay (₹).",
                    f"Price is driven most by **{pretty(cw.index[-1])}** (r = {cw.iloc[-1]:.2f}). Income correlates "
                    f"r = {r_inc_w:.2f} with price but r = {r_inc_i:.2f} with intent.",
                    "Target by behaviour, price by income." if r_inc_w > r_inc_i else
                    "Income matters as much for intent as for price, so include it in targeting too.")
        f, ax = plt.subplots(figsize=(10, 7.5))
        sns.heatmap(corr, cmap=CORR_CMAP, vmin=-1, vmax=1, center=0, annot=True, fmt=".2f", annot_kws={"size": 7},
                    linewidths=1, linecolor="white", cbar_kws={"shrink": .7}, ax=ax,
                    xticklabels=[pretty(c) for c in corr.columns], yticklabels=[pretty(c) for c in corr.index])
        ax.set_title("Correlation matrix (Pearson)"); ax.grid(False)
        show(f)
        inputs = corr.drop(index=["intent_score", "wtp_base_inr"], columns=["intent_score", "wtp_base_inr"])
        up_t = inputs.where(np.triu(np.ones(inputs.shape, dtype=bool), 1)).stack()
        pair = up_t.abs().idxmax()
        strong = int((up_t.abs() >= 0.5).sum())
        explain("Every pair of numeric answers: +1 = always rise together, -1 = opposite, 0 = no linear link. Espresso = positive, slate blue = negative.",
                f"The strongest link between two inputs is **{pretty(pair[0])}** and **{pretty(pair[1])}** "
                f"(r = {up_t[pair]:.2f}); {strong} input pairs have |r| ≥ 0.5.",
                "Correlated inputs make single regression coefficients unstable, which is why Ridge and Lasso are compared "
                "with plain linear regression." if strong else
                "Inputs are not strongly correlated with each other, so linear coefficients can be read one at a time.")

    elif view == "Behaviour vs intent":
        c1, c2 = st.columns(2)
        with c1:
            g = yes_rate_bar(df.assign(carry=df.carry_extra_pair.astype(int).map(dict(enumerate(CARRY_FREQ, 1)))),
                             "carry", "Intent by 'How often do you carry an extra pair?'", order=CARRY_FREQ)
            explain("Answer mix (Yes / Maybe / No) for each level of Q17.",
                    f"The Yes share goes from {pct(g['Yes'].iloc[0])} ({g.index[0]}) to {pct(g['Yes'].iloc[-1])} "
                    f"({g.index[-1]}) and {trend_word(g['Yes'])}.",
                    "Women who already carry a second pair are the first customers and a ready-made ad audience."
                    if g["Yes"].iloc[-1] > g["Yes"].iloc[0] else "Carrying a spare pair does not lift intent; do not target on it.")
        with c2:
            g = yes_rate_bar(df, "heel_frequency", "Intent by heel-wearing frequency", order=HEEL_FREQ)
            explain("Answer mix for each heel-frequency level (Q8).",
                    f"The Yes share goes from {pct(g['Yes'].iloc[0])} ({g.index[0]}) to {pct(g['Yes'].iloc[-1])} "
                    f"({g.index[-1]}) and {trend_word(g['Yes'])}.",
                    "Start the go-to-market with heavy heel users." if g["Yes"].iloc[-1] > g["Yes"].iloc[0]
                    else "Frequency alone does not predict buying.")
        c1, c2 = st.columns(2)
        with c1:
            g = yes_rate_bar(df, "commute_mode", "Intent by commute mode")
            explain("Answer mix by how she usually travels (Q6).",
                    f"**{g['Yes'].idxmax()}** has the highest Yes share ({pct(g['Yes'].max())}); **{g['Yes'].idxmin()}** "
                    f"the lowest ({pct(g['Yes'].min())}).",
                    f"Build a campaign around the {g['Yes'].idxmax()} use case.")
        with c2:
            g = yes_rate_bar(df, "occupation", "Intent by occupation")
            explain("Answer mix by occupation (Q3).",
                    f"**{g['Yes'].idxmax()}** have the highest Yes share ({pct(g['Yes'].max())}) and **{g['No'].idxmax()}** "
                    f"the highest No share ({pct(g['No'].max())}).",
                    f"Prioritise {g['Yes'].idxmax()}; deprioritise {g['No'].idxmax()} at launch.")

    elif view == "Price & income":
        c1, c2 = st.columns(2)
        with c1:
            f, ax = fig(6.4, 4)
            order = [o for o in ["<25k", "25-50k", "50k-1L", "1-2L", "2L+"] if o in df.income_band.unique()]
            sns.boxplot(data=df, x="income_band", y="wtp_base_inr", order=order, color=CAMEL, fliersize=2, width=0.55,
                        ax=ax, boxprops=dict(alpha=.55))
            ax.set_title("Willingness to pay by monthly income"); ax.set_xlabel("Monthly income (₹)"); ax.set_ylabel("₹")
            show(f)
            med = df.groupby("income_band")["wtp_base_inr"].median().reindex(order)
            explain("Spread of willingness to pay inside each income band (box = middle 50%, line = median).",
                    f"Median WTP is lowest in the {med.idxmin()} band ({inr(med.min())}) and highest in the {med.idxmax()} "
                    f"band ({inr(med.max())}); across bands it {trend_word(med.values)}.",
                    "Offer a tiered range: an entry line for low-income buyers and a premium line for high-income buyers.")
        with c2:
            f, ax = fig(6.4, 4)
            sns.boxplot(data=df, x="purchase_intent", y="wtp_base_inr", order=["Yes", "Maybe", "No"], hue="purchase_intent",
                        legend=False, palette=INTENT_COL, fliersize=2, width=0.55, ax=ax, boxprops=dict(alpha=.75))
            ax.set_title("Willingness to pay by purchase intent"); ax.set_xlabel(""); ax.set_ylabel("₹")
            show(f)
            groups = [df.loc[df.purchase_intent == k, "wtp_base_inr"] for k in ["Yes", "Maybe", "No"]]
            F, pv = stats.f_oneway(*groups)
            meds = [g_.median() for g_ in groups]
            ordered = meds[0] > meds[1] > meds[2]
            explain("Willingness to pay for each intent group, with a one-way ANOVA test of equal means.",
                    f"Median: Yes {inr(meds[0])}, Maybe {inr(meds[1])}, No {inr(meds[2])}. ANOVA F = {F:.0f}, p = {pv:.1e}: "
                    + ("the difference is significant at 5%." if pv < 0.05 else "the difference is not significant at 5%."),
                    "Buyers who want it also value it more, so the Yes group does not need discounts." if ordered and pv < 0.05
                    else "Price sensitivity does not follow intent cleanly; test prices with each group.")

    elif view == "Significance tests":
        tests = []
        for col, label in [("occupation", "Occupation"), ("commute_mode", "Commute mode"), ("city_tier", "City tier"),
                           ("heel_frequency", "Heel frequency"), ("age_group", "Age group"), ("income_band", "Income band"),
                           ("marital_status", "Marital status"), ("aware_of_convertible_heels", "Aware before"),
                           ("preferred_channel", "Preferred channel")]:
            tab = pd.crosstab(df[col], df["purchase_intent"])
            chi2, p, dof, _ = stats.chi2_contingency(tab)
            v = np.sqrt(chi2 / (tab.values.sum() * (min(tab.shape) - 1)))
            tests.append({"Variable": label, "Chi²": round(chi2, 1), "p-value": p, "Cramér's V (strength)": round(v, 3),
                          "Significant at 5%?": "Yes" if p < 0.05 else "No"})
        tests = pd.DataFrame(tests).sort_values("Cramér's V (strength)", ascending=True)
        vcol = "Cramér's V (strength)"
        f, ax = fig(9, 4.2)
        ax.barh(tests["Variable"], tests[vcol], color=[MARK if s_ == "Yes" else NEUTRAL_LIGHT for s_ in tests["Significant at 5%?"]],
                height=0.62)
        for i, v in enumerate(tests[vcol]):
            ax.text(v, i, f" {v:.2f}", va="center", fontsize=9)
        for x_, lab in [(0.1, "weak"), (0.3, "moderate")]:
            ax.axvline(x_, color=NEUTRAL, ls=":", lw=1); ax.text(x_, len(tests) - 0.4, f" {lab}", fontsize=8, color=MUTED)
        ax.set_title("Strength of link with purchase intent (Cramér's V; espresso = significant at 5%)"); ax.grid(False)
        show(f)
        top_t = tests.iloc[-1]
        top_v = top_t[vcol]
        not_sig = ", ".join(tests.loc[tests["Significant at 5%?"] == "No", "Variable"]) or "none"
        explain("Chi-square tests whether a category and intent are independent; Cramér's V measures strength "
                "(about 0.1 weak, 0.3 moderate, 0.5 strong). Greige bars are not significant.",
                f"Strongest link: **{top_t['Variable']}** (V = {top_v:.2f}). Not significant at 5%: {not_sig}.",
                "Use only significant variables for targeting decisions, even if a chart looks different.")
        with st.expander("Test results table"):
            t2 = tests.sort_values(vcol, ascending=False).copy()
            t2["p-value"] = t2["p-value"].map(lambda x: f"{x:.2e}")
            st.dataframe(t2, hide_index=True, width="stretch")

    else:
        comp = pd.DataFrame({k: df.loc[df.purchase_intent == k, A.CONCERN_COLS].mean().values for k in ["Yes", "Maybe", "No"]},
                            index=CONCERNS)
        gap = (comp["Maybe"] - comp["Yes"]).sort_values(ascending=False)
        c1, c2 = st.columns([1.6, 1])
        with c1:
            f, ax = fig(8, 4)
            x = np.arange(len(comp)); wdt = 0.26
            for i, k in enumerate(["Yes", "Maybe", "No"]):
                ax.bar(x + (i - 1) * wdt, comp[k], wdt - 0.02, color=INTENT_COL[k], label=k,
                       edgecolor=INK if k == "No" else "none", linewidth=0.6)
            ax.set_xticks(x, [c.replace(" / ", "/\n").replace("Mechanism too complicated", "Too\ncomplicated") for c in comp.index],
                          fontsize=8.5)
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
            ax.legend(); ax.set_title("Concerns by intent group"); ax.grid(axis="x", visible=False)
            show(f)
        with c2:
            st.markdown("**What would unlock the Maybes**")
            for i, c in enumerate(gap.index[:2]):
                word = "more" if gap.iloc[i] > 0 else "fewer"
                st.markdown(f"<div class='card' style='--c:{RISK}'><div class='h'>{i + 1}. {_md(c)}</div>"
                            f"<div class='sub'>{abs(gap.iloc[i]) * 100:.0f} points {word} Maybes than Yes-buyers worry about this</div>"
                            f"<p>Fix: {_md(REMEDY.get(c, '-'))}</p></div>", unsafe_allow_html=True)
        explain("Share of each intent group that ticked each concern.",
                f"Compared with Yes-buyers, Maybes worry most about **{gap.index[0]}** ({gap.iloc[0] * 100:+.0f} points) and "
                f"**{gap.index[1]}** ({gap.iloc[1] * 100:+.0f} points).",
                "Conversion plan for Maybes: " + "; ".join(REMEDY.get(c, c) for c in gap.index[:2]) + ".")


REMEDY = {"Heel detaching / Safety": "safety demo videos, a load-test certificate and a heel-lock warranty",
          "Price": "an entry kit, launch offer and no-cost EMI",
          "Limited designs": "more uppers and colours at launch and user-generated styling reels",
          "Durability": "a warranty and replaceable heel tips",
          "Sizing / Fit": "a size guide, half sizes and free exchanges",
          "Mechanism too complicated": "a short click-on, click-off demo and try-on pop-ups"}


# =================================================================== PREDICTIVE
FAMILY_COL = {"Classification": ESPRESSO, "Regression": CAMEL, "K-Means": "#3E5C76", "Latent Class": "#7A4E6D",
              "Apriori": "#5B6B2F"}


def col_groups(cols):
    """Split input columns into plain columns and flag groups (exact lists from analytics.py)."""
    flag_sets = [(set(A.PAIN_COLS), "Q15 pain flags"), (set(A.SIT_COLS), "Q18 situation flags"),
                 (set(A.CONCERN_COLS), "Q24 concern flags"), (set(A.FEAT_COLS), "Q23 feature flags")]
    groups, plain = {}, []
    for c in cols:
        label = next((l for s_, l in flag_sets if c in s_), None)
        if label is None and c.startswith("occ_"):
            label = "occupation dummies (Q3)"
        elif label is None and c.startswith("commute_"):
            label = "commute dummies (Q6)"
        if label:
            groups[label] = groups.get(label, 0) + 1
        else:
            plain.append(c)
    return plain, groups


def col_summary(cols):
    plain, groups = col_groups(cols)
    parts = ([", ".join(plain)] if plain else []) + [f"{v} {k}" for k, v in groups.items()]
    return f"{len(cols)} columns: " + " + ".join(parts)


def tint(hex_color, strength=0.14):
    """Light tint of a colour (blend with white), as a plain hex value that every browser understands."""
    h = hex_color.lstrip("#")
    return "#%02x%02x%02x" % tuple(round(255 - (255 - int(h[i:i + 2], 16)) * strength) for i in (0, 2, 4))


def chips(items, color):
    return "".join(f"<span class='chip' style='--c:{color};background:{tint(color)}'>{_md(i)}</span>" for i in items)


def algorithm_map():
    st.caption("Generated from the feature lists in analytics.py, so it always matches what the models really use.")
    clf_cols = list(A.model_matrix(df.head(5)).columns)
    reg_cols = cached_regression(df)["columns"]
    lca_cols = list(A.lca_inputs(df.head(50)).columns)
    basket = list(A.baskets(df.head(5), ("features",)).columns)
    n_clf, n_reg, n_km, n_lca, n_ap = len(clf_cols), len(reg_cols), len(A.CLUSTER_VARS), len(lca_cols), len(basket)
    kpis([(f"{n_clf}", "inputs to the classifiers", FAMILY_COL["Classification"]), (f"{n_reg}", "inputs to the regressions", FAMILY_COL["Regression"]),
          (f"{n_km}", "inputs to K-Means", FAMILY_COL["K-Means"]), (f"{n_ap}", "Q23 items in the default rule basket", FAMILY_COL["Apriori"])])
    names = ["Classification", "Regression", "K-Means", "Latent Class", "Apriori"]
    counts_ = [n_clf, n_reg, n_km, n_lca, n_ap]
    f, ax = fig(9, 3.2)
    ax.barh(names[::-1], counts_[::-1], color=[FAMILY_COL[k] for k in names[::-1]], height=0.6)
    for i, v in enumerate(counts_[::-1]):
        ax.text(v, i, f" {v}", va="center", fontsize=9.5, fontweight="bold")
    ax.set_title("Number of input columns used by each algorithm family"); ax.grid(False); ax.set_xticks([]); ax.set_xlim(0, max(counts_) * 1.12)
    show(f)
    extras = [c for c in reg_cols if c not in clf_cols]
    leak = [c for c in clf_cols if c.startswith("wtp")]
    explain("Each bar counts the input columns an algorithm family receives; the target column is never counted.",
            f"**{names[int(np.argmax(counts_))]}** uses the most inputs ({max(counts_)}) and **{names[int(np.argmin(counts_))]}** the fewest "
            f"({min(counts_)}). Regression adds {len(extras)} inputs to the classifier's set ({', '.join(pretty(c) for c in extras[:3])}"
            f"{', ...' if len(extras) > 3 else ''}).",
            ("No willingness-to-pay column is an input to the classifier, so the buy / not-buy prediction does not use information "
             "a new lead would not have." if not leak else f"Check for leakage: {', '.join(leak)} is an input to the classifier."))

    fam = [
        ("Classification", "KNN · Decision Tree · Random Forest · Gradient Boosting", n_clf, col_groups(clf_cols),
         "purchase_intent (Q29: Yes / Maybe / No); business view uses intent_yes", "Class and probability per lead; confusion matrix; ROC"),
        ("Regression", "Linear · Ridge · Lasso", n_reg, col_groups(reg_cols), "wtp_base_inr (Q27, ₹)",
         "Predicted ₹ willingness to pay; coefficient per input"),
        ("K-Means", "Clustering", n_km, ([], {}), "None (unsupervised); intent only profiles the segments afterwards",
         "Segment label per respondent and a segment profile"),
        ("Latent Class", "Latent Class Analysis", n_lca, ([], {}), "None (unsupervised)", "Probability of belonging to each class"),
        ("Apriori", "Association rules", n_ap, ([], {}), "None (finds co-occurrence)",
         "If-then rules with support, confidence, lift, conviction"),
    ]
    inputs_ = {"K-Means": list(A.CLUSTER_VARS), "Latent Class": lca_cols, "Apriori": basket}
    fcols = st.columns(3)
    for i, (name, algos, n_in, (plain, groups), target, out) in enumerate(fam):
        col_ = FAMILY_COL[name]
        if name in inputs_:
            items = [pretty(c) for c in inputs_[name]]
            note = "Default basket: Q23 feature flags only (likely buyers)." if name == "Apriori" else ""
        else:
            items = ([f"{len(plain)} numeric / ordinal answers"] if plain else []) + [f"{v} {k}" for k, v in groups.items()]
            note = ""
        fcols[i % 3].markdown(
            f"<div class='card fam' style='--c:{col_}'><div class='h' style='color:{INK}'>{_md(name)}</div>"
            f"<div class='sub'>{_md(algos)}</div><div class='big'>{n_in} <span style='font-size:.85rem;font-weight:600;color:{MUTED}'>inputs</span></div>"
            f"<div style='margin:6px 0'>{chips(items, col_)}</div>"
            + (f"<p style='font-size:.82rem;color:{MUTED}'>{_md(note)}</p>" if note else "")
            + f"<div class='lbl'>Target</div><p>{_md(target)}</p><div class='lbl'>Output</div><p>{_md(out)}</p></div>",
            unsafe_allow_html=True)
    st.markdown("#### Algorithm to columns table")
    rows = [
        ["Classification: KNN, Decision Tree, Random Forest, Gradient Boosting", col_summary(clf_cols),
         "purchase_intent (Q29: Yes / Maybe / No)", "Predicted class + probability per class; confusion matrix"],
        ["Classification, business view (same 4 models)", f"Same {len(clf_cols)} columns as above", "intent_yes (1 if Q29 = Yes)",
         "P(Yes) per lead; TP / FP / FN / TN; ROC curve"],
        ["Regression: Linear, Ridge, Lasso", col_summary(reg_cols), "wtp_base_inr (Q27, ₹)",
         "Predicted ₹ willingness to pay; coefficient per input"],
        ["Clustering: K-Means", f"{len(A.CLUSTER_VARS)} columns (standardised): " + ", ".join(A.CLUSTER_VARS),
         "None (unsupervised); intent is used only afterwards to profile segments", "Segment label per respondent + segment profile"],
        ["Clustering: Latent Class Analysis", f"{len(lca_cols)} categorical columns (numeric answers binned): " + ", ".join(lca_cols),
         "None (unsupervised)", "Probability of belonging to each class"],
        ["Association rules: Apriori", f"Default: {len(basket)} Q23 feature flags only (" + ", ".join(basket) + ") for likely "
         "buyers (Yes + Maybe). Pains, situations and concerns can be added in Rule settings.",
         "None (finds co-occurrence)", "If-then rules with support, confidence, lift, conviction"],
        ["Prospect scorer", "Same inputs as each model above", "-",
         f"Gradient Boosting P(Yes), Lasso ₹ WTP, K-Means segment (k = {K_CHOSEN})"],
    ]
    t = pd.DataFrame(rows, columns=["Algorithm family", "Input columns", "Target column", "Output"])
    st.table(t.set_index("Algorithm family"))   # static table: long column lists wrap instead of being cut off
    with st.expander("Exact input column lists"):
        for name, cols in [("Classification", clf_cols), ("Regression", reg_cols), ("K-Means", A.CLUSTER_VARS),
                           ("Latent Class Analysis", lca_cols), ("Apriori (default basket)", basket)]:
            st.markdown(f"**{name}** ({len(cols)}): " + ", ".join(f"`{c}`" for c in cols))


def page_predictive():
    view = subnav("pred", ["Algorithm map", "Classification", "Regression", "Clustering", "Association rules", "Score a prospect"])
    if view == "Algorithm map":
        algorithm_map()
    elif view == "Classification":
        classification_view()
    elif view == "Regression":
        regression_view()
    elif view == "Clustering":
        clustering_view()
    elif view == "Association rules":
        rules_view()
    else:
        prospect_view()


def classification_view():
    C = cached_classification(df)
    res = C["results"].copy()
    ri = res.set_index("Model")
    best = C["best"]
    br = ri.loc[best]
    rank = {m: i + 1 for i, m in enumerate(res.sort_values("Accuracy", ascending=False)["Model"])}
    mods = A.classification_models()
    knn, dt, rf, gb = (mods["K-Nearest Neighbours"][-1], mods["Decision Tree"], mods["Random Forest"],
                       mods["Gradient Boosted Trees"])
    p = len(C["columns"])
    f1_rank = {m: i + 1 for i, m in enumerate(res["Model"])}

    def tt(m):
        return f"train {ri.loc[m, 'Train accuracy']:.0%} vs test {ri.loc[m, 'Accuracy']:.0%}"
    st.markdown("#### Why these algorithms")
    rationale([
        dict(name="K-Nearest Neighbours", why=f"Labels a lead by the answers of her {knn.n_neighbors} most similar respondents "
             "(distance-weighted, inputs standardised).",
             business="Finds look-alike customers: women who answer like her said Yes or No.",
             strengths="No assumptions about the shape of the data; captures local patterns.",
             weakness=f"{p} inputs, most of them 0/1 flags, make distances noisy; test accuracy {ri.loc['K-Nearest Neighbours', 'Accuracy']:.0%} "
                      f"(rank {rank['K-Nearest Neighbours']} of 4); {tt('K-Nearest Neighbours')}."),
        dict(name="Decision Tree", why=f"Learns if-then splits (max depth {dt.max_depth}, at least {dt.min_samples_leaf} people per leaf).",
             business="Readable rules showing which answers push a lead towards Yes.",
             strengths="Transparent; no scaling needed; mixes numeric and 0/1 inputs.",
             weakness=f"A single tree is unstable; test accuracy {ri.loc['Decision Tree', 'Accuracy']:.0%} (rank {rank['Decision Tree']} of 4)."),
        dict(name="Random Forest", why=f"Averages {rf.n_estimators} trees, each built on a random sample of people and inputs.",
             business="Robust buy / not-buy scores and a ranking of the answers that matter most.",
             strengths="Lower variance than one tree; gives feature importance.",
             weakness=f"{tt('Random Forest')}: the gap of {(ri.loc['Random Forest', 'Train accuracy'] - ri.loc['Random Forest', 'Accuracy']) * 100:.0f} "
                      "points shows it memorises the training set."),
        dict(name="Gradient Boosting", why=f"Adds {gb.n_estimators} shallow trees (depth {gb.max_depth}, learning rate {gb.learning_rate}) "
             "one after another, each correcting the previous errors.",
             business=f"Powers the prospect scorer; rank {f1_rank['Gradient Boosted Trees']} of 4 on macro-F1 "
                      f"({ri.loc['Gradient Boosted Trees', 'F1 (macro)']:.2f}).",
             strengths="Captures non-linear effects and interactions; strong on tabular survey data.",
             weakness=f"Slower and has more settings to tune; {tt('Gradient Boosted Trees')}."),
    ])

    kpis([(best, "Best model (highest macro-F1)"), (pct(br["Accuracy"]), "Accuracy on unseen test data"),
          (f"{br['F1 (macro)']:.2f}", "Macro F1 score"), (f"{br['ROC-AUC (one-vs-rest)']:.2f}", "ROC-AUC")])
    f, ax = fig(10, 3.6)
    mets = ["Accuracy", "Precision (macro)", "Recall (macro)", "F1 (macro)", "ROC-AUC (one-vs-rest)"]
    x = np.arange(len(mets)); w = 0.2
    shades = {m: (MARK if m == best else c) for m, c in zip(res["Model"], [CAMEL, "#9C8F80", GREIGE_DARK, GREIGE])}
    for i, (_, r) in enumerate(res.iterrows()):
        ax.bar(x + (i - 1.5) * w, [r[m] for m in mets], w - 0.02, color=shades[r["Model"]], label=r["Model"])
    ax.set_xticks(x, mets); ax.set_ylim(0, 1); ax.legend(ncol=4, fontsize=8.5, loc="upper left", bbox_to_anchor=(0, 1.15))
    ax.grid(axis="x", visible=False); ax.set_title("Model comparison on the test set (espresso = best)", pad=28)
    show(f)
    gaps = (ri["Train accuracy"] - ri["Accuracy"]).sort_values(ascending=False)
    explain(f"Four algorithms scored on {C['n_test']} test respondents they never saw ({C['n_train']} used for training, "
            "stratified 75/25 split) plus 5-fold cross-validation.",
            f"**{best}** has the highest macro-F1 ({br['F1 (macro)']:.2f}) and accuracy {br['Accuracy']:.1%} "
            f"(5-fold CV {br['5-fold CV accuracy']:.1%} ± {br['CV std']:.1%}). Largest train-test gap: **{gaps.index[0]}** "
            f"({gaps.iloc[0] * 100:.0f} points); smallest: {gaps.index[-1]} ({gaps.iloc[-1] * 100:.0f} points).",
            f"Score leads with {best}; judge models on test and cross-validated scores, not training accuracy.")
    with st.expander("Full metrics table"):
        st.dataframe(res.style.format({c: "{:.3f}" for c in res.columns if c != "Model"}), hide_index=True, width="stretch")

    st.markdown("#### Confusion matrices (rows = actual, columns = predicted)")
    names = list(C["cms"])
    for i in range(0, len(names), 2):
        cols = st.columns(2)
        for col, name in zip(cols, names[i:i + 2]):
            cm = C["cms"][name]
            with col:
                f, ax = plt.subplots(figsize=(4.6, 3.4))
                sns.heatmap(cm, annot=True, fmt="d", cmap=GREY_CMAP if name != best else SEQ_CMAP, cbar=False,
                            xticklabels=A.CLASSES, yticklabels=A.CLASSES, linewidths=1.5, linecolor="white", ax=ax,
                            annot_kws={"size": 11, "weight": "bold"})
                ax.set_title(name + ("  (best)" if name == best else ""), fontsize=10)
                ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.grid(False)
                show(f)
                acc = np.trace(cm) / cm.sum()
                recall = {c: cm[j, j] / cm[j].sum() for j, c in enumerate(A.CLASSES)}
                yn = int(cm[0, 2] + cm[2, 0])
                worst = min(recall, key=recall.get)
                prec_yes = cm[2, 2] / cm[:, 2].sum() if cm[:, 2].sum() else 0
                explain("Each cell counts test respondents with that actual and predicted answer; the diagonal is correct.",
                        f"{acc:.0%} correct. Recall: Yes {recall['Yes']:.0%}, Maybe {recall['Maybe']:.0%}, No {recall['No']:.0%} "
                        f"(weakest: {worst}). Yes and No confused {yn} times; {prec_yes:.0%} of predicted Yes are real Yes.",
                        f"Chosen for lead scoring (highest macro-F1)." if name == best else
                        f"Not chosen: macro-F1 {ri.loc[name, 'F1 (macro)']:.2f} vs {br['F1 (macro)']:.2f} for {best}.")

    st.markdown("#### Business view: buyer vs not-yet-buyer")
    mdl = st.selectbox("Model", list(mods), index=list(mods).index(best))
    B = cached_binary(df, mdl)
    kpis([(pct(B["accuracy"]), "Accuracy"), (pct(B["precision"]), "Precision (targeted leads that buy)"),
          (pct(B["recall"]), "Recall (buyers we catch)"), (f"{B['f1']:.2f}", "F1")])
    c1, c2 = st.columns(2)
    with c1:
        f, ax = plt.subplots(figsize=(4.6, 3.6))
        m = np.array([[B["tp"], B["fn"]], [B["fp"], B["tn"]]])
        lab = np.array([[f"TP\n{B['tp']}", f"FN\n{B['fn']}"], [f"FP\n{B['fp']}", f"TN\n{B['tn']}"]])
        sns.heatmap(m, annot=lab, fmt="", cmap=SEQ_CMAP, cbar=False, xticklabels=["Pred: Yes", "Pred: Not yes"],
                    yticklabels=["Actual: Yes", "Actual: Not yes"], linewidths=1.5, linecolor="white", ax=ax,
                    annot_kws={"size": 11, "weight": "bold"})
        ax.grid(False); ax.set_title(f"Yes vs not-yes: {mdl}")
        show(f)
        explain("TP = real buyer flagged; FP = flagged but not a buyer (wasted ad money); FN = buyer missed (lost sale); "
                "TN = correctly ignored.",
                f"Of {B['tp'] + B['fp']} leads flagged as buyers, {B['tp']} are real ({pct(B['precision'])} precision); "
                f"{B['fn']} real buyers are missed ({pct(B['recall'])} recall).",
                "Wasted ad spend (FP) " + ("outnumbers" if B["fp"] > B["fn"] else "does not outnumber")
                + " missed sales (FN) at the 50% cut-off; a cash-tight launch should raise the cut-off to favour precision.")
    with c2:
        f, ax = plt.subplots(figsize=(4.6, 3.6))
        ax.plot(B["fpr"], B["tpr"], color=MARK, lw=2.5, label=f"AUC = {B['auc']:.2f}")
        ax.fill_between(B["fpr"], B["tpr"], color=MARK2, alpha=.15)
        ax.plot([0, 1], [0, 1], color=NEUTRAL, ls="--", lw=1, label="Random guess")
        ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate"); ax.legend(loc="lower right")
        ax.set_title("ROC curve")
        show(f)
        explain("Each point is a different probability cut-off: how many buyers are caught (up) for how many non-buyers "
                "wrongly flagged (right). The dashed line is random guessing.",
                f"AUC {B['auc']:.2f}: the model ranks a random buyer above a random non-buyer {B['auc']:.0%} of the time "
                f"(random guessing = 50%).",
                "Use the model's probability to rank leads and spend on the top of the list first.")
    imp = C["importance"].head(12)
    hbar(pd.Series(imp.values, index=[pretty(i) for i in imp.index]), "Top 12 answers the tree model relies on",
         fmt="{:.3f}", w=9, h=4.2)
    imp_model = best if best in ("Random Forest", "Gradient Boosted Trees", "Decision Tree") else "Random Forest"
    explain(f"Feature importance from {imp_model}: how much each input reduces prediction error, summed over all splits.",
            f"Top drivers: **{', '.join(pretty(i) for i in imp.index[:3])}**; together the top 3 carry "
            f"{imp.iloc[:3].sum() / C['importance'].sum():.0%} of total importance.",
            "Ask these questions in the website quiz and use them as ad-audience filters.")


def regression_view():
    R = cached_regression(df)
    res = R["results"]
    ri = res.set_index("Model")
    bestr = res.sort_values("R² (test)", ascending=False).iloc[0]
    lasso, ridge, lin = ri.loc["Lasso (L1)"], ri.loc["Ridge (L2)"], ri.loc["Linear Regression"]
    p = len(R["columns"])
    X = A.model_matrix(df, extra=regression_extra(R["columns"]))[R["columns"]]
    cm_ = X.corr().abs().where(np.triu(np.ones((p, p), dtype=bool), 1)).stack()
    mx_pair = cm_.idxmax()
    st.markdown("#### Why these algorithms")
    rationale([
        dict(name="Linear regression", why="Fits willingness to pay as a weighted sum of the inputs (ordinary least squares).",
             business="The ₹ effect of each answer on the price she will pay.",
             strengths="Transparent coefficients; fast baseline to beat.",
             weakness=f"{p} inputs; the most correlated pair ({pretty(mx_pair[0])}, {pretty(mx_pair[1])}) has |r| = "
                      f"{cm_.max():.2f}" + (", i.e. the same information twice (perfect collinearity), so their separate "
                                            "coefficients cannot be trusted." if cm_.max() > 0.999 else
                                            ", which can make single coefficients unstable.")),
        dict(name="Ridge (L2)", why=f"Linear regression with a penalty that shrinks all coefficients; strength alpha = "
             f"{ridge['Alpha']:.1f} chosen by cross-validation.",
             business="Stable price drivers when inputs overlap.",
             strengths="Handles correlated inputs and reduces overfitting.",
             weakness=f"Keeps all {int(ridge['Features used'])} inputs, so it does not simplify; test R² {ridge['R² (test)']:.3f} "
                      f"vs {lin['R² (test)']:.3f} for plain linear."),
        dict(name="Lasso (L1)", why=f"Penalty that can set coefficients to exactly zero (alpha = {lasso['Alpha']:.1f}).",
             business="A shortlist of the answers that really move price.",
             strengths=f"Automatic feature selection: keeps {int(lasso['Features used'])} of {p} inputs.",
             weakness="Among correlated inputs it keeps one and drops the others somewhat arbitrarily, so a dropped input "
                      "is not proof of no effect."),
    ])
    kpis([(bestr["Model"], "Best model (highest test R²)"), (f"{bestr['R² (test)']:.0%}", "Variation in price explained (R²)"),
          (inr(bestr["MAE (₹)"]), "Average error (MAE)"), (f"{int(lasso['Features used'])} / {p}", "Inputs Lasso keeps")])
    c1, c2 = st.columns([1, 1.5])
    with c1:
        f, ax = fig(4.8, 3.6)
        r2 = ri["R² (test)"]
        ax.bar([m.replace(" Regression", "") for m in r2.index], r2.values,
               color=[MARK if m == bestr["Model"] else NEUTRAL_LIGHT for m in r2.index], width=0.55)
        for i, v in enumerate(r2.values):
            ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontweight="bold")
        ax.set_ylim(0, 1); ax.set_title("Test R² by model"); ax.grid(False); ax.set_yticks([])
        show(f)
        gap = (ri["Train R²"] - ri["R² (test)"]).abs().max()
        explain("R² = share of the variation in willingness to pay that the model explains on unseen test data.",
                f"**{bestr['Model']}** explains {bestr['R² (test)']:.1%} with average error ±{inr(bestr['MAE (₹)'])}. "
                f"The three models are within {(r2.max() - r2.min()) * 100:.1f} R² points; the largest train-test gap is "
                f"{gap:.3f}.",
                ("No sign of overfitting, so " if gap < 0.05 else "There is some overfitting, so ")
                + f"use {bestr['Model']} for price bands per segment.")
    with c2:
        st.dataframe(res.style.format({"R² (test)": "{:.3f}", "MAE (₹)": "₹{:,.0f}", "RMSE (₹)": "₹{:,.0f}",
                                       "5-fold CV R²": "{:.3f}", "Train R²": "{:.3f}", "Alpha": "{:.2f}"}),
                     hide_index=True, width="stretch")
    mname = st.selectbox("Model to inspect", list(R["preds"]), index=list(R["preds"]).index("Lasso (L1)"))
    resid = R["y_test"] - R["preds"][mname]
    c1, c2 = st.columns(2)
    with c1:
        f, ax = fig(6, 4.4)
        ax.scatter(R["y_test"], R["preds"][mname], s=12, alpha=0.45, color=MARK, edgecolors="none")
        lo, hi = R["y_test"].min(), R["y_test"].max()
        ax.plot([lo, hi], [lo, hi], color=INK, lw=1.2, ls="--", label="Perfect prediction")
        ax.set_xlabel("Actual WTP (₹)"); ax.set_ylabel("Predicted WTP (₹)"); ax.legend(); ax.set_title(f"Predicted vs actual: {mname}")
        show(f)
        within = (np.abs(resid) <= 500).mean()
        explain("Each dot is a test respondent; dots on the dashed line are perfect predictions.",
                f"Average miss ±{inr(np.abs(resid).mean())}; {within:.0%} of predictions are within ±₹500 of the real answer.",
                "Use predictions for price tiers (bands of about ±₹500), not for exact individual prices.")
    with c2:
        co = R["coefs"][mname]
        co = co[co.abs() > 1e-6].sort_values(key=np.abs, ascending=False).head(12).sort_values()
        f, ax = fig(6, 4.4)
        ax.barh([pretty(i) for i in co.index], co.values, color=[MARK if v > 0 else NEUTRAL for v in co.values], height=0.62)
        ax.axvline(0, color=MUTED, lw=0.8); ax.set_title("₹ change per 1 std. dev. rise in each input")
        ax.grid(axis="y", visible=False); ax.tick_params(axis="y", labelsize=8.5)
        show(f)
        pos, negs = co[co > 0].sort_values(ascending=False), co[co < 0].sort_values()
        explain("Coefficients on standardised inputs, so effects are comparable: espresso raises price, greige lowers it.",
                f"Biggest positive: **{', '.join(pretty(i) for i in pos.index[:3])}** (top +{inr(pos.iloc[0]) if len(pos) else '-'}). "
                + (f"Biggest negative: **{pretty(negs.index[0])}** ({inr(negs.iloc[0])})." if len(negs) else "No negative coefficient in the top 12."),
                "Price the premium line for buyers strong on the positive drivers; keep an entry SKU for the rest.")
    f, ax = fig(9, 2.8)
    ax.hist(resid, bins=40, color=MARK, alpha=0.8, edgecolor="white"); ax.axvline(0, color=INK, lw=1.2)
    ax.set_title(f"Residuals (actual - predicted): {mname}"); ax.set_xlabel("₹"); ax.set_yticks([])
    show(f)
    sk = stats.skew(resid)
    explain("Distribution of prediction errors; a centred, roughly symmetric shape means no systematic bias.",
            f"Mean error {inr(resid.mean())}, median {inr(np.median(resid))}, skewness {sk:.2f}.",
            "No systematic over- or under-pricing, so a linear model suits this data." if abs(resid.mean()) < 0.1 * np.abs(resid).mean()
            and abs(sk) < 0.5 else "Errors are biased or skewed; check for a missing non-linear effect before relying on it.")
    zero = R["coefs"]["Lasso (L1)"]
    zero = zero[zero.abs() <= 1e-6].index
    with st.expander(f"Inputs Lasso dropped ({len(zero)})"):
        st.write(", ".join(pretty(z) for z in zero) or "None")
    xh = interested["wtp_extra_heel_inr"]
    kpis([(inr(xh.median()), "Median WTP per extra heel (Yes + Maybe)"), (inr(xh.quantile(.25)), "25th percentile"),
          (pct((df["wtp_extra_heel_inr"] / df["wtp_base_inr"]).median()), "Extra heel as % of base WTP (median)")])


def clustering_view():
    scan = cached_scan(df).set_index("k")
    inertia = scan["Inertia (WCSS)"]
    drops = -inertia.diff().dropna()                     # drop from k-1 to k, indexed by k
    bend = (drops - drops.shift(-1)).dropna()            # how much the drop shrinks after k
    k_elbow = int(bend.idxmax())
    k_sil = int(scan["Silhouette"].idxmax())
    s_chosen = scan.loc[K_CHOSEN, "Silhouette"]
    bic = cached_lca_scan(df).set_index("Classes")["BIC"]
    k_bic = int(bic.idxmin())
    L4 = cached_lca(df, K_CHOSEN)
    st.markdown("#### Why these algorithms")
    rationale([
        dict(name="K-Means", why="Groups respondents with similar behaviour by minimising distance to each group's centre; "
             f"{len(A.CLUSTER_VARS)} standardised inputs, intent excluded.",
             business="Personas with size, Yes rate and price, used for targeting and messaging.",
             strengths="Fast, simple and gives every respondent exactly one segment.",
             weakness=f"Assumes round, similar-sized groups; silhouette at k = {K_CHOSEN} is {s_chosen:.2f} "
                      f"({silhouette_quality(s_chosen)} separation), so segments overlap."),
        dict(name="Latent Class Analysis", why="Probability model for categorical answers fitted with the EM algorithm; "
             "each person gets a probability of belonging to each class.",
             business="A second, statistical check on the personas; BIC picks the number of classes.",
             strengths="Built for categorical survey answers; shows uncertainty, not just a label.",
             weakness=f"Numeric answers had to be binned into {len(L4['vars'])} categorical variables, losing detail; "
                      f"entropy at {K_CHOSEN} classes is {L4['entropy']:.2f} (1 = perfectly separated)."),
    ])
    c1, c2 = st.columns(2)
    with c1:
        f, ax = fig()
        ax.plot(inertia.index, inertia.values, marker="o", color=MARK, lw=2, ms=7)
        ax.scatter([k_elbow], [inertia[k_elbow]], s=160, facecolors="none", edgecolors=INK, lw=1.5, zorder=5)
        ax.set_title("Elbow method"); ax.set_xlabel("Number of clusters (k)"); ax.set_ylabel("Within-cluster sum of squares")
        show(f)
        explain("Within-cluster sum of squares for each k; the 'elbow' is where adding a cluster stops helping much (circled).",
                f"WCSS falls by {drops[k_elbow]:,.0f} going to k = {k_elbow} but only by {drops[k_elbow + 1]:,.0f} going to "
                f"k = {k_elbow + 1}, so the elbow is at k = {k_elbow}.",
                f"k = {K_CHOSEN} is used for the personas" + (" (the elbow)." if k_elbow == K_CHOSEN else
                f"; it is {'one step past' if K_CHOSEN == k_elbow + 1 else 'away from'} the elbow, trading some compactness for "
                f"one more targetable group."))
    with c2:
        f, ax = fig()
        ax.plot(scan.index, scan["Silhouette"], marker="o", color=MARK, lw=2, ms=7)
        ax.scatter([K_CHOSEN], [s_chosen], s=160, facecolors="none", edgecolors=INK, lw=1.5, zorder=5)
        ax.set_title("Silhouette score"); ax.set_xlabel("Number of clusters (k)")
        show(f)
        explain("Silhouette: how much closer each person is to her own cluster than to the next one (-1 to 1); circle = chosen k.",
                f"Highest at k = {k_sil} ({scan.loc[k_sil, 'Silhouette']:.2f}); at k = {K_CHOSEN} it is {s_chosen:.2f}. "
                f"All values are {silhouette_quality(scan['Silhouette'].max())} (below 0.25 at every k)."
                if scan["Silhouette"].max() < 0.25 else
                f"Highest at k = {k_sil} ({scan.loc[k_sil, 'Silhouette']:.2f}); at k = {K_CHOSEN} it is {s_chosen:.2f}.",
                "Treat segments as tendencies, not hard boxes; confirm them with LCA below and with the real survey."
                if s_chosen < 0.25 else "Segments are well separated enough to target separately.")
    k = st.slider("Number of segments to explore (k)", 2, 8, K_CHOSEN)
    K = cached_kmeans(df, k)
    prof = K["profile"].sort_values("Yes_pct", ascending=False)
    f, ax = fig(9, 4.6)
    nm = seg_names(prof)
    for c in sorted(prof.index):
        msk = K["labels"] == c
        ax.scatter(K["pcs"][msk, 0], K["pcs"][msk, 1], s=9, alpha=0.5, color=seg_col(c),
                   label=nm[c], edgecolors="none")
    ax.set_title(f"{k} K-Means segments on the first 2 principal components"); ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.legend(fontsize=8, markerscale=2.5, loc="center left", bbox_to_anchor=(1, 0.5))
    show(f)
    top, bottom = prof.iloc[0], prof.iloc[-1]
    explain(f"Each dot is a respondent, coloured by segment. PCA squeezes {len(A.CLUSTER_VARS)} inputs into 2 axes, so "
            "some overlap is expected.",
            f"Highest Yes rate: **{top['Name']}** ({pct(top['Yes_pct'])}, {pct(top['Share_pct'])} of respondents); lowest: "
            f"**{bottom['Name']}** ({pct(bottom['Yes_pct'])}). Spread of Yes rates: {(top['Yes_pct'] - bottom['Yes_pct']) * 100:.0f} points.",
            "Segments differ strongly in conversion even though they overlap in answers, so segment-level targeting is worthwhile."
            if top["Yes_pct"] - bottom["Yes_pct"] > 0.2 else "Conversion differs little between segments; target on behaviour instead.")
    show_cols = {"Name": "Persona", "Conversion_tier": "Conversion", "Size": "Size", "Share_pct": "Share",
                 "Avg_age": "Avg age", "Median_income": "Median income ₹", "Heel_freq_0_5": "Heel freq (0-5)",
                 "Carry_extra_1_5": "Carries extra (1-5)", "Pains": "Avg pains", "Drives_pct": "Drives %",
                 "Style_imp": "Style imp.", "Yes_pct": "Yes %", "Maybe_pct": "Maybe %", "Median_WTP": "Median WTP ₹"}
    st.dataframe(prof[list(show_cols)].rename(columns=show_cols).style.format(
        {"Share": "{:.0%}", "Avg age": "{:.0f}", "Median income ₹": "{:,.0f}", "Heel freq (0-5)": "{:.1f}",
         "Carries extra (1-5)": "{:.1f}", "Avg pains": "{:.1f}", "Drives %": "{:.0%}", "Style imp.": "{:.1f}",
         "Yes %": "{:.0%}", "Maybe %": "{:.0%}", "Median WTP ₹": "{:,.0f}"}), width="stretch")

    st.markdown("#### Latent Class Analysis: a second, probabilistic segmentation")
    c1, c2 = st.columns([1, 1.6])
    with c1:
        f, ax = fig(4.6, 3.4)
        ax.plot(bic.index, bic.values, marker="o", color=MARK, lw=2)
        ax.scatter([k_bic], [bic[k_bic]], s=160, facecolors="none", edgecolors=INK, lw=1.5, zorder=5)
        ax.set_title("BIC by number of latent classes (lower = better)"); ax.set_xlabel("Latent classes")
        show(f)
        bdrop = -bic.diff().dropna()
        gains = ", ".join(f"to {k_}: -{v:,.0f}" for k_, v in bdrop.items())
        at_edge = k_bic == bic.index.max()
        explain("BIC rewards fit but penalises extra classes; the lowest value is the best trade-off (circled).",
                (f"BIC {trend_word(bic.values)} up to {k_bic} classes, so there is no minimum in the range tested. "
                 if at_edge else f"BIC is lowest at {k_bic} classes ({bic[k_bic]:,.0f}). ")
                + f"Improvement per extra class: {gains}.",
                (f"BIC alone would allow more classes, but the gains shrink after {int(bdrop.idxmax())} classes; "
                 f"{K_CHOSEN} classes are used so LCA can be compared one-to-one with the {K_CHOSEN} K-Means segments."
                 if at_edge else
                 f"Use {k_bic} latent classes" + (f", which matches the {K_CHOSEN} K-Means segments." if k_bic == K_CHOSEN
                                                  else f"; K-Means uses {K_CHOSEN}, so compare the two before finalising personas.")))
        nl = st.slider("Latent classes", 2, 6, K_CHOSEN)
    L = cached_lca(df, nl)
    with c2:
        dl = df.assign(lc=L["labels"])
        lp = dl.groupby("lc").agg(Share=("respondent_id", lambda s: len(s) / n), Yes=("intent_yes", "mean"),
                                  Median_WTP=("wtp_base_inr", "median"), Age=("age", "mean"))
        lp["Defining answers"] = [
            "; ".join(f"{v}: {L['levels'][j][int(np.argmax(L['theta'][j][c]))]} ({L['theta'][j][c].max():.0%})"
                      for j, v in enumerate(L["vars"]) if v in ("Heel frequency", "Carries extra pair", "Occupation", "Commute"))
            for c in lp.index]
        st.dataframe(lp.sort_values("Yes", ascending=False).style.format(
            {"Share": "{:.0%}", "Yes": "{:.0%}", "Median_WTP": "₹{:,.0f}", "Age": "{:.0f}"}), width="stretch")
        st.caption(f"Entropy (separation quality, 1 = perfect): {L['entropy']:.2f}")
    Kc = cached_kmeans(df, K_CHOSEN)
    kname = seg_names(Kc["profile"])
    xt = pd.crosstab(pd.Series(Kc["labels"], name="K-Means segment").map(kname),
                     pd.Series(L["labels"], name="Latent class"), normalize="index")
    f, ax = fig(8, 3.2)
    sns.heatmap(xt, annot=True, fmt=".0%", cmap=SEQ_CMAP, cbar=False, linewidths=1, linecolor="white", ax=ax)
    ax.set_title(f"How the {K_CHOSEN} K-Means segments map onto {nl} latent classes"); ax.grid(False)
    show(f)
    agree = xt.max(axis=1)
    explain("Each row is a K-Means segment; cells show the share of its members in each latent class.",
            f"{int((agree >= 0.7).sum())} of {len(xt)} K-Means segments put 70% or more of their members in a single latent "
            f"class; the clearest is **{agree.idxmax()}** ({agree.max():.0%}) and the least clear **{agree.idxmin()}** ({agree.min():.0%}).",
            f"Treat {agree.idxmax()} as a robust persona; re-check {agree.idxmin()} with the real survey before building a product line for it.")


def rules_view():
    st.markdown("#### Why this algorithm")
    st.caption("Support = share of baskets with the whole combination · Confidence = P(then | if) · "
               "Lift = confidence ÷ how common the 'then' item is anyway (above 1 = a real association).")
    with st.expander("Rule settings", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        pop = c1.selectbox("Respondents", ["Likely buyers (Yes + Maybe)", "Everyone"])
        groups = c2.multiselect("Items", ["features", "pains", "situations", "concerns"], default=["features"])
        ms = c3.slider("Min support", 0.02, 0.30, 0.05, 0.01)
        mc = c4.slider("Min confidence", 0.2, 0.9, 0.5, 0.05)
        only_feat = st.checkbox("Only show rules whose consequent (then-part) is a product feature", value=True)
    if not groups:
        st.warning("Pick at least one item group in Rule settings.")
        return
    base = interested if pop.startswith("Likely") else df
    B = A.baskets(base, tuple(groups))
    sup, names, fi = A.apriori(B, ms, 3)
    rules = A.association_rules(sup, names, mc, (lambda x: x in FEATURES) if only_feat else None)
    rationale([dict(name="Apriori (association rule mining)",
                    why=f"Finds items that appear together in respondents' baskets (default basket: the Q23 feature "
                        f"wish-list), keeping combinations with support ≥ {ms:.0%} and rules with confidence ≥ {mc:.0%}.",
                    business="Which features must launch together so first buyers are not disappointed, and which can wait.",
                    strengths="Easy-to-read 'if A and B, then C' rules; lift separates real associations from chance.",
                    weakness=f"Uses yes/no items only (no price or intensity); {len(rules)} rules at these settings, many of "
                             "them overlapping versions of the same pattern.")])
    kpis([(f"{len(base):,}", "Baskets analysed"), (f"{len(fi)}", "Frequent itemsets"), (f"{len(rules)}", "Rules found")])
    if len(rules) == 0:
        st.warning("No rules at these thresholds. Lower the support or confidence.")
        return
    show_r = rules.drop(columns=["_ante", "_cons"])
    c1, c2 = st.columns([1.3, 1])
    with c1:
        top = show_r.head(10)
        lab = [f"{a[:40]} → {b[:24]}" for a, b in zip(top.iloc[:, 0], top.iloc[:, 1])]
        f, ax = fig(7, 4.4)
        ax.barh(lab[::-1], top["Lift"].values[::-1], color=MARK, height=0.6)
        for i, v in enumerate(top["Lift"].values[::-1]):
            ax.text(v, i, f" {v:.2f}x", va="center", fontsize=8.5)
        ax.axvline(1, color=NEUTRAL, ls="--", lw=1); ax.set_title("Top 10 rules by lift"); ax.tick_params(axis="y", labelsize=8)
        ax.grid(False); ax.set_xticks([])
        show(f)
        r0 = rules.iloc[0]
        explain("Each bar is one rule 'if left side, then right side'; length = lift (dashed line = 1, no association).",
                f"Strongest rule: if a buyer wants **{r0.iloc[0]}**, then {r0['Confidence']:.0%} also want **{r0.iloc[1]}**, "
                f"{r0['Lift']:.1f}x more often than average (support {r0['Support']:.1%}).",
                f"Bundle '{r0.iloc[1]}' with '{r0.iloc[0]}' in the same kit.")
    with c2:
        f, ax = fig(5.4, 4.4)
        sc = ax.scatter(rules["Support"], rules["Confidence"], c=rules["Lift"], cmap=SEQ_CMAP, s=50,
                        edgecolors=MUTED, linewidths=0.4, vmin=min(0.8, rules["Lift"].min()))
        plt.colorbar(sc, ax=ax, label="Lift")
        ax.set_xlabel("Support"); ax.set_ylabel("Confidence"); ax.set_title("All rules: support vs confidence")
        show(f)
        strong = rules[(rules["Lift"] > 1.2) & (rules["Support"] >= rules["Support"].median())]
        explain("Each dot is a rule; right = common, up = reliable, darker = stronger lift.",
                f"{int((rules['Lift'] > 1).sum())} of {len(rules)} rules have lift above 1; {len(strong)} combine lift above 1.2 "
                f"with above-median support. Confidence ranges from {rules['Confidence'].min():.0%} to {rules['Confidence'].max():.0%}.",
                "Act on rules that are both common and strong (top-right, dark); rare rules are interesting but too small to "
                "shape a launch.")
    with st.expander("All rules (table)"):
        st.dataframe(show_r.head(40).style.format({"Support": "{:.3f}", "Confidence": "{:.2f}", "Lift": "{:.2f}",
                                                   "Conviction": "{:.2f}"}), hide_index=True, width="stretch", height=380)


def gauge(p, color):
    f, ax = plt.subplots(figsize=(3.6, 2.3))
    ax.pie([p, 1 - p, 1], colors=[color, GREIGE, "white"], startangle=180, counterclock=False,
           wedgeprops=dict(width=0.32, edgecolor="white", linewidth=2))
    ax.text(0, 0.3, pct(p), ha="center", va="center", fontsize=24, fontweight="bold", color=INK)
    ax.text(0, -0.08, "chance she buys", ha="center", va="center", fontsize=9, color=MUTED)
    ax.set_ylim(-0.2, 1.05); ax.set_xlim(-1.05, 1.05); ax.set_aspect("equal"); ax.grid(False)
    show(f)


def prospect_view():
    st.caption("The trained models in action: describe a prospect (from a website quiz or a sales conversation) "
               "and get her probability of buying, expected price and segment.")
    result_box = st.container()
    st.markdown("#### Describe the prospect")
    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.number_input("Age", 16, 75, 30)
        occ = st.selectbox("Occupation", OCCUPATIONS)
        inc = st.number_input("Monthly income (₹)", 0, 1000000, 70000, 5000)
        city = st.selectbox("City", CITIES)
        commute = st.selectbox("Commute", COMMUTE)
        hours = st.slider("Hours on feet per day", 0.0, 16.0, 5.0, 0.5)
    with c2:
        hf = st.select_slider("Heel frequency", HEEL_FREQ, "Weekly")
        hh = st.slider("Hours in heels per wear", 0.0, 12.0, 5.0, 0.5)
        pairs = st.number_input("Pairs of heels owned", 0, 60, 5)
        spend = st.number_input("Annual footwear spend (₹)", 0, 200000, 15000, 1000)
        maxp = st.number_input("Max paid for a pair (₹)", 0, 50000, 4000, 500)
        carry = st.select_slider("Carries an extra pair", CARRY_FREQ, "Often")
    with c3:
        pains = st.multiselect("Pains experienced", PAINS, ["Knee pain", "Calf pain"])
        sits = st.multiselect("Switch situations", SITUATIONS, ["Office to party"])
        feats = st.multiselect("Features interested", FEATURES, ["Block heel module", "Orthopaedic cushioned insole"])
        conc = st.multiselect("Concerns", CONCERNS, ["Heel detaching / Safety"])
        sev = st.slider("Pain severity", 1, 10, 6)
        trust = st.slider("Trust in D2C brands", 1, 5, 3)
        infl = st.slider("Influencer influence", 1, 5, 3)
        comf, style, dur = st.slider("Comfort imp.", 1, 5, 4), st.slider("Style imp.", 1, 5, 4), st.slider("Durability imp.", 1, 5, 4)
        aware = st.checkbox("Had heard of convertible heels")
    row = {"respondent_id": "NEW", "age": age, "city": city, "occupation": occ, "monthly_income_inr": inc,
           "marital_status": "Single", "commute_mode": commute, "hours_on_feet": hours, "heel_frequency": hf,
           "heel_hours_per_wear": hh, "heel_pairs_owned": pairs, "annual_footwear_spend_inr": spend,
           "max_price_paid_inr": maxp, "preferred_channel": "Myntra / Ajio",
           "heel_types_preferred": "Block", "pains_experienced": "; ".join(pains) or "None", "pain_severity": sev,
           "carry_extra_pair": CARRY_FREQ.index(carry) + 1, "switch_situations": "; ".join(sits) or "None",
           "importance_comfort": comf, "importance_style": style, "importance_durability": dur,
           "aware_of_convertible_heels": "Yes" if aware else "No", "features_interested": "; ".join(feats) or "None",
           "concerns": "; ".join(conc) or "None", "trust_d2c_brands": trust, "influencer_influence": infl,
           "wtp_base_inr": 3000, "wtp_extra_heel_inr": 700, "purchase_intent": "Maybe", "open_feedback": ""}
    # run the new row through the same cleaning pipeline (no outlier capping is triggered for 1 row of valid data)
    new = pd.concat([df[raw.columns.intersection(df.columns)].head(200), pd.DataFrame([row])], ignore_index=True)
    new_clean, _ = clean(new)
    x = new_clean.tail(1)
    clf = cached_binary_model(DF_HASH, df, "Gradient Boosted Trees")
    p_yes = clf.predict_proba(A.model_matrix(x))[0, 1]
    regs, sc, rcols = cached_regression_models(DF_HASH, df)
    wtp = regs["Lasso (L1)"].predict(sc.transform(A.model_matrix(x, extra=regression_extra(rcols))[rcols]))[0]
    km, ksc, kprof = cached_kmeans_model(DF_HASH, df, K_CHOSEN)
    seg = km.predict(ksc.transform(x[A.CLUSTER_VARS]))[0]
    tier, color, action = (("Hot lead", MARK, "Retarget with the premium kit and a safety demo video") if p_yes > HOT else
                           ("Warm lead", MAYBE, "Nurture with content, reviews and a first-order offer") if p_yes > WARM else
                           ("Cold lead", NEUTRAL, "Do not spend paid media; organic only"))
    with result_box:
        c1, c2, c3, c4 = st.columns([1.1, 1, 1, 1.3])
        with c1:
            gauge(p_yes, color)
        c2.markdown(f"<div class='kpi' style='--c:{ACCENT}'><div class='v'>{inr(wtp)}</div>"
                    "<div class='l'>Predicted willingness to pay (Lasso)</div></div>", unsafe_allow_html=True)
        c3.markdown(f"<div class='kpi' style='--c:{seg_col(seg)}'><div class='v' style='font-size:1.1rem'>{_md(short(kprof.loc[seg, 'Name']))}</div>"
                    "<div class='l'>Customer segment (K-Means)</div></div>", unsafe_allow_html=True)
        c4.markdown(f"<div class='kpi' style='--c:{color}'><div class='v' style='font-size:1.3rem'>{tier}</div>"
                    f"<div class='l' style='color:{INK}'>{action}</div></div>", unsafe_allow_html=True)
        st.write("")
        explain(f"Gauge = Gradient Boosting probability that this prospect says Yes. Leads above {HOT:.0%} are hot, "
                f"{WARM:.0%}-{HOT:.0%} warm, below {WARM:.0%} cold.",
                f"This prospect: {pct(p_yes)} chance of Yes versus a {pct(yes_rate)} Yes rate in the survey; predicted WTP "
                f"{inr(wtp)} is {'above' if wtp > med_wtp else 'at or below'} the {inr(med_wtp)} median of interested respondents.",
                f"{tier}: {action}.")
        st.divider()


# =================================================================== PRESCRIPTIVE
def page_prescriptive():
    view = subnav("presc", ["Launch plan", "Price", "Who to target", "Break-even"])

    if view == "Launch plan":
        seq = A.launch_sequence(interested)
        st.markdown("#### Recommended launch roadmap")
        phases = list(dict.fromkeys(seq["Phase"]))
        pcol = [MARK, CAMEL, GREIGE_DARK, GREIGE_DARK]
        cols = st.columns(len(phases))
        for i, (col, ph) in enumerate(zip(cols, phases)):
            part = seq[seq.Phase == ph]
            body = "".join(bar_html(("[bundled] " if w_.startswith("Bundled") else "") + f_, d, pcol[i % 4])
                           for f_, d, w_ in zip(part["Feature"], part["Demand among likely buyers"], part["Why"]))
            col.markdown(f"<div class='card' style='--c:{pcol[i % 4]}'><div class='h'>{_md(ph)}</div>"
                         f"<div class='sub'>{len(part)} features · demand among likely buyers</div>{body}</div>",
                         unsafe_allow_html=True)
        p1 = seq[seq.Phase.str.startswith("Phase 1")]
        bundled = p1[p1.Why.str.startswith("Bundled")]
        explain("Step 1: the most-demanded features. Step 2: any feature that most buyers of the phase-1 features also "
                "expect (a strong association rule) is pulled into the launch and marked [bundled].",
                f"Launch with **{', '.join(p1.Feature)}** ({len(p1)} features, demand {pct(p1['Demand among likely buyers'].min())}"
                f"-{pct(p1['Demand among likely buyers'].max())}). "
                + (f"Added by an association rule: {'; '.join(w_.replace('Bundled: ', '') for w_ in bundled.Why)}." if len(bundled) else "No feature was added by a rule."),
                f"Fewer SKUs and moulds at launch ({len(p1)} of {len(seq)} features) lowers setup cost and inventory risk; "
                "later phases start only after the previous one sells.")

    elif view == "Price":
        mw = st.slider("How many 'Maybe' respondents actually convert? (weight)", 0.0, 1.0, 0.3, 0.05,
                       help="Survey intent overstates real purchases. Counting only part of the Maybes is a cautious assumption.")
        dc = A.demand_curve(df, mw)
        best_p = dc.loc[dc["Revenue index"].idxmax()]
        entry = dc.iloc[(dc["Price"] - ENTRY_PRICE).abs().argmin()]
        kpis([(inr(best_p["Price"]), "Revenue-maximising price"), (pct(best_p["Share of market buying"]), "Respondents who buy at that price"),
              (inr(ENTRY_PRICE), "Planned entry kit price (assumption)")])
        c1, c2 = st.columns(2)
        with c1:
            f, ax = fig(6.4, 3.9)
            ax.plot(dc["Price"], dc["Share of market buying"], color=MARK, lw=2.5)
            ax.axvline(GST_THRESHOLD, color=NEUTRAL, ls=":", lw=1.2)
            ax.text(GST_THRESHOLD + 30, ax.get_ylim()[1] * .9, f"₹{GST_THRESHOLD:,} GST step", fontsize=8, color=MUTED)
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
            ax.set_title("Demand curve"); ax.set_xlabel("Price (₹)"); ax.set_ylabel("Share of respondents who buy")
            show(f)
            explain(f"At each price, the share of respondents whose willingness to pay is at least that price (Yes counted fully, "
                    f"Maybe weighted {mw:.2f}).",
                    f"At {inr(entry['Price'])} {pct(entry['Share of market buying'])} would buy; at the revenue peak "
                    f"({inr(best_p['Price'])}) {pct(best_p['Share of market buying'])} would.",
                    "Use a low entry price to win volume and a premium kit near the revenue peak; never price above the point "
                    "where demand collapses.")
        with c2:
            f, ax = fig(6.4, 3.9)
            ax.plot(dc["Price"], dc["Revenue index"], color=MARK, lw=2.5)
            ax.axvline(best_p["Price"], color=INK, ls="--", lw=1.2)
            ax.scatter([best_p["Price"]], [best_p["Revenue index"]], color=CAMEL, s=60, zorder=5, edgecolors="white")
            ax.text(best_p["Price"], ax.get_ylim()[1] * .92, f" peak ₹{best_p['Price']:,.0f}", fontsize=9, fontweight="bold")
            ax.set_title("Revenue per respondent reached"); ax.set_xlabel("Price (₹)"); ax.set_ylabel("Price × share buying (₹)")
            show(f)
            rel = entry["Revenue index"] / best_p["Revenue index"]
            above = best_p["Price"] > GST_THRESHOLD
            explain("Price × share of respondents buying: the curve peaks at the revenue-maximising price.",
                    f"Revenue peaks at **{inr(best_p['Price'])}**; the {inr(ENTRY_PRICE)} entry price earns {rel:.0%} of that peak "
                    f"revenue. The peak is {'above' if above else 'at or below'} the ₹{GST_THRESHOLD:,} GST step.",
                    f"{GST_NOTE}. " + ("Offer an entry kit below the step to keep the 5% rate and a premium kit near the peak."
                                       if above else "The revenue peak already sits in the 5% GST band."))

    elif view == "Who to target":
        km, scaler, prof = cached_kmeans_model(DF_HASH, df, K_CHOSEN)
        prof = prof.sort_values("Yes_pct", ascending=False)
        plan = []
        for c, r in prof.iterrows():
            tier = r["Conversion_tier"]
            plan.append({"cluster": c, "Segment": r["Name"], "Share": r["Share_pct"], "Yes": r["Yes_pct"],
                         "WTP": r["Median_WTP"], "Tier": tier,
                         "Priority": {"High": "1 - Launch focus", "Medium": "2 - Nurture", "Low": "3 - Later / organic"}[tier],
                         "Offer": ("Premium kit (2 heels + pouch)" if r["Median_WTP"] > PREMIUM_WTP else
                                   f"Entry kit at {inr(ENTRY_PRICE)}" if r["Median_WTP"] < ENTRY_WTP else
                                   "Standard kit + 1 add-on heel"),
                         "Message": ("Drive flat, arrive in heels" if r["Drives_pct"] > .3 else
                                     "Office to party in one pair" if r["Heel_freq_0_5"] > 2.3 else
                                     "Comfort without giving up height" if r["Avg_age"] > 35 else "One shoe, four looks")})
        cols = st.columns(len(plan))
        for col, p in zip(cols, plan):
            name, occ = (p["Segment"].split(" (") + [""])[:2]
            sc = seg_col(p["cluster"])
            col.markdown(
                f"<div class='card' style='--c:{sc}'><div class='h' style='color:{sc}'>{_md(name)}</div>"
                f"<div class='sub'>{_md(occ.rstrip(')'))}</div><span class='badge' style='--c:{MUTED}'>Priority {p['Priority']}</span>"
                + bar_html("Share of respondents", p["Share"], sc) + bar_html("Say Yes", p["Yes"], YES)
                + f"<div class='row2'><span>Median WTP</span><b>{inr(p['WTP'])}</b></div>"
                f"<div class='lbl'>Offer</div><p>{_md(p['Offer'])}</p><div class='lbl'>Message</div><p>“{_md(p['Message'])}”</p></div>",
                unsafe_allow_html=True)
        st.write("")
        pf = pd.DataFrame(plan)
        f, ax = fig(10, 3.8)
        ax.scatter(pf["Share"], pf["Yes"], s=pf["WTP"] / 4, color=[seg_col(c) for c in pf["cluster"]], alpha=.75,
                   edgecolors="white", linewidths=2)
        for _, r in pf.iterrows():
            ax.annotate(short(r["Segment"]), (r["Share"], r["Yes"]), xytext=(14, 0), textcoords="offset points", va="center",
                        fontsize=9, color=seg_col(r["cluster"]), fontweight="bold")
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.set_xlim(0, pf["Share"].max() * 1.6); ax.set_ylim(-0.05, min(1, pf["Yes"].max() * 1.3))
        ax.set_xlabel("Share of respondents"); ax.set_ylabel("Say Yes"); ax.set_title("Segment map: bubble size = median willingness to pay")
        show(f)
        big = pf.loc[pf["Share"].idxmax()]
        explain("Each bubble is a segment: further right = bigger, higher = keener, bigger bubble = pays more.",
                f"**{short(plan[0]['Segment'])}** has the highest Yes rate ({pct(plan[0]['Yes'])}) and median WTP {inr(plan[0]['WTP'])}; "
                f"the largest segment is **{short(big['Segment'])}** ({pct(big['Share'])} of respondents, {pct(big['Yes'])} Yes).",
                f"Launch budget behind {short(plan[0]['Segment'])}; nurture medium-conversion segments with content and offers; "
                f"reach {short(plan[-1]['Segment'])} organically later.")

    else:
        st.caption("Defaults are planning assumptions for an Indian D2C footwear start-up. Open the panel to test your own.")
        with st.expander("Change the assumptions", expanded=False):
            c1, c2, c3 = st.columns(3)
            with c1:
                price = st.number_input("Kit price incl. GST (₹)", 999, 9999, ENTRY_PRICE, 100)
                unit_cost = st.number_input("Product cost per kit, COGS (₹)", 200, 5000, 900, 50)
                gst = st.selectbox("GST rate", [0.05, 0.18], index=0 if price <= GST_THRESHOLD else 1, format_func=lambda x: f"{x:.0%}")
                attach = st.slider("Customers buying an extra heel", 0.0, 1.0, 0.35, 0.05)
            with c2:
                xh_price = st.number_input("Extra heel price (₹)", 99, 3000, 699, 50)
                xh_cost = st.number_input("Extra heel cost (₹)", 20, 1500, 180, 10)
                channel = st.slider("Marketplace + payment fees (% of net)", 0.0, 0.45, 0.20, 0.01)
                returns = st.slider("Returns & damages (% of net)", 0.0, 0.30, 0.08, 0.01)
            with c3:
                ship = st.number_input("Shipping & packaging per order (₹)", 0, 500, 120, 10)
                cac = st.number_input("Marketing cost per order, CAC (₹)", 0, 3000, 400, 25)
                fixed = st.number_input("Fixed costs per month (₹)", 0, 5000000, 350000, 25000, help="Team, warehouse, software, rent")
                setup = st.number_input("One-time setup (₹)", 0, 20000000, 2500000, 100000,
                                        help="Design & moulds, design registration, first inventory, website, launch")
            c1, c2, c3 = st.columns(3)
            start = c1.number_input("Orders in month 1", 10, 20000, 300, 50)
            growth = c2.slider("Monthly order growth", 0.0, 0.30, 0.10, 0.01)
            cap = c3.number_input("Max orders per month (capacity / market ceiling)", 100, 100000, 3000, 250)
        sim, unit, be, need = A.break_even(price, unit_cost, xh_price, xh_cost, attach, gst, channel, returns, ship, cac,
                                           fixed, setup, start, growth, 36, cap)
        low_point = -sim["Cumulative cash"].min()
        kpis([(inr(unit["Contribution margin"]), "Contribution per order", ACCENT if unit["Contribution margin"] > 0 else DANGER),
              (f"{need:,.0f}" if np.isfinite(need) else "Never", "Orders/month to cover fixed costs"),
              (f"Month {be}" if be else "Not in 36 months", "Cumulative break-even", ACCENT if be else DANGER),
              (inr(low_point), "Cash needed before break-even", RISK)])
        c1, c2 = st.columns([1.3, 1])
        with c1:
            f, ax = fig(7, 4)
            cc = sim["Cumulative cash"] / 1e5
            ax.fill_between(sim["Month"], cc, 0, where=cc < 0, color=RISK, alpha=.18, interpolate=True)
            ax.fill_between(sim["Month"], cc, 0, where=cc >= 0, color=MARK2, alpha=.3, interpolate=True)
            ax.plot(sim["Month"], cc, color=INK, lw=2)
            ax.axhline(0, color=MUTED, lw=.8)
            if be:
                ax.axvline(be, color=MARK, ls="--", lw=1.4)
                ax.text(be, cc.min() * .9, f" break-even: month {be}", fontsize=9, fontweight="bold", color=MARK)
            ax.set_title("Cumulative cash position (₹ lakh)"); ax.set_xlabel("Month")
            show(f)
            low_m = int(sim.loc[sim["Cumulative cash"].idxmin(), "Month"])
            explain("Cumulative cash over 36 months: setup cost first, then monthly profit or loss. Orange = cash below zero.",
                    f"Cash bottoms out at -{inr(low_point)} in month {low_m} and "
                    + (f"turns positive in month {be}." if be else "does not turn positive within 36 months."),
                    (f"Raise at least {inr(low_point * 1.25)} (the low point plus a 25% buffer)." if be else
                     "Not sustainable under these assumptions: lower CAC or COGS, or raise the price, before launching."))
        with c2:
            u = unit.drop("Customer pays (incl. GST)")
            f, ax = fig(5, 4)
            colors = [MARK if i == "Contribution margin" and v > 0 else (DANGER if i == "Contribution margin" else NEUTRAL_LIGHT)
                      for i, v in zip(u.index[::-1], u.values[::-1])]
            ax.barh(u.index[::-1], u.values[::-1], color=colors, height=0.6)
            for i, v in enumerate(u.values[::-1]):
                ax.text(v if v > 0 else 0, i, f" ₹{v:,.0f}", va="center", fontsize=8.5)
            ax.set_title(f"Where ₹{unit['Customer pays (incl. GST)']:,.0f} per order goes"); ax.grid(False); ax.set_xticks([])
            show(f)
            costs = u.drop("Contribution margin")
            explain("Each bar is a slice of what the customer pays per order (kit plus expected add-on heels).",
                    f"Largest cost: **{costs.idxmax()}** ({inr(costs.max())}, {costs.max() / unit['Customer pays (incl. GST)']:.0%} "
                    f"of the price). Contribution: {inr(unit['Contribution margin'])} "
                    f"({unit['Contribution margin'] / unit['Customer pays (incl. GST)']:.0%}).",
                    f"Work on {costs.idxmax()} first: it is the biggest cost per order.")
        prices = [1999, 2499, 2999, 3499, 3999]
        cacs = [250, 400, 600, 800]
        vals = np.full((len(prices), len(cacs)), np.nan)
        for i, p in enumerate(prices):
            for j, c in enumerate(cacs):
                g = 0.05 if p <= GST_THRESHOLD else 0.18
                _, _, b, _ = A.break_even(p, unit_cost, xh_price, xh_cost, attach, g, channel, returns, ship, c, fixed,
                                          setup, start, growth, 60, cap)
                vals[i, j] = b if b else np.nan
        labs = np.where(np.isnan(vals), ">60", np.char.add("M", np.nan_to_num(vals).astype(int).astype(str)))
        f, ax = fig(8, 3.4)
        failed = np.isnan(vals)
        sns.heatmap(np.where(failed, 1, np.nan), mask=~failed, cmap=matplotlib.colors.ListedColormap(["#FDE2E2"]), cbar=False,
                    linewidths=2, linecolor="white", ax=ax)
        sns.heatmap(vals, mask=failed, annot=labs, fmt="", cmap=SEQ_CMAP.reversed(), vmin=np.nanmin(vals) if (~failed).any() else 0,
                    vmax=60, cbar=False, linewidths=2, linecolor="white", ax=ax, annot_kws={"size": 11, "weight": "bold"},
                    xticklabels=[f"CAC ₹{c}" for c in cacs], yticklabels=[f"₹{p:,}" for p in prices])
        for i, j in zip(*np.where(failed)):
            ax.text(j + 0.5, i + 0.5, ">60", ha="center", va="center", color=DANGER, fontweight="bold", fontsize=11)
        ax.set_title("Break-even month by kit price and marketing cost per order (darker = sooner; red = not within 60 months)")
        ax.set_ylabel("Kit price"); ax.grid(False)
        show(f)
        pi_ = prices.index(ENTRY_PRICE) if ENTRY_PRICE in prices else 1
        cj = cacs.index(400) if 400 in cacs else 1

        def span(a):
            a = a[~np.isnan(a)]
            return (a.max() - a.min()) if len(a) > 1 else np.nan
        cac_span, price_span = span(vals[pi_, :]), span(vals[:, cj])
        n_fail = int(failed.sum())
        explain("The same simulation repeated for 5 prices × 4 marketing costs (other assumptions as above).",
                f"Fastest break-even: month {int(np.nanmin(vals)) if (~failed).any() else '-'}; {n_fail} of {vals.size} "
                f"combinations do not break even within 60 months. At {inr(prices[pi_])}, moving CAC from ₹{cacs[0]} to ₹{cacs[-1]} "
                f"shifts break-even by {cac_span:.0f} months; at CAC ₹{cacs[cj]}, moving price from {inr(prices[0])} to "
                f"{inr(prices[-1])} shifts it by {price_span:.0f} months." if np.isfinite(cac_span) and np.isfinite(price_span) else
                f"{n_fail} of {vals.size} combinations do not break even within 60 months.",
                ("Marketing cost matters more than price: keep CAC low through organic reels, influencer seeding and referrals."
                 if np.isfinite(cac_span) and np.isfinite(price_span) and cac_span > price_span else
                 "Price matters at least as much as marketing cost: protect the price and avoid deep discounts."))

    limitations()


def limitations():
    st.divider()
    st.markdown("#### Limitations and next steps")
    c1, c2 = st.columns(2)
    c1.markdown(
        f"<div class='card' style='--c:{RISK}'><div class='h'>Limitations of this analysis</div><ul>"
        f"<li>The data is simulated; every number is a test of the method, not a market fact.</li>"
        f"<li>Stated intent overstates real purchases: {pct(yes_rate)} say Yes today, real conversion will be lower.</li>"
        f"<li>The sample is mainly metro: {pct(tier1_share)} of respondents live in Tier-1 cities.</li>"
        f"<li>Costs, CAC and growth in the break-even model are planning assumptions.</li>"
        f"<li>GST rate (5% up to ₹{GST_THRESHOLD:,}, 18% above) to be confirmed with a CA.</li></ul></div>",
        unsafe_allow_html=True)
    c2.markdown(
        f"<div class='card' style='--c:{ACCENT}'><div class='h'>Next steps</div><ul>"
        "<li>Run the real survey and upload the export in the sidebar; every chart and model re-runs.</li>"
        "<li>Run a pre-order pilot to measure real conversion and price acceptance.</li>"
        "<li>Analyse the open-text answers in Q30 for objections and feature ideas.</li>"
        "<li>Validate demand and price in Tier-2 cities before expanding there.</li></ul></div>",
        unsafe_allow_html=True)


# =================================================================== router
section = st.session_state.section
if section == "Home":
    page_home()
else:
    top_nav()
    if section == "Survey & data":
        page_data()
    else:
        q, d = SECTIONS[section]
        header(f"{PRODUCT_NAME} · {section} analytics", q, d)
        {"Descriptive": page_descriptive, "Diagnostic": page_diagnostic,
         "Predictive": page_predictive, "Prescriptive": page_prescriptive}[section]()
