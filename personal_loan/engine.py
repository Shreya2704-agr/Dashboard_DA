"""
Universal Bank - Personal Loan analytics engine.

Pure-Python / pandas / scikit-learn. No UI code lives here, so the same functions can later
power an API, a batch scoring job or a different front end.

Pipeline
    load_data()         -> clean data + business-friendly bands
    train_models()      -> classification (Logistic Regression, Decision Tree, Random Forest)
    k_diagnostics()     -> elbow + silhouette for choosing k
    segment()           -> clustering (K-Means or Gaussian Mixture / latent profiles) with auto names
    basket()/mine_rules -> association rule mining (Apriori)
    gains_table()/simulate_budget() -> prescriptive targeting maths
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import silhouette_score

DATA_PATH = Path(__file__).parent / "data" / "UniversalBank.csv"
SEED = 42

TARGET = "Personal Loan"
PRODUCTS = ["Securities Account", "CD Account", "Online", "CreditCard"]
PRODUCT_LABEL = {"Securities Account": "Securities", "CD Account": "CD Account", "Online": "Online Banking",
                 "CreditCard": "Bank Credit Card", "Has Mortgage": "Mortgage", TARGET: "Personal Loan"}
EDU_LABEL = {1: "Undergrad", 2: "Graduate", 3: "Advanced/Prof."}
EDU_ORDER = list(EDU_LABEL.values())
FAMILY_ORDER = [f"Family of {i}" for i in (1, 2, 3, 4)]
TIERS = ["Hot (50%+)", "Warm (20-50%)", "Cool (5-20%)", "Cold (<5%)"]

# Experience is ~perfectly correlated with Age (r≈0.99) and ZIP is an identifier, so both are left out.
MODEL_FEATURES = ["Income", "CCAvg", "Family", "Education", "CD Account", "Mortgage", "Age",
                  "Securities Account", "Online", "CreditCard"]
# Clustering uses customer *profile* only. Products and the loan flag are kept out on purpose so we can
# then ask "what does each segment own / buy?" without the answer leaking into the segmentation.
CLUSTER_FEATURES = ["Income", "CCAvg", "Family", "Education", "Mortgage"]

FEATURE_LABEL = {"Income": "Annual income", "CCAvg": "Card spend / month", "Family": "Family size",
                 "Education": "Education level", "CD Account": "Has CD account", "Mortgage": "Mortgage value",
                 "Age": "Age", "Securities Account": "Has securities acct", "Online": "Uses online banking",
                 "CreditCard": "Has bank credit card", "Experience": "Experience"}

BANDS = {
    "Income Band": ("Income", [0, 50, 75, 100, 125, 150, 250], ["<$50k", "$50-75k", "$75-100k", "$100-125k", "$125-150k", "$150k+"]),
    "Age Band": ("Age", [0, 30, 40, 50, 60, 100], ["23-30", "31-40", "41-50", "51-60", "61-67"]),
    "Card Spend Band": ("CCAvg", [-0.01, 1, 2, 3, 5, 100], ["<$1k/mo", "$1-2k/mo", "$2-3k/mo", "$3-5k/mo", "$5k+/mo"]),
    "Mortgage Band": ("Mortgage", [-1, 0, 100, 200, 1000], ["No mortgage", "<$100k", "$100-200k", "$200k+"]),
}


# ============================================================== 1. data
def _region(z: int) -> str:
    p = z // 1000
    return {90: "Los Angeles", 91: "Los Angeles", 92: "San Diego & Inland", 93: "Central California",
            94: "SF Bay Area", 95: "Sacramento & North", 96: "Far North"}.get(p, "Unknown")


def load_data(path: Path | str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df = df.rename(columns={"ZIP Code": "ZIPCode"})
    df["Experience"] = df["Experience"].clip(lower=0)            # 52 negative values are data-entry errors
    df["Region"] = df["ZIPCode"].map(_region)
    df["Education Level"] = pd.Categorical(df["Education"].map(EDU_LABEL), EDU_ORDER, ordered=True)
    df["Family Size"] = pd.Categorical("Family of " + df["Family"].astype(str), FAMILY_ORDER, ordered=True)
    df["Has Mortgage"] = (df["Mortgage"] > 0).astype(int)
    df["Products Held"] = df[PRODUCTS].sum(axis=1)
    for name, (col, bins, labels) in BANDS.items():
        df[name] = pd.cut(df[col], bins=bins, labels=labels, ordered=True)
    df["Loan Status"] = np.where(df[TARGET] == 1, "Accepted", "Declined")
    return df


def data_quality(raw_path: Path | str = DATA_PATH) -> dict:
    raw = pd.read_csv(raw_path)
    raw.columns = raw.columns.str.strip()
    return {"rows": len(raw), "columns": raw.shape[1], "missing": int(raw.isna().sum().sum()),
            "duplicates": int(raw.duplicated().sum()), "negative_experience": int((raw["Experience"] < 0).sum()),
            "bad_zip": int((raw["ZIP Code"] < 90000).sum()),
            "age_experience_corr": float(raw["Age"].corr(raw["Experience"]))}


def rate_table(df: pd.DataFrame, by: str | list[str]) -> pd.DataFrame:
    """Acceptance rate, volume and share of loans for one or more grouping columns."""
    g = df.groupby(by, observed=True)[TARGET].agg(customers="size", loans="sum").reset_index()
    g["rate"] = g["loans"] / g["customers"]
    g["share_customers"] = g["customers"] / g["customers"].sum()
    g["share_loans"] = g["loans"] / max(g["loans"].sum(), 1)
    return g


# ============================================================== 2. classification
def _models() -> dict:
    return {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "Decision Tree": DecisionTreeClassifier(max_depth=4, min_samples_leaf=25, random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=3, n_jobs=-1, random_state=SEED),
    }


def train_models(df: pd.DataFrame) -> dict:
    """70/30 stratified hold-out for honest metrics, then 5-fold out-of-fold scores for every customer."""
    X, y = df[MODEL_FEATURES], df[TARGET]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=SEED)
    out = {"metrics": [], "roc": {}, "cm": {}, "fitted": {}}
    for name, m in _models().items():
        m.fit(Xtr, ytr)
        p = m.predict_proba(Xte)[:, 1]
        pred = (p >= 0.5).astype(int)
        fpr, tpr, _ = roc_curve(yte, p)
        out["roc"][name] = (fpr, tpr)
        out["cm"][name] = confusion_matrix(yte, pred)
        out["metrics"].append({"Model": name, "AUC": roc_auc_score(yte, p), "Accuracy": accuracy_score(yte, pred),
                               "Precision": precision_score(yte, pred, zero_division=0),
                               "Recall": recall_score(yte, pred), "F1": f1_score(yte, pred)})
        out["fitted"][name] = m
    met = pd.DataFrame(out["metrics"]).sort_values("AUC", ascending=False).reset_index(drop=True)
    out["metrics"] = met
    best = met.loc[0, "Model"]
    out["best"] = best
    out["test_size"] = len(yte)

    # Out-of-fold propensity for every customer: each score comes from a model that never saw that customer.
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    out["scores"] = cross_val_predict(_models()[best], X, y, cv=cv, method="predict_proba")[:, 1]

    # Refit on all data for the what-if scorer and explanations.
    full = {n: m.fit(X, y) for n, m in _models().items()}
    out["full"] = full
    rf = full["Random Forest"]
    out["importance"] = (pd.DataFrame({"feature": MODEL_FEATURES, "importance": rf.feature_importances_})
                         .assign(label=lambda d: d.feature.map(FEATURE_LABEL))
                         .sort_values("importance", ascending=False).reset_index(drop=True))
    lr = full["Logistic Regression"]
    coefs = lr[-1].coef_[0]
    out["odds"] = (pd.DataFrame({"feature": MODEL_FEATURES, "coef": coefs, "odds_ratio": np.exp(coefs),
                                 "sd": X.std().values})
                   .assign(label=lambda d: d.feature.map(FEATURE_LABEL))
                   .sort_values("coef", key=np.abs, ascending=False).reset_index(drop=True))
    out["tree_rules"] = tree_rules(full["Decision Tree"], X, y)
    return out


def _fmt_bound(f: str, lo: float | None, hi: float | None) -> str:
    if f in PRODUCTS:
        name = PRODUCT_LABEL[f]
        return f"has {name}" if lo is not None else f"no {name}"
    if f == "Education":
        levels = [EDU_LABEL[v] for v in (1, 2, 3) if (lo is None or v > lo) and (hi is None or v <= hi)]
        return "education " + " or ".join(levels)
    if f == "Family":
        vals = [v for v in (1, 2, 3, 4) if (lo is None or v > lo) and (hi is None or v <= hi)]
        return f"family of {vals[0]}" if len(vals) == 1 else f"family of {vals[0]}-{vals[-1]}"
    unit = {"Income": ("income", lambda v: f"${v:,.0f}k"), "Mortgage": ("mortgage", lambda v: f"${v:,.0f}k"),
            "CCAvg": ("card spend", lambda v: f"${v:,.1f}k/mo"), "Age": ("age", lambda v: f"{v:.0f}")}[f]
    label, fmt = unit
    if lo is not None and hi is not None:
        return f"{label} {fmt(lo)}-{fmt(hi)}"
    return f"{label} above {fmt(lo)}" if lo is not None else f"{label} up to {fmt(hi)}"


def tree_rules(tree: DecisionTreeClassifier, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Turn every leaf of the decision tree into a plain-English 'IF ... THEN x% accept' rule."""
    t = tree.tree_
    leaves = tree.apply(X)
    stats = pd.DataFrame({"leaf": leaves, "y": y.values}).groupby("leaf")["y"].agg(["size", "sum"])
    rows = []

    def walk(node, bounds):
        if t.children_left[node] == -1:
            parts = [_fmt_bound(f, lo, hi) for f, (lo, hi) in bounds.items()]
            n, k = stats.loc[node, "size"], stats.loc[node, "sum"]
            rows.append({"rule": " AND ".join(parts) if parts else "everyone", "customers": int(n),
                         "loans": int(k), "rate": k / n})
            return
        f, thr = X.columns[t.feature[node]], t.threshold[node]
        lo, hi = bounds.get(f, (None, None))
        walk(t.children_left[node], {**bounds, f: (lo, thr if hi is None else min(hi, thr))})
        walk(t.children_right[node], {**bounds, f: (thr if lo is None else max(lo, thr), hi)})

    walk(0, {})
    r = pd.DataFrame(rows).sort_values("rate", ascending=False).reset_index(drop=True)
    r["share_loans"] = r["loans"] / r["loans"].sum()
    return r


def gains_table(scores: np.ndarray, y: np.ndarray, steps: int = 20) -> pd.DataFrame:
    """Cumulative gains: contact the top x% by score -> what % of all borrowers do we reach?"""
    order = np.argsort(-scores)
    ys = np.asarray(y)[order]
    n, total = len(ys), ys.sum()
    rows = []
    for pct in np.linspace(0, 1, steps + 1):
        k = int(round(pct * n))
        hit = ys[:k].sum()
        rows.append({"pct_contacted": pct, "contacts": k, "loans": int(hit), "pct_loans": hit / total,
                     "conv_rate": hit / k if k else np.nan, "lift": (hit / k) / (total / n) if k else np.nan})
    return pd.DataFrame(rows)


def simulate_budget(scores, y, budget_share: float, cost_per_contact: float, value_per_loan: float) -> dict:
    """Compare 'contact everyone' (last campaign) with 'contact only the top budget_share by score'."""
    y = np.asarray(y)
    n, total = len(y), int(y.sum())
    k = int(round(budget_share * n))
    order = np.argsort(-scores)
    hit = int(y[order[:k]].sum())
    base_cost, new_cost = n * cost_per_contact, k * cost_per_contact
    return {"n": n, "k": k, "base_loans": total, "new_loans": hit, "lost_loans": total - hit,
            "base_cost": base_cost, "new_cost": new_cost, "saving": base_cost - new_cost,
            "base_rate": total / n, "new_rate": hit / k if k else 0.0, "capture": hit / total if total else 0.0,
            "base_cpa": base_cost / total if total else np.nan, "new_cpa": new_cost / hit if hit else np.nan,
            "base_profit": total * value_per_loan - base_cost, "new_profit": hit * value_per_loan - new_cost,
            "cutoff": float(scores[order[k - 1]]) if k else 1.0}


def tier(scores) -> pd.Categorical:
    return pd.cut(np.asarray(scores), [-0.01, 0.05, 0.2, 0.5, 1.01], labels=TIERS[::-1]).reorder_categories(TIERS, ordered=True)


def smallest_budget_for(scores, y, capture: float = 0.9) -> float:
    order = np.argsort(-scores)
    cum = np.cumsum(np.asarray(y)[order]) / np.asarray(y).sum()
    return (int(np.searchsorted(cum, capture)) + 1) / len(y)


# ============================================================== 3. clustering
def _cluster_matrix(df: pd.DataFrame) -> np.ndarray:
    X = df[CLUSTER_FEATURES].astype(float).copy()
    X["Mortgage"] = np.log1p(X["Mortgage"])      # heavy right skew, most customers have 0
    X["CCAvg"] = np.log1p(X["CCAvg"])
    return StandardScaler().fit_transform(X)


def k_diagnostics(df: pd.DataFrame, ks=range(2, 9)) -> pd.DataFrame:
    Z = _cluster_matrix(df)
    rows = []
    for k in ks:
        km = KMeans(k, n_init=10, random_state=SEED).fit(Z)
        rows.append({"k": k, "inertia": km.inertia_,
                     "silhouette": silhouette_score(Z, km.labels_, sample_size=3000, random_state=0)})
    return pd.DataFrame(rows)


def _name_segments(prof: pd.DataFrame, overall: pd.Series, sd: pd.Series) -> list[str]:
    """Give each cluster a business name: income tier + its most distinctive trait."""
    traits = {  # (feature, direction) -> label
        ("CCAvg", 1): "Big Spenders", ("Mortgage", 1): "Homeowners", ("Family", 1): "Families",
        ("Family", -1): "Singles & Couples", ("Education", 1): "Professionals", ("Education", -1): "Undergrads",
        ("CCAvg", -1): "Light Spenders", ("Mortgage", -1): "Renters",
    }
    names = []
    for _, r in prof.iterrows():
        tier = "Affluent" if r["Income"] >= 100 else "Mid-income" if r["Income"] >= 60 else "Mass-market"
        z = {f: (r[f] - overall[f]) / sd[f] for f in ["CCAvg", "Mortgage", "Family", "Education"]}
        ranked = sorted(z.items(), key=lambda kv: -abs(kv[1]))
        labels = [traits[(f, 1 if v > 0 else -1)] for f, v in ranked]
        names.append((tier, labels))
    out, used = [], set()
    for tier, labels in names:
        name = f"{tier} {labels[0]}"
        if name in used:
            name = f"{tier} {labels[0]} · {labels[1]}"
        i = 2
        while name in used:
            name, i = f"{tier} {labels[0]} ({i})", i + 1
        used.add(name)
        out.append(name)
    return out


def segment(df: pd.DataFrame, k: int = 4, method: str = "K-Means") -> tuple[pd.Series, pd.DataFrame]:
    """Cluster customers on profile features; return segment names (ordered by loan rate) and a profile table."""
    Z = _cluster_matrix(df)
    if method == "K-Means":
        labels = KMeans(k, n_init=10, random_state=SEED).fit_predict(Z)
    else:  # Gaussian Mixture = latent profile analysis (the continuous-variable cousin of LCA)
        labels = GaussianMixture(k, covariance_type="full", n_init=3, random_state=SEED).fit(Z).predict(Z)
    tmp = df.assign(_c=labels)
    num = ["Income", "CCAvg", "Family", "Education", "Mortgage", "Age"]
    prof = tmp.groupby("_c")[num + PRODUCTS + ["Has Mortgage", TARGET]].mean()
    prof["customers"] = tmp.groupby("_c").size()
    prof["loans"] = tmp.groupby("_c")[TARGET].sum()
    prof = prof.sort_values(TARGET, ascending=False)
    prof["Segment"] = _name_segments(prof, df[num].mean(), df[num].std())
    mapping = prof["Segment"].to_dict()
    prof = prof.reset_index(drop=True)
    prof["share_customers"] = prof["customers"] / prof["customers"].sum()
    prof["share_loans"] = prof["loans"] / max(prof["loans"].sum(), 1)
    seg = pd.Series(pd.Categorical(tmp["_c"].map(mapping), prof["Segment"].tolist(), ordered=True), index=df.index)
    return seg, prof


# ============================================================== 4. association rules
PROFILE_ITEMS = {
    "Income $100k+": lambda d: d["Income"] >= 100,
    "Graduate+": lambda d: d["Education"] >= 2,
    "Family 3+": lambda d: d["Family"] >= 3,
    "Card spend $2.5k+": lambda d: d["CCAvg"] >= 2.5,
}


def basket(df: pd.DataFrame, include_profile: bool = True) -> pd.DataFrame:
    """One row per customer, one True/False column per 'item' (product held or profile flag)."""
    b = pd.DataFrame({PRODUCT_LABEL[p]: df[p].astype(bool) for p in PRODUCTS + ["Has Mortgage", TARGET]})
    if include_profile:
        for name, f in PROFILE_ITEMS.items():
            b[name] = f(df).astype(bool)
    return b


def mine_rules(df: pd.DataFrame, min_support=0.01, min_confidence=0.2, include_profile=True,
               loan_only=False, max_len=4) -> pd.DataFrame:
    b = basket(df, include_profile)
    if len(b) == 0:
        return pd.DataFrame()
    fi = apriori(b, min_support=min_support, use_colnames=True, max_len=max_len)
    if fi.empty or fi["itemsets"].map(len).max() < 2:
        return pd.DataFrame()
    r = association_rules(fi, metric="confidence", min_threshold=min_confidence, num_itemsets=len(b))
    if r.empty:
        return r
    r = r[r["consequents"].map(len) == 1]
    if loan_only:
        r = r[r["consequents"].map(lambda s: "Personal Loan" in s)]
    else:
        r = r[r["antecedents"].map(lambda s: "Personal Loan" not in s)]
    r = r.assign(IF=r["antecedents"].map(lambda s: " + ".join(sorted(s))),
                 THEN=r["consequents"].map(lambda s: next(iter(s))),
                 customers=(r["support"] * len(b)).round().astype(int),
                 n_items=r["antecedents"].map(len))
    return (r[["IF", "THEN", "support", "confidence", "lift", "customers", "n_items"]]
            .sort_values(["lift", "confidence"], ascending=False).reset_index(drop=True))


def lift_matrix(df: pd.DataFrame, include_profile=False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pairwise lift and conditional probability P(column | row) between items."""
    b = basket(df, include_profile).astype(float)
    n = len(b)
    p = b.mean()
    joint = (b.T @ b) / n
    lift = joint / np.outer(p, p)
    cond = joint.div(p, axis=0)          # row -> column
    return lift, cond
