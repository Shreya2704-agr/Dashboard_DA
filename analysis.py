"""
Analytics engine for the Campaign Intelligence dashboard.

Pipeline
--------
1. load_data()          -> clean raw CSV, engineer features
2. campaign_summary()   -> response rate, cost, revenue, profit per campaign
3. add_rfm_cltv()       -> RFM scores + RFM-based CLTV ($) per customer
4. segment_kmeans()     -> K-Means (k=3) on RFM -> High / Medium / Low CLTV
5. add_fatigue()        -> rule-based fatigue level + 0-100 fatigue index
6. train_models()       -> Logistic Reg / Random Forest / XGBoost propensity models
                           (5-fold out-of-fold scores) + SHAP drivers
7. gains_table()        -> cumulative gains / lift / ROI by targeting depth
8. association_rules_() -> Apriori rules: which traits lead to a "Yes" on campaign 6
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CAMPAIGNS = ["AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3", "AcceptedCmp4", "AcceptedCmp5", "Response"]
CAMPAIGN_LABELS = ["Cmp 1", "Cmp 2", "Cmp 3", "Cmp 4", "Cmp 5", "Cmp 6"]
PRODUCTS = ["MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts", "MntSweetProducts", "MntGoldProds"]
PRODUCT_LABELS = ["Wine", "Fruit", "Meat", "Fish", "Sweets", "Gold"]
CHANNELS = ["NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases"]
CHANNEL_LABELS = ["Web", "Catalog", "Store"]

SEGMENT_ORDER = ["High CLTV", "Medium CLTV", "Low CLTV"]
FATIGUE_ORDER = ["Engaged", "Low Fatigue", "Moderate Fatigue", "High Fatigue"]


# --------------------------------------------------------------------------- 1. data
def load_data(path: str = "data/marketing_campaign.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    raw_n = len(df)

    # --- cleaning --------------------------------------------------------
    df["Income"] = df.groupby("Education")["Income"].transform(lambda s: s.fillna(s.median()))
    df = df[df["Year_Birth"] >= 1920]          # 3 impossible birth years (1893, 1899, 1900)
    df = df[df["Income"] < 600_000]            # 1 extreme outlier (666,666)
    df["Marital_Status"] = df["Marital_Status"].replace({"Alone": "Single", "Absurd": "Single", "YOLO": "Single"})
    df["Education"] = df["Education"].replace({"2n Cycle": "Master"})
    df = df.drop(columns=["Z_CostContact", "Z_Revenue"])  # constants: cost 3 / revenue 11 per contact

    # --- feature engineering --------------------------------------------
    df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], dayfirst=True)
    ref_date = df["Dt_Customer"].max() + pd.Timedelta(days=1)
    df["Age"] = ref_date.year - df["Year_Birth"]
    df["Tenure_Days"] = (ref_date - df["Dt_Customer"]).dt.days
    df["Children"] = df["Kidhome"] + df["Teenhome"]
    df["Couple"] = df["Marital_Status"].isin(["Married", "Together"]).astype(int)
    df["Household"] = 1 + df["Couple"] + df["Children"]
    df["Monetary"] = df[PRODUCTS].sum(axis=1)
    df["Frequency"] = df[CHANNELS].sum(axis=1)
    df["AOV"] = df["Monetary"] / df["Frequency"].replace(0, 1)
    df["Deal_Share"] = df["NumDealsPurchases"] / df["Frequency"].replace(0, 1)
    df["Web_Conversion"] = df["NumWebPurchases"] / (df["NumWebVisitsMonth"] + 1)
    df["Prior_Accepts"] = df[CAMPAIGNS[:5]].sum(axis=1)
    df["Total_Accepts"] = df[CAMPAIGNS].sum(axis=1)
    df["Age_Band"] = pd.cut(df["Age"], [0, 35, 45, 55, 65, 120], labels=["<35", "35-44", "45-54", "55-64", "65+"])
    df["Income_Band"] = pd.qcut(df["Income"], 4, labels=["Q1 Low", "Q2", "Q3", "Q4 High"])
    df.attrs["raw_n"] = raw_n
    df.attrs["ref_date"] = ref_date
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------- 2. campaigns
def campaign_summary(df: pd.DataFrame, cost: float = 3, revenue: float = 11) -> pd.DataFrame:
    n = len(df)
    rows = []
    for col, lab in zip(CAMPAIGNS, CAMPAIGN_LABELS):
        acc = int(df[col].sum())
        rows.append({
            "Campaign": lab, "Responders": acc, "Contacted": n,
            "Response Rate": acc / n,
            "Cost": n * cost, "Revenue": acc * revenue,
            "Profit": acc * revenue - n * cost,
            "ROI": (acc * revenue - n * cost) / (n * cost),
        })
    out = pd.DataFrame(rows)
    out.attrs["break_even"] = cost / revenue
    return out


def acceptor_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Index (=100 is customer-base average) of key traits among each campaign's acceptors."""
    traits = {"Income": "Income", "Spend": "Monetary", "Purchases": "Frequency", "Age": "Age",
              "Children": "Children", "Deal share": "Deal_Share", "Web visits": "NumWebVisitsMonth",
              "Wine $": "MntWines", "Meat $": "MntMeatProducts", "Gold $": "MntGoldProds", "Tenure": "Tenure_Days"}
    base = {k: df[v].mean() for k, v in traits.items()}
    data = {}
    for col, lab in zip(CAMPAIGNS, CAMPAIGN_LABELS):
        sub = df[df[col] == 1]
        data[lab] = {k: 100 * sub[v].mean() / base[k] for k, v in traits.items()}
    return pd.DataFrame(data)


def co_acceptance(df: pd.DataFrame) -> pd.DataFrame:
    """P(accept column campaign | accepted row campaign)."""
    m = df[CAMPAIGNS].values
    out = np.zeros((6, 6))
    for i in range(6):
        sel = m[:, i] == 1
        out[i] = m[sel].mean(axis=0) if sel.any() else 0
    return pd.DataFrame(out, index=CAMPAIGN_LABELS, columns=CAMPAIGN_LABELS)


# --------------------------------------------------------------------------- 3. RFM + CLTV
def add_rfm_cltv(df: pd.DataFrame, margin: float = 0.25, discount: float = 0.10) -> pd.DataFrame:
    """
    RFM scores (1-5 quintiles) and an RFM-driven CLTV:

        annual_value = Monetary / tenure_years                (M and F: spend velocity)
        retention r  = 0.50 + 0.10 * (R_score - 1)            (R: recent buyers stay)
        CLTV         = annual_value * margin * r / (1 + d - r)  (standard retention-discount CLV)
    """
    df = df.copy()
    rank = lambda s: s.rank(method="first")
    df["R_Score"] = pd.qcut(rank(-df["Recency"]), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    df["F_Score"] = pd.qcut(rank(df["Frequency"]), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    df["M_Score"] = pd.qcut(rank(df["Monetary"]), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    df["RFM_Cell"] = df["R_Score"].astype(str) + df["F_Score"].astype(str) + df["M_Score"].astype(str)
    df["RFM_Score"] = df[["R_Score", "F_Score", "M_Score"]].sum(axis=1)

    tenure_years = (df["Tenure_Days"] / 365).clip(lower=0.5)   # floor: avoid inflating brand-new customers
    df["Annual_Value"] = df["Monetary"] / tenure_years
    df["Retention"] = 0.50 + 0.10 * (df["R_Score"] - 1)
    df["CLTV"] = df["Annual_Value"] * margin * df["Retention"] / (1 + discount - df["Retention"])
    return df


def segment_kmeans(df: pd.DataFrame, k: int = 3, seed: int = 42):
    """K-Means on standardised RFM; clusters are named by their mean CLTV."""
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score

    X = np.column_stack([-df["Recency"], np.log1p(df["Frequency"]), np.log1p(df["Monetary"]), np.log1p(df["CLTV"])])
    Xs = StandardScaler().fit_transform(X)

    diag = []
    for kk in range(2, 9):
        km = KMeans(n_clusters=kk, n_init=10, random_state=seed).fit(Xs)
        diag.append({"k": kk, "Inertia": km.inertia_, "Silhouette": silhouette_score(Xs, km.labels_, sample_size=1500, random_state=seed)})

    km = KMeans(n_clusters=k, n_init=20, random_state=seed).fit(Xs)
    df = df.copy()
    df["Cluster"] = km.labels_
    order = df.groupby("Cluster")["CLTV"].mean().sort_values(ascending=False).index
    names = dict(zip(order, SEGMENT_ORDER[:k] if k == 3 else [f"Tier {i+1}" for i in range(k)]))
    df["Segment"] = pd.Categorical(df["Cluster"].map(names), categories=list(names.values()), ordered=True)
    return df, pd.DataFrame(diag)


# --------------------------------------------------------------------------- 4. fatigue
def add_fatigue(df: pd.DataFrame) -> pd.DataFrame:
    """
    Campaigns are chronological (Cmp1 ... Cmp6=Response).
      Ignore_Streak = consecutive campaigns ignored since last acceptance (6 = never accepted)

      Engaged          : accepted >=2 campaigns AND accepted one of the last 2
      Low Fatigue      : last acceptance within the last 3 campaigns
      Moderate Fatigue : accepted before, but ignored the last >=3 (lapsed responder)
      High Fatigue     : ignored all 6 campaigns

    Fatigue_Index (0-100) = 60% ignore streak + 25% lifetime non-acceptance + 15% purchase recency
    """
    df = df.copy()
    m = df[CAMPAIGNS].values
    last = np.where(m.any(axis=1), 5 - np.argmax(m[:, ::-1], axis=1), -1)   # index 0..5, -1 never
    df["Last_Accept"] = np.where(last >= 0, np.array(CAMPAIGN_LABELS)[last.clip(0)], "Never")
    df["Ignore_Streak"] = np.where(last >= 0, 5 - last, 6)
    acc = df["Total_Accepts"]
    lvl = np.select(
        [(acc >= 2) & (df["Ignore_Streak"] <= 1), (acc >= 1) & (df["Ignore_Streak"] <= 2), acc >= 1],
        FATIGUE_ORDER[:3], default=FATIGUE_ORDER[3])
    df["Fatigue"] = pd.Categorical(lvl, categories=FATIGUE_ORDER, ordered=True)
    df["Fatigue_Index"] = (100 * (0.60 * df["Ignore_Streak"] / 6 + 0.25 * (1 - acc / 6) + 0.15 * df["Recency"] / 99)).round(1)
    # still buying but blind to campaigns = "silent loyal" (fix the channel, not the customer)
    df["Silent_Loyal"] = ((df["Fatigue"] == "High Fatigue") & (df["R_Score"] >= 3) & (df["F_Score"] >= 4)).astype(int)
    return df


# --------------------------------------------------------------------------- 5. ML
FEATURES = ["Income", "Age", "Children", "Couple", "Tenure_Days", "Recency", "Monetary", "Frequency", "AOV",
            "MntWines", "MntMeatProducts", "MntGoldProds", "MntFishProducts", "NumDealsPurchases",
            "NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases", "NumWebVisitsMonth", "Deal_Share",
            "AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3", "AcceptedCmp4", "AcceptedCmp5", "Complain",
            "Edu_PhD", "Edu_Basic"]


def _X(df):
    X = df.copy()
    X["Edu_PhD"] = (X["Education"] == "PhD").astype(int)
    X["Edu_Basic"] = (X["Education"] == "Basic").astype(int)
    return X[FEATURES]


def train_models(df: pd.DataFrame, seed: int = 42) -> dict:
    """Out-of-fold (5-fold stratified) propensity of responding to campaign 6."""
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score
    from xgboost import XGBClassifier

    X, y = _X(df), df["Response"].values
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    spw = (y == 0).sum() / (y == 1).sum()
    models = {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
        "Random Forest": RandomForestClassifier(n_estimators=400, min_samples_leaf=5, class_weight="balanced", random_state=seed, n_jobs=-1),
        "XGBoost": XGBClassifier(n_estimators=400, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                                 scale_pos_weight=spw, eval_metric="auc", random_state=seed, n_jobs=-1),
    }
    res = {"y": y, "scores": {}, "metrics": [], "roc": {}}
    for name, mdl in models.items():
        p = cross_val_predict(mdl, X, y, cv=cv, method="predict_proba")[:, 1]
        fpr, tpr, _ = roc_curve(y, p)
        res["scores"][name] = p
        res["roc"][name] = (fpr, tpr)
        res["metrics"].append({"Model": name, "ROC-AUC": roc_auc_score(y, p), "PR-AUC": average_precision_score(y, p)})
    res["metrics"] = pd.DataFrame(res["metrics"]).sort_values("ROC-AUC", ascending=False)
    res["best"] = res["metrics"].iloc[0]["Model"]

    # SHAP on a full-data XGBoost fit (explanations, not evaluation)
    import shap
    xgb = models["XGBoost"].fit(X, y)
    sv = shap.TreeExplainer(xgb).shap_values(X)
    res["shap"] = pd.DataFrame(sv, columns=FEATURES)
    res["X"] = X
    return res


def gains_table(y: np.ndarray, score: np.ndarray, cost: float = 3, revenue: float = 11) -> pd.DataFrame:
    order = np.argsort(-score)
    ys = y[order]
    n, pos = len(y), y.sum()
    rows = []
    for pct in range(5, 101, 5):
        k = int(round(n * pct / 100))
        hits = ys[:k].sum()
        rows.append({"Target %": pct, "Contacted": k, "Responders": int(hits),
                     "Capture %": 100 * hits / pos, "Response Rate": hits / k,
                     "Lift": (hits / k) / (pos / n), "Profit": hits * revenue - k * cost})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- 6. association rules
def association_rules_(df: pd.DataFrame, min_support: float = 0.03, min_conf: float = 0.3) -> pd.DataFrame:
    from mlxtend.frequent_patterns import apriori, association_rules

    q = lambda c: df[c] >= df[c].quantile(0.75)
    b = pd.DataFrame({
        "High Income": q("Income"), "High Wine": q("MntWines"), "High Meat": q("MntMeatProducts"),
        "High Gold": q("MntGoldProds"), "Catalog Buyer": df["NumCatalogPurchases"] >= 4,
        "Deal Seeker": df["Deal_Share"] >= 0.3, "No Kids": df["Children"] == 0, "Has Teen": df["Teenhome"] > 0,
        "Single": df["Couple"] == 0, "PhD": df["Education"] == "PhD", "Recent Buyer": df["Recency"] <= 30,
        "New Customer": df["Tenure_Days"] <= df["Tenure_Days"].median(),
        "Accepted Cmp1": df["AcceptedCmp1"] == 1, "Accepted Cmp3": df["AcceptedCmp3"] == 1,
        "Accepted Cmp4": df["AcceptedCmp4"] == 1, "Accepted Cmp5": df["AcceptedCmp5"] == 1,
        "Responded Cmp6": df["Response"] == 1,
    })
    fi = apriori(b, min_support=min_support, use_colnames=True, max_len=4)
    r = association_rules(fi, metric="confidence", min_threshold=min_conf)
    r = r[r["consequents"].apply(lambda s: s == frozenset({"Responded Cmp6"}))]
    r["Rule"] = r["antecedents"].apply(lambda s: " + ".join(sorted(s)))
    return r[["Rule", "support", "confidence", "lift"]].sort_values("lift", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- 7. strategy
def recommend(segment: str, fatigue: str) -> str:
    table = {
        ("High CLTV", "Engaged"): "VIP early-access & premium bundles (wine/meat); every campaign, personalised",
        ("High CLTV", "Low Fatigue"): "Personalised catalog + loyalty points; keep cadence monthly",
        ("High CLTV", "Moderate Fatigue"): "Win-back: exclusive offer, new channel, pause mass emails 4-6 wks",
        ("High CLTV", "High Fatigue"): "Stop generic campaigns; service-led touch (concierge, tasting events)",
        ("Medium CLTV", "Engaged"): "Upsell to premium categories; cross-sell gold & fish",
        ("Medium CLTV", "Low Fatigue"): "Targeted category offers based on basket; bi-monthly",
        ("Medium CLTV", "Moderate Fatigue"): "Re-activation test: deals + web retargeting, reduced frequency",
        ("Medium CLTV", "High Fatigue"): "Suppress from campaigns; only in-store / web personalisation",
        ("Low CLTV", "Engaged"): "Low-cost digital offers (deals); nurture to medium",
        ("Low CLTV", "Low Fatigue"): "Deal-driven web/email only; no catalog",
        ("Low CLTV", "Moderate Fatigue"): "Quarterly digital deal only",
        ("Low CLTV", "High Fatigue"): "Suppress - contacting costs more than it returns",
    }
    return table.get((segment, fatigue), "-")


def build_all(margin: float = 0.25, discount: float = 0.10, path: str = "data/marketing_campaign.csv"):
    df = load_data(path)
    df = add_rfm_cltv(df, margin, discount)
    df, diag = segment_kmeans(df)
    df = add_fatigue(df)
    df["Action"] = [recommend(s, f) for s, f in zip(df["Segment"].astype(str), df["Fatigue"].astype(str))]
    return df, diag
