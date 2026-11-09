"""
Analytics engine: the 4 algorithm families + pricing and break-even.
Kept separate from the dashboard so each piece can be read, tested and
re-used on the real survey data.
"""
from itertools import combinations
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
                             roc_auc_score, silhouette_score, r2_score, mean_absolute_error,
                             mean_squared_error)
from survey_design import FEATURES, PAINS, SITUATIONS, CONCERNS, OCCUPATIONS, COMMUTE
from pipeline import slug

RS = 42
CLASSES = ["No", "Maybe", "Yes"]

# ============================================================ feature sets
BASE_NUM = ["age", "log_income", "hours_on_feet", "heel_freq_score", "heel_hours_per_wear",
            "heel_pairs_owned", "annual_footwear_spend_inr", "pain_severity", "carry_extra_pair",
            "importance_comfort", "importance_style", "importance_durability", "trust_d2c_brands",
            "influencer_influence", "aware_flag", "drives_self", "feat_count", "pain_count", "sit_count"]
PAIN_COLS = [f"pain_{slug(p)}" for p in PAINS]
SIT_COLS = [f"sit_{slug(s)}" for s in SITUATIONS]
CONCERN_COLS = [f"concern_{slug(c)}" for c in CONCERNS]
FEAT_COLS = [f"feat_{slug(f)}" for f in FEATURES]


def model_matrix(df, extra=None):
    X = df[BASE_NUM + PAIN_COLS + SIT_COLS + CONCERN_COLS + (extra or [])].copy()
    X["tier1_city"] = (df["city_tier"] == "Tier 1").astype(int)
    occ = pd.get_dummies(df["occupation"], prefix="occ", dtype=int).reindex(
        columns=[f"occ_{o}" for o in OCCUPATIONS], fill_value=0)
    com = pd.get_dummies(df["commute_mode"], prefix="commute", dtype=int).reindex(
        columns=[f"commute_{c}" for c in COMMUTE], fill_value=0)
    return pd.concat([X, occ, com], axis=1).astype(float)


def pretty(col):
    for p in ("pain_", "sit_", "concern_", "feat_", "occ_", "commute_"):
        if col.startswith(p) and not col.endswith("_count"):
            return f"{p[:-1].capitalize()}: {col[len(p):].replace('_', ' ')}"
    return col.replace("_", " ")


# ======================================================== CLASSIFICATION
def classification_models():
    return {
        "K-Nearest Neighbours": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=15, weights="distance")),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, min_samples_leaf=20, random_state=RS),
        "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=3, random_state=RS, n_jobs=-1),
        "Gradient Boosted Trees": GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=RS),
    }


def run_classification(df, test_size=0.25):
    X = model_matrix(df)
    y = df["purchase_intent"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, stratify=y, random_state=RS)
    cv = StratifiedKFold(5, shuffle=True, random_state=RS)
    rows, cms, fitted, reports = [], {}, {}, {}
    for name, m in classification_models().items():
        m.fit(Xtr, ytr)
        pred = m.predict(Xte)
        proba = m.predict_proba(Xte)
        cv_acc = cross_val_score(m, X, y, cv=cv, scoring="accuracy")
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(yte, pred),
            "Precision (macro)": precision_score(yte, pred, average="macro"),
            "Recall (macro)": recall_score(yte, pred, average="macro"),
            "F1 (macro)": f1_score(yte, pred, average="macro"),
            "ROC-AUC (one-vs-rest)": roc_auc_score(yte, proba, multi_class="ovr", labels=list(m.classes_)),
            "5-fold CV accuracy": cv_acc.mean(),
            "CV std": cv_acc.std(),
            "Train accuracy": accuracy_score(ytr, m.predict(Xtr)),
        })
        cms[name] = confusion_matrix(yte, pred, labels=CLASSES)
        reports[name] = pd.DataFrame({
            "Precision": precision_score(yte, pred, labels=CLASSES, average=None),
            "Recall": recall_score(yte, pred, labels=CLASSES, average=None),
            "F1": f1_score(yte, pred, labels=CLASSES, average=None),
            "Support": [int((yte == c).sum()) for c in CLASSES]}, index=CLASSES)
        fitted[name] = m
    res = pd.DataFrame(rows).sort_values("F1 (macro)", ascending=False).reset_index(drop=True)
    best = res.iloc[0]["Model"]
    imp_model = fitted[best] if best in ("Random Forest", "Gradient Boosted Trees", "Decision Tree") else fitted["Random Forest"]
    imp = pd.Series(imp_model.feature_importances_, index=X.columns).sort_values(ascending=False)
    return dict(results=res, cms=cms, reports=reports, models=fitted, best=best,
                importance=imp, columns=list(X.columns), n_train=len(Xtr), n_test=len(Xte))


# ============================================================ CLUSTERING
CLUSTER_VARS = ["age", "log_income", "heel_freq_score", "heel_hours_per_wear", "carry_extra_pair",
                "pain_count", "pain_severity", "importance_comfort", "importance_style",
                "influencer_influence", "drives_self", "sit_count", "hours_on_feet"]


def cluster_scan(df, kmin=2, kmax=8):
    Z = StandardScaler().fit_transform(df[CLUSTER_VARS])
    out = []
    for k in range(kmin, kmax + 1):
        km = KMeans(k, n_init=10, random_state=RS).fit(Z)
        out.append({"k": k, "Inertia (WCSS)": km.inertia_,
                    "Silhouette": silhouette_score(Z, km.labels_, sample_size=min(1500, len(Z)), random_state=RS)})
    return pd.DataFrame(out)


def run_kmeans(df, k):
    scaler = StandardScaler().fit(df[CLUSTER_VARS])
    Z = scaler.transform(df[CLUSTER_VARS])
    km = KMeans(k, n_init=20, random_state=RS).fit(Z)
    labels = km.labels_
    pcs = PCA(2, random_state=RS).fit_transform(Z)
    d = df.assign(cluster=labels)
    prof = d.groupby("cluster").agg(
        Size=("respondent_id", "count"), Avg_age=("age", "mean"),
        Median_income=("monthly_income_inr", "median"), Heel_freq_0_5=("heel_freq_score", "mean"),
        Heel_hours=("heel_hours_per_wear", "mean"), Carry_extra_1_5=("carry_extra_pair", "mean"),
        Pains=("pain_count", "mean"), Pain_severity=("pain_severity", "mean"),
        Style_imp=("importance_style", "mean"), Influencer=("influencer_influence", "mean"),
        Drives_pct=("drives_self", "mean"), Yes_pct=("intent_yes", "mean"),
        Maybe_pct=("purchase_intent", lambda s: (s == "Maybe").mean()),
        Median_WTP=("wtp_base_inr", "median"))
    prof["Share_pct"] = prof["Size"] / prof["Size"].sum()
    top_occ = d.groupby("cluster")["occupation"].agg(lambda s: s.value_counts().index[0])
    top_city = d.groupby("cluster")["city_tier"].agg(lambda s: s.value_counts().index[0])
    # conversion tier by Yes rate
    order = prof["Yes_pct"].rank(ascending=False, method="first")
    n = len(prof)
    tier = order.apply(lambda r: "High" if r <= max(1, round(n / 3)) else ("Low" if r > n - max(1, round(n / 3)) else "Medium"))
    prof["Conversion_tier"] = tier
    prof["Name"] = [persona_name(prof.loc[c], top_occ[c]) for c in prof.index]
    prof["Top_occupation"] = top_occ
    prof["Main_city_tier"] = top_city
    return dict(labels=labels, profile=prof, pcs=pcs, model=km, scaler=scaler)


def persona_name(r, occ):
    if r["Heel_freq_0_5"] >= 3.2 and r["Drives_pct"] >= 0.35:
        base = "Driving daily heel-wearers"
    elif r["Heel_freq_0_5"] >= 3.2:
        base = "Frequent heel-wearers"
    elif r["Avg_age"] < 25:
        base = "Young trend-seekers"
    elif r["Heel_freq_0_5"] < 1.5:
        base = "Low-heel / flats loyalists"
    elif r["Pain_severity"] >= 5.5 or r["Avg_age"] >= 38:
        base = "Comfort-first occasional wearers"
    else:
        base = "Occasion & party wearers"
    return f"{base} ({occ.split(' /')[0]})"


# --------- Latent Class Analysis (EM algorithm on categorical answers) ----------
def lca_inputs(df):
    cat = pd.DataFrame({
        "Heel frequency": pd.cut(df["heel_freq_score"], [-1, 1, 2, 3, 5], labels=["Rare", "Monthly", "Weekly", "Several/Daily"]),
        "Carries extra pair": pd.cut(df["carry_extra_pair"], [0, 2, 3, 5], labels=["Rarely", "Sometimes", "Often"]),
        "Pains": pd.cut(df["pain_count"], [-1, 0, 2, 6], labels=["None", "1-2", "3+"]),
        "Income band": df["income_band"],
        "Commute": df["commute_mode"],
        "Style importance": pd.cut(df["importance_style"], [0, 2, 3, 5], labels=["Low", "Medium", "High"]),
        "Occupation": df["occupation"],
    }).astype(str)
    return cat


def run_lca(df, n_classes=4, n_starts=6, iters=300, seed=RS):
    cat = lca_inputs(df)
    rng = np.random.default_rng(seed)
    codes, levels = [], []
    for c in cat.columns:
        lv = sorted(cat[c].unique())
        levels.append(lv)
        codes.append(cat[c].map({v: i for i, v in enumerate(lv)}).values)
    n = len(cat)
    best = None
    for _ in range(n_starts):
        pi = np.full(n_classes, 1 / n_classes)
        theta = [rng.dirichlet(np.ones(len(lv)), size=n_classes) for lv in levels]
        prev = -np.inf
        for _ in range(iters):
            logp = np.log(pi)[None, :].repeat(n, 0)
            for j, cd in enumerate(codes):
                logp += np.log(theta[j][:, cd].T + 1e-12)
            m = logp.max(1, keepdims=True)
            ll = (m.squeeze() + np.log(np.exp(logp - m).sum(1))).sum()
            post = np.exp(logp - m)
            post /= post.sum(1, keepdims=True)
            pi = post.mean(0)
            for j, cd in enumerate(codes):
                t = np.zeros((n_classes, len(levels[j])))
                for lvl in range(len(levels[j])):
                    t[:, lvl] = post[cd == lvl].sum(0)
                theta[j] = (t + 0.01) / (t + 0.01).sum(1, keepdims=True)
            if ll - prev < 1e-6:
                break
            prev = ll
        if best is None or ll > best["ll"]:
            best = dict(ll=ll, pi=pi, theta=theta, post=post)
    k_params = (n_classes - 1) + n_classes * sum(len(lv) - 1 for lv in levels)
    best["bic"] = -2 * best["ll"] + k_params * np.log(n)
    best["labels"] = best["post"].argmax(1)
    best["levels"] = levels
    best["vars"] = list(cat.columns)
    # entropy (classification certainty, 1 = perfectly separated)
    p = np.clip(best["post"], 1e-12, 1)
    best["entropy"] = 1 - (-(p * np.log(p)).sum()) / (n * np.log(n_classes))
    return best


def lca_bic_scan(df, ks=(2, 3, 4, 5, 6)):
    return pd.DataFrame([{"Classes": k, "BIC": run_lca(df, k, n_starts=3, iters=200)["bic"]} for k in ks])


# =============================================== ASSOCIATION RULE MINING
def baskets(df, include=("features",)):
    items = []
    groups = {"features": ("F", FEATURES, "feat"), "pains": ("Pain", PAINS, "pain"),
              "situations": ("Need", SITUATIONS, "sit"), "concerns": ("Worry", CONCERNS, "concern")}
    cols, names = [], []
    for g in include:
        tag, opts, pre = groups[g]
        for o in opts:
            cols.append(f"{pre}_{slug(o)}")
            names.append(o if tag == "F" else f"{tag}: {o}")
    M = df[cols].values.astype(bool)
    return pd.DataFrame(M, columns=names)


def apriori(B, min_support=0.05, max_len=3):
    n = len(B)
    M = B.values
    names = list(B.columns)
    support = {}
    current = []
    for i in range(len(names)):
        s = M[:, i].mean()
        if s >= min_support:
            support[(i,)] = s
            current.append((i,))
    k = 2
    while current and k <= max_len:
        cand = set()
        for a, b in combinations(current, 2):
            u = tuple(sorted(set(a) | set(b)))
            if len(u) == k and all(tuple(sorted(sub)) in support for sub in combinations(u, k - 1)):
                cand.add(u)
        current = []
        for c in cand:
            s = M[:, list(c)].all(1).mean()
            if s >= min_support:
                support[c] = s
                current.append(c)
        k += 1
    fi = pd.DataFrame([{"itemset": " + ".join(names[i] for i in c), "size": len(c), "support": s}
                       for c, s in support.items()]).sort_values("support", ascending=False)
    return support, names, fi


def association_rules(support, names, min_conf=0.5, consequent_filter=None):
    rows = []
    for items, s in support.items():
        if len(items) < 2:
            continue
        for cons in items:
            ante = tuple(x for x in items if x != cons)
            if consequent_filter and not consequent_filter(names[cons]):
                continue
            conf = s / support[ante]
            if conf < min_conf:
                continue
            lift = conf / support[(cons,)]
            conviction = (1 - support[(cons,)]) / (1 - conf) if conf < 1 else np.inf
            rows.append({"If customer wants (antecedent)": " + ".join(names[a] for a in ante),
                         "...they also want (consequent)": names[cons],
                         "Support": s, "Confidence": conf, "Lift": lift, "Conviction": conviction,
                         "_ante": ante, "_cons": cons})
    r = pd.DataFrame(rows)
    if len(r):
        r = r.sort_values(["Lift", "Confidence"], ascending=False).reset_index(drop=True)
    return r


def launch_sequence(df_buyers, phase1_size=4, min_conf=0.6, min_lift=1.15):
    """Phase plan: start with the most-demanded features, then pull in any
    feature that buyers of phase-1 features strongly expect (confidence >= min_conf)
    so the first product does not disappoint (Professor's 'if A and B then C')."""
    B = baskets(df_buyers, ("features",))
    sup = B.mean().sort_values(ascending=False)
    support, names, _ = apriori(B, min_support=0.05, max_len=3)
    rules = association_rules(support, names, min_conf=0.0)
    phase1 = list(sup.index[:phase1_size])
    reason = {f: f"Top-{phase1_size} demand: {sup[f]:.0%} of likely buyers want it" for f in phase1}
    changed = True
    while changed and len(rules):
        changed = False
        for _, r in rules.iterrows():
            ante = [names[a] for a in r["_ante"]]
            cons = names[r["_cons"]]
            if (set(ante) <= set(phase1) and cons not in phase1 and r["Confidence"] >= min_conf
                    and r["Lift"] >= min_lift and r["Support"] >= 0.08):
                phase1.append(cons)
                reason[cons] = (f"Bundled: {r['Confidence']:.0%} of buyers who want {' + '.join(ante)} "
                                f"also want it (lift {r['Lift']:.2f})")
                changed = True
                break
    rest = [f for f in sup.index if f not in phase1]
    phase2, phase3 = rest[:3], rest[3:]
    out = []
    for ph, lst in (("Phase 1 - Launch", phase1), ("Phase 2 - Month 6-9", phase2), ("Phase 3 - Year 2", phase3)):
        for f in lst:
            out.append({"Phase": ph, "Feature": f, "Demand among likely buyers": sup[f],
                        "Why": reason.get(f, f"Demand {sup[f]:.0%}; added once launch features are proven")})
    return pd.DataFrame(out)


# ============================================================ REGRESSION
def run_regression(df, test_size=0.25):
    X = model_matrix(df, extra=["max_price_paid_inr"] + FEAT_COLS)
    y = df["wtp_base_inr"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=RS)
    sc = StandardScaler().fit(Xtr)
    Ztr, Zte = sc.transform(Xtr), sc.transform(Xte)
    models = {"Linear Regression": LinearRegression(),
              "Ridge (L2)": RidgeCV(alphas=np.logspace(-2, 3, 30)),
              "Lasso (L1)": LassoCV(alphas=np.logspace(-1, 2.5, 30), cv=5, random_state=RS, max_iter=20000)}
    rows, preds, coefs = [], {}, {}
    cv = KFold(5, shuffle=True, random_state=RS)
    for name, m in models.items():
        m.fit(Ztr, ytr)
        p = m.predict(Zte)
        preds[name] = p
        coefs[name] = pd.Series(m.coef_, index=X.columns)
        cvr2 = cross_val_score(make_pipeline(StandardScaler(), m.__class__(**m.get_params())), X, y, cv=cv, scoring="r2")
        rows.append({"Model": name, "R² (test)": r2_score(yte, p), "MAE (₹)": mean_absolute_error(yte, p),
                     "RMSE (₹)": mean_squared_error(yte, p) ** 0.5, "5-fold CV R²": cvr2.mean(),
                     "Train R²": r2_score(ytr, m.predict(Ztr)),
                     "Alpha": getattr(m, "alpha_", 0.0),
                     "Features used": int((np.abs(m.coef_) > 1e-6).sum())})
    return dict(results=pd.DataFrame(rows), preds=preds, y_test=yte.values, coefs=pd.DataFrame(coefs),
                models=models, scaler=sc, columns=list(X.columns))


# ===================================================== PRICING & BREAK-EVEN
def demand_curve(df, maybe_weight=0.3, prices=None):
    prices = prices if prices is not None else np.arange(999, 7001, 100)
    yes = df.loc[df["purchase_intent"] == "Yes", "wtp_base_inr"].values
    maybe = df.loc[df["purchase_intent"] == "Maybe", "wtp_base_inr"].values
    n = len(df)
    rows = []
    for p in prices:
        buyers = ((yes >= p).sum() + maybe_weight * (maybe >= p).sum()) / n
        rows.append({"Price": p, "Share of market buying": buyers, "Revenue index": p * buyers})
    return pd.DataFrame(rows)


def break_even(price, unit_cost, extra_heel_price, extra_heel_cost, attach_rate, gst_rate,
               channel_pct, return_pct, shipping, cac, fixed_monthly, setup_cost,
               start_orders, growth, months=36, cap=3000):
    gross = price + attach_rate * extra_heel_price
    net_of_gst = gross / (1 + gst_rate)
    channel = channel_pct * net_of_gst
    returns = return_pct * net_of_gst
    cogs = unit_cost + attach_rate * extra_heel_cost
    contrib = net_of_gst - channel - returns - cogs - shipping - cac
    rows, cum = [], -setup_cost
    be_month = None
    for m in range(1, months + 1):
        orders = min(start_orders * (1 + growth) ** (m - 1), cap)
        profit = orders * contrib - fixed_monthly
        cum += profit
        if be_month is None and cum >= 0:
            be_month = m
        rows.append({"Month": m, "Orders": orders, "Revenue (net of GST)": orders * net_of_gst,
                     "Monthly profit": profit, "Cumulative cash": cum})
    unit = pd.Series({"Customer pays (incl. GST)": gross, "GST": gross - net_of_gst,
                      "Marketplace / payment fees": channel, "Returns & damages": returns,
                      "Product cost (COGS)": cogs, "Shipping": shipping, "Marketing (CAC)": cac,
                      "Contribution margin": contrib})
    orders_needed = fixed_monthly / contrib if contrib > 0 else np.inf
    return pd.DataFrame(rows), unit, be_month, orders_needed


def run_binary(df, model_name="Gradient Boosted Trees", test_size=0.25):
    """Yes vs Not-yes view (the professor's TP / FP / FN / TN framing).
    Positive class = 'Yes' (customer will buy)."""
    X = model_matrix(df)
    y = (df["purchase_intent"] == "Yes").astype(int)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, stratify=y, random_state=RS)
    m = classification_models()[model_name]
    m.fit(Xtr, ytr)
    proba = m.predict_proba(Xte)[:, 1]
    pred = (proba >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(yte, pred, labels=[0, 1]).ravel()
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(yte, proba)
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, accuracy=accuracy_score(yte, pred),
                precision=precision_score(yte, pred), recall=recall_score(yte, pred),
                f1=f1_score(yte, pred), auc=roc_auc_score(yte, proba), fpr=fpr, tpr=tpr, model=m)
