"""
Data cleaning & transformation pipeline.

Used by 02_clean_data.py (batch) AND by the dashboard (so a real survey CSV
uploaded later goes through exactly the same steps).

Every step appends a row to a cleaning log: what was done, why, and how many
records were affected. The log is shown in the dashboard and saved to CSV.
"""
import numpy as np
import pandas as pd
from survey_design import (CITIES, CITIES_TIER1, OCCUPATIONS, HEEL_FREQ, MULTI_SELECT)

CITY_MAP = {"bombay": "Mumbai", "mumbai": "Mumbai", "bangalore": "Bengaluru", "bengaluru": "Bengaluru",
            "delhi": "Delhi NCR", "new delhi": "Delhi NCR", "gurgaon": "Delhi NCR", "gurugram": "Delhi NCR",
            "noida": "Delhi NCR", "delhi ncr": "Delhi NCR", "madras": "Chennai", "chennai": "Chennai",
            "calcutta": "Kolkata", "kolkata": "Kolkata", "pune": "Pune", "hyderabad": "Hyderabad",
            "ahmedabad": "Ahmedabad", "jaipur": "Jaipur", "lucknow": "Lucknow", "chandigarh": "Chandigarh",
            "indore": "Indore"}
INTENT_MAP = {"yes": "Yes", "y": "Yes", "definitely": "Yes", "no": "No", "n": "No", "nope": "No",
              "maybe": "Maybe", "may be": "Maybe", "not sure": "Maybe"}
LIKERT_COLS = ["carry_extra_pair", "importance_comfort", "importance_style", "importance_durability",
               "trust_d2c_brands", "influencer_influence"]
NUMERIC_COLS = ["age", "monthly_income_inr", "hours_on_feet", "heel_hours_per_wear", "heel_pairs_owned",
                "annual_footwear_spend_inr", "max_price_paid_inr", "pain_severity", "wtp_base_inr",
                "wtp_extra_heel_inr"] + LIKERT_COLS


def slug(s):
    return (s.lower().replace(" / ", "_").replace("/", "_").replace(" ", "_")
            .replace("-", "_").replace("(", "").replace(")", "").replace("'", ""))


def clean(raw: pd.DataFrame):
    df = raw.copy()
    log = []

    def add(step, action, why, n):
        log.append({"step": step, "action": action, "reason": why, "records_affected": int(n)})

    add("0. Start", f"Loaded {len(df):,} rows x {df.shape[1]} columns", "Raw survey export", len(df))

    # 1. duplicates --------------------------------------------------------
    before = len(df)
    df = df.drop_duplicates(subset=["respondent_id"], keep="first") if "respondent_id" in df else df.drop_duplicates()
    add("1. Duplicates", "Removed duplicate submissions (same respondent_id)",
        "A person who pressed 'Submit' twice would be counted twice and bias every %", before - len(df))

    # 2. text standardisation ---------------------------------------------
    for c in df.select_dtypes(include=["object", "string"]).columns:
        df[c] = df[c].astype("string").str.strip()
    raw_city = df["city"].copy()
    df["city"] = df["city"].str.lower().map(CITY_MAP).fillna(df["city"])
    add("2a. City names", "Mapped spelling variants (Bombay, Bangalore, Gurgaon, Madras...) to one standard name",
        "Same city written 3 ways would look like 3 different markets", (raw_city.fillna("") != df["city"].fillna("")).sum())

    raw_occ = df["occupation"].copy()
    occ_lookup = {o.lower(): o for o in OCCUPATIONS}
    df["occupation"] = df["occupation"].str.lower().map(occ_lookup).fillna(df["occupation"])
    add("2b. Occupation", "Fixed lower-case / extra-space occupation labels",
        "Category counts must not split because of letter case", (raw_occ != df["occupation"]).sum())

    raw_int = df["purchase_intent"].copy()
    df["purchase_intent"] = df["purchase_intent"].str.lower().map(INTENT_MAP)
    add("2c. Purchase intent (target)", "Mapped 'Y', 'yes', 'Definitely', 'Nope', 'Not sure'... to Yes / No / Maybe",
        "The classification label must have exactly 3 clean classes", (raw_int != df["purchase_intent"]).sum())
    n_bad = df["purchase_intent"].isna().sum()
    df = df[df["purchase_intent"].notna()]
    add("2d. Purchase intent (target)", "Dropped rows with no usable intent answer",
        "Cannot train or test a classifier on a row with no label", n_bad)

    # 3. types --------------------------------------------------------------
    for c in NUMERIC_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # 4. impossible values -> missing --------------------------------------
    rules = [
        ("age", (df["age"] < 16) | (df["age"] > 75), "Survey is for women 16-75; ages like 4 or 150 are typos"),
        ("hours_on_feet", df["hours_on_feet"] > 18, "Nobody stands more than 18 h a day; 24/30/48 are typos"),
        ("annual_footwear_spend_inr", df["annual_footwear_spend_inr"] < 0, "Spending cannot be negative"),
        ("heel_pairs_owned", df["heel_pairs_owned"] > 60, "150-300 pairs of heels is not credible for this survey"),
        ("wtp_base_inr", (df["wtp_base_inr"] < 300) | (df["wtp_base_inr"] > 25000),
         "Below ₹300 or above ₹25,000 for a sandal is a joke/typo answer"),
    ]
    for col, mask, why in rules:
        n = int(mask.fillna(False).sum())
        df.loc[mask.fillna(False), col] = np.nan
        add("4. Impossible values", f"{col}: set {n} impossible values to missing", why, n)
    for col in LIKERT_COLS:
        mask = ~df[col].between(1, 5) & df[col].notna()
        df.loc[mask, col] = np.nan
        if mask.sum():
            add("4. Impossible values", f"{col}: values outside the 1-5 scale set to missing",
                "A 1-5 Likert scale cannot contain 0, 6 or 7", mask.sum())

    # 5. extreme-but-possible outliers -> cap (winsorise) ----------------
    for col in ["monthly_income_inr", "annual_footwear_spend_inr", "max_price_paid_inr"]:
        q1, q3 = df[col].quantile([.25, .75])
        upper = q3 + 3 * (q3 - q1)
        mask = df[col] > upper
        df.loc[mask, col] = upper
        add("5. Outliers", f"{col}: capped {mask.sum()} values at Q3 + 3xIQR (₹{upper:,.0f})",
            "Very high values are possible but would distort means and regression; capping keeps the row", mask.sum())

    # 6. missing values -----------------------------------------------------
    miss = df[NUMERIC_COLS].isna().sum()
    df["monthly_income_inr"] = df.groupby("occupation")["monthly_income_inr"].transform(lambda s: s.fillna(s.median()))
    add("6. Missing values", "monthly_income_inr: filled with the median income of the same occupation",
        "A student's missing income should not be filled with a professional's income", miss["monthly_income_inr"])
    for col in [c for c in NUMERIC_COLS if c != "monthly_income_inr"]:
        if miss[col] > 0:
            df[col] = df[col].fillna(df[col].median())
            add("6. Missing values", f"{col}: filled {miss[col]} blanks with the median",
                "Median is robust to the skew/outliers seen in survey answers", miss[col])
    for col in ["city", "preferred_channel"]:
        n = df[col].isna().sum()
        df[col] = df[col].fillna(df[col].mode()[0])
        add("6. Missing values", f"{col}: filled {n} blanks with the most common answer (mode)",
            "Categorical column; mode is the standard fill and affects < 2% of rows", n)
    df["age"] = df["age"].round().astype(int)

    # 7. transformation / feature engineering ----------------------------
    for col, options in MULTI_SELECT.items():
        s = df[col].fillna("None")
        prefix = {"heel_types_preferred": "heel", "pains_experienced": "pain", "switch_situations": "sit",
                  "features_interested": "feat", "concerns": "concern"}[col]
        for o in options:
            df[f"{prefix}_{slug(o)}"] = s.str.split("; ").apply(lambda L: int(o in L))
        df[f"{prefix}_count"] = df[[f"{prefix}_{slug(o)}" for o in options]].sum(axis=1)
    add("7a. One-hot encoding", "Split the 5 multi-select questions ('A; B; C') into 0/1 columns + a count column",
        "Algorithms need one column per option; association rules need item baskets", len(df))

    df["heel_freq_score"] = df["heel_frequency"].map({h: i for i, h in enumerate(HEEL_FREQ)}).fillna(0).astype(int)
    add("7b. Ordinal encoding", "heel_frequency (Never ... Daily) -> score 0-5",
        "Keeps the natural order so models understand 'Daily' > 'Weekly'", len(df))

    df["city_tier"] = np.where(df["city"].isin(CITIES_TIER1), "Tier 1", "Tier 2")
    df["age_group"] = pd.cut(df["age"], [15, 24, 34, 44, 54, 80], labels=["16-24", "25-34", "35-44", "45-54", "55+"]).astype(str)
    df["income_band"] = pd.cut(df["monthly_income_inr"], [-1, 25000, 50000, 100000, 200000, 1e12],
                               labels=["<25k", "25-50k", "50k-1L", "1-2L", "2L+"]).astype(str)
    df["drives_self"] = (df["commute_mode"] == "Self-drive car").astype(int)
    df["aware_flag"] = (df["aware_of_convertible_heels"] == "Yes").astype(int)
    df["intent_yes"] = (df["purchase_intent"] == "Yes").astype(int)
    df["intent_score"] = df["purchase_intent"].map({"No": 0, "Maybe": 1, "Yes": 2})
    df["log_income"] = np.log1p(df["monthly_income_inr"])
    df["footwear_share_of_income"] = (df["annual_footwear_spend_inr"] / (df["monthly_income_inr"] * 12)).round(4)
    add("7c. New features", "Created city_tier, age_group, income_band, drives_self, aware_flag, intent_yes, "
        "intent_score, log_income, footwear_share_of_income",
        "Bands are for readable charts; log/ratio features help models; numeric age/income are kept for algorithms", len(df))

    df = df.reset_index(drop=True)
    add("8. Final", f"Clean dataset: {len(df):,} rows x {df.shape[1]} columns, 0 missing values in model columns",
        "Ready for EDA and the 4 algorithms", len(df))
    return df, pd.DataFrame(log)
