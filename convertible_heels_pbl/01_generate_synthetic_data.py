"""
STEP 1 - Generate a realistic synthetic consumer-survey dataset (India).

Why synthetic?  The real survey (third-party panel / Google Form) will take
~1 month. Building the whole analytics pipeline on a realistic mock means the
model is ready the day real responses arrive: replace the CSV and re-run.

How it is made realistic:
  * Respondents are drawn from 5 hidden consumer personas (the "truth" the
    clustering should rediscover). The persona column is NOT saved.
  * Answers are linked the way they are in real life (e.g. more heel hours ->
    more pain -> more interest in the orthopaedic insole; drivers -> click-lock).
  * Purchase intent and willingness-to-pay come from a scoring formula plus
    random noise, so no algorithm can be 100 % accurate (like real data).
  * Intentional data-quality problems are injected (missing values, outliers,
    typos, inconsistent labels, invalid scale values, duplicate submissions)
    so the cleaning step has real work to do.

Output: data/raw_survey_responses.csv
Run:    python 01_generate_synthetic_data.py
"""
import os
import numpy as np
import pandas as pd
from survey_design import (CITIES_TIER1, CITIES_TIER2, OCCUPATIONS, COMMUTE, HEEL_FREQ,
                           CHANNELS, HEEL_TYPES, PAINS, SITUATIONS, CONCERNS, FEATURES)

SEED = 42
N = 2000
rng = np.random.default_rng(SEED)
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# ---------------------------------------------------------------- personas
PERSONAS = {
    #  name               share  age(mu,sd,lo,hi)   income median  heel_freq probs (0..5)
    "Corporate commuter": dict(p=0.28, age=(31, 5, 23, 52), inc=75000,
                               hf=[.01, .04, .10, .30, .33, .22]),
    "Occasion glam":      dict(p=0.24, age=(29, 6, 21, 46), inc=55000,
                               hf=[.01, .10, .45, .30, .10, .04]),
    "Comfort seeker":     dict(p=0.16, age=(41, 8, 28, 62), inc=65000,
                               hf=[.08, .40, .30, .15, .05, .02]),
    "Budget student":     dict(p=0.20, age=(21, 2, 18, 27), inc=11000,
                               hf=[.05, .30, .35, .20, .07, .03]),
    "Traditional / low-heel": dict(p=0.12, age=(43, 9, 25, 65), inc=26000,
                                   hf=[.35, .45, .12, .06, .02, .00]),
}
pnames = list(PERSONAS)
persona = rng.choice(pnames, size=N, p=[PERSONAS[k]["p"] for k in pnames])

OCC_BY_PERSONA = {
    "Corporate commuter": ([0, 1, 4, 6, 7], [.62, .10, .10, .10, .08]),
    "Occasion glam": ([0, 1, 3, 6, 5, 4], [.30, .18, .20, .17, .08, .07]),
    "Comfort seeker": ([4, 5, 0, 7, 1], [.30, .25, .20, .15, .10]),
    "Budget student": ([2, 6], [.90, .10]),
    "Traditional / low-heel": ([3, 5, 7, 1], [.55, .20, .15, .10]),
}
COMMUTE_BY_PERSONA = {
    "Corporate commuter": [.42, .10, .23, .20, .05],
    "Occasion glam": [.30, .15, .30, .18, .07],
    "Comfort seeker": [.30, .10, .25, .20, .15],
    "Budget student": [.05, .30, .25, .30, .10],
    "Traditional / low-heel": [.15, .25, .30, .15, .15],
}


def pick(options, probs):
    return options[rng.choice(len(options), p=np.array(probs) / np.sum(probs))]


rows = []
for i in range(N):
    P = persona[i]
    cfg = PERSONAS[P]
    mu, sd, lo, hi = cfg["age"]
    age = int(np.clip(round(rng.normal(mu, sd)), lo, hi))

    tier1 = rng.random() < (0.85 if P in ("Corporate commuter", "Occasion glam") else 0.65)
    city = rng.choice(CITIES_TIER1 if tier1 else CITIES_TIER2)

    occ_idx, occ_p = OCC_BY_PERSONA[P]
    occupation = OCCUPATIONS[pick(occ_idx, occ_p)]

    income = cfg["inc"] * rng.lognormal(0, 0.45) * (1.15 if tier1 else 0.85)
    if occupation == "Homemaker":
        income *= 0.45
    income = round(income, -2)

    marital = "Married" if rng.random() < np.clip((age - 22) / 18, 0.05, 0.9) else "Single"
    commute = pick(COMMUTE, COMMUTE_BY_PERSONA[P])

    hours_on_feet = float(np.clip(rng.normal(
        {"Corporate commuter": 5, "Occasion glam": 4.5, "Comfort seeker": 6.5,
         "Budget student": 5.5, "Traditional / low-heel": 6}[P], 1.8), 0.5, 14))

    hf = rng.choice(6, p=cfg["hf"])
    heel_hours = 0.0 if hf == 0 else float(np.clip(rng.normal(2 + hf * 1.1, 1.5), 0.5, 13))
    pairs = int(rng.poisson(0.5 + hf * 1.3 + (2 if P == "Occasion glam" else 0)))

    spend = (0.012 + 0.0045 * hf) * income * 12 * rng.lognormal(0, 0.35) + 2500
    spend = round(min(spend, 150000), -2)
    max_paid = round(max(399, spend * rng.uniform(0.18, 0.45)), -1)

    channel = pick(CHANNELS, {
        "Corporate commuter": [.30, .15, .20, .15, .17, .03],
        "Occasion glam": [.25, .22, .15, .18, .15, .05],
        "Comfort seeker": [.20, .10, .25, .10, .25, .10],
        "Budget student": [.30, .10, .35, .12, .03, .10],
        "Traditional / low-heel": [.10, .03, .30, .05, .17, .35]}[P])

    # heel type preferences
    heel_p = {"Kitten": .35 + .10 * (P == "Corporate commuter"),
              "Stiletto": .15 + .25 * (P == "Occasion glam") + .05 * hf,
              "Block": .45 + .10 * (P in ("Comfort seeker", "Corporate commuter")),
              "Wedge": .25 + .10 * (P == "Comfort seeker"),
              "Platform": .12 + .15 * (P == "Budget student"),
              "Cone": .10}
    heel_types = [h for h in HEEL_TYPES if hf > 0 and rng.random() < heel_p[h]]
    if hf > 0 and not heel_types:
        heel_types = [rng.choice(["Block", "Kitten"])]

    # pains - driven by frequency, hours, age
    risk = 0.05 + 0.06 * hf + 0.025 * heel_hours + 0.004 * max(age - 30, 0)
    pain_base = {"Knee pain": 1.0, "Calf pain": 1.1, "Heel pain / Plantar fasciitis": 0.9,
                 "Lower back pain": 0.8, "Ankle strain": 0.6, "Blisters / Shoe bite": 1.3}
    pains = [p for p in PAINS if rng.random() < np.clip(risk * pain_base[p], 0, 0.9)]
    if "Stiletto" in heel_types and rng.random() < 0.25 and "Ankle strain" not in pains:
        pains.append("Ankle strain")
    pain_sev = int(np.clip(round(1 + 1.3 * len(pains) + 0.25 * heel_hours + rng.normal(0, 1.2)), 1, 10))

    carry = int(np.clip(round(1 + 0.45 * hf + 0.35 * len(pains)
                              + (0.8 if commute == "Self-drive car" else 0)
                              + (0.5 if P == "Corporate commuter" else 0)
                              + rng.normal(0, 0.7)), 1, 5))

    sit_p = {"Driving": .75 if commute == "Self-drive car" else .08,
             "Office to party": .55 if P == "Corporate commuter" else (.30 if P == "Occasion glam" else .08),
             "Weddings / Functions": .65 if P == "Occasion glam" else .30,
             "Travel": .30, "Long standing events": .20 + .04 * len(pains),
             "Walking commute": .45 if commute in ("Metro / Public transport", "Walk") else .10}
    situations = [s for s in SITUATIONS if hf > 0 and rng.random() < sit_p[s]]

    imp_comfort = int(np.clip(round(3 + 0.35 * len(pains) + (1 if P == "Comfort seeker" else 0) + rng.normal(0, .8)), 1, 5))
    imp_style = int(np.clip(round(3 + (1.2 if P in ("Occasion glam", "Budget student") else 0)
                                  - (0.8 if P == "Traditional / low-heel" else 0) + rng.normal(0, .8)), 1, 5))
    imp_dur = int(np.clip(round(3.4 + (0.6 if P in ("Comfort seeker", "Traditional / low-heel") else 0) + rng.normal(0, .8)), 1, 5))

    aware = "Yes" if rng.random() < (0.22 if P in ("Occasion glam", "Budget student") else 0.10) else "No"
    trust = int(np.clip(round(3 + (0.6 if P in ("Budget student", "Occasion glam") else 0)
                              - (0.7 if P == "Traditional / low-heel" else 0) + rng.normal(0, .9)), 1, 5))
    influence = int(np.clip(round(2.6 + (1.3 if P == "Budget student" else 0) + (0.9 if P == "Occasion glam" else 0)
                                  - 0.02 * max(age - 30, 0) * 3 + rng.normal(0, .9)), 1, 5))

    # features of interest (association rules are planted here on purpose)
    f = set()
    if "Kitten" in heel_types and rng.random() < .70: f.add("Kitten heel module")
    if "Stiletto" in heel_types and rng.random() < .75: f.add("Stiletto heel module")
    if "Block" in heel_types and rng.random() < .75: f.add("Block heel module")
    if "Wedge" in heel_types and rng.random() < .55: f.add("Wedge heel module")
    if rng.random() < .12: f.add(rng.choice(FEATURES[:4]))
    if len(pains) >= 2 or "Heel pain / Plantar fasciitis" in pains or "Knee pain" in pains:
        if rng.random() < .70: f.add("Orthopaedic cushioned insole")
    elif rng.random() < .15: f.add("Orthopaedic cushioned insole")
    if "Driving" in situations and rng.random() < .65: f.add("Click-lock safety mechanism")
    if "Driving" in situations and rng.random() < .45: f.add("Kitten heel module")
    if rng.random() < .25 + .10 * imp_dur / 5: f.add("Click-lock safety mechanism")
    n_heel_modules = sum(x in f for x in FEATURES[:4])
    if n_heel_modules >= 2 and rng.random() < .65: f.add("Travel pouch for heels")
    elif rng.random() < .08: f.add("Travel pouch for heels")
    if "Office to party" in situations and rng.random() < .55:
        f.add("Interchangeable straps"); f.add("Stiletto heel module")
    elif rng.random() < .15: f.add("Interchangeable straps")
    if imp_style >= 4 and rng.random() < .45: f.add("Custom colours / prints")
    if "Custom colours / prints" in f and "Interchangeable straps" not in f and rng.random() < .3:
        f.add("Interchangeable straps")
    if pairs >= 6 and rng.random() < .25: f.add("Heel replacement subscription")
    elif rng.random() < .04: f.add("Heel replacement subscription")
    features = [x for x in FEATURES if x in f]

    concern_p = {"Heel detaching / Safety": .55, "Price": .30 + (.35 if income < 30000 else 0),
                 "Limited designs": .20 + .15 * (imp_style >= 4), "Durability": .30 + .10 * (imp_dur >= 4),
                 "Sizing / Fit": .20, "Mechanism too complicated": .15 + .15 * (age > 45)}
    concerns = [c for c in CONCERNS if rng.random() < concern_p[c]]

    # ----------------------------------------------------- willingness to pay
    wtp = (950 + 0.020 * min(income, 250000) + 180 * hf + 220 * (carry - 3)
           + 110 * len(pains) + 260 * (imp_style - 3) + 90 * len(features)
           + {"Corporate commuter": 500, "Occasion glam": 300, "Comfort seeker": 250,
              "Budget student": -250, "Traditional / low-heel": -300}[P]
           - (250 if "Price" in concerns else 0)
           + rng.normal(0, 420))
    wtp = int(round(max(499, wtp), -1))
    wtp_extra = int(round(max(149, wtp * rng.uniform(0.16, 0.30) + rng.normal(0, 60)), -1))

    # ------------------------------------------------------ purchase intent
    z = (-1.55 + 0.38 * (hf - 2) + 0.55 * (carry - 3) + 0.22 * min(len(pains), 4)
         + 0.55 * ("Driving" in situations) + 0.55 * ("Office to party" in situations)
         + 0.30 * (trust - 3) + 0.22 * (influence - 3) + 0.16 * len(features)
         + 0.35 * (aware == "Yes") - 0.35 * ("Heel detaching / Safety" in concerns)
         - 0.30 * ("Mechanism too complicated" in concerns)
         + 0.000004 * min(income, 250000)
         + rng.logistic(0, 0.55))
    intent = "Yes" if z > 0.85 else ("Maybe" if z > -0.85 else "No")

    # ------------------------------------------------------------ open text
    if intent == "Yes":
        txt = rng.choice(["Would love not carrying flats in my bag anymore",
                          "If the heel lock is really safe I will buy two pairs",
                          "Perfect for office to party, please launch soon",
                          "Need it for driving, heels are a nightmare in the car",
                          "My knees hurt by evening, this solves it"])
    elif intent == "Maybe":
        txt = rng.choice(["Depends on price and how sturdy the heel is",
                          "Want to try it in a store first",
                          "Show me reviews and a return policy",
                          "Only if designs look as good as normal heels",
                          "Not sure the mechanism will last"])
    else:
        txt = rng.choice(["I rarely wear heels", "Too expensive for me",
                          "Sounds complicated", "I prefer my regular flats",
                          "Worried the heel will come off"])

    rows.append({
        "respondent_id": f"R{i + 1:05d}",
        "timestamp": (pd.Timestamp("2026-10-01") + pd.Timedelta(minutes=int(rng.integers(0, 60 * 24 * 30)))).strftime("%Y-%m-%d %H:%M"),
        "age": age, "city": city, "occupation": occupation, "monthly_income_inr": income,
        "marital_status": marital, "commute_mode": commute, "hours_on_feet": round(hours_on_feet, 1),
        "heel_frequency": HEEL_FREQ[hf], "heel_hours_per_wear": round(heel_hours, 1),
        "heel_pairs_owned": pairs, "annual_footwear_spend_inr": spend, "max_price_paid_inr": max_paid,
        "preferred_channel": channel,
        "heel_types_preferred": "; ".join(heel_types),
        "pains_experienced": "; ".join(pains) if pains else "None",
        "pain_severity": pain_sev, "carry_extra_pair": carry,
        "switch_situations": "; ".join(situations) if situations else "None",
        "importance_comfort": imp_comfort, "importance_style": imp_style, "importance_durability": imp_dur,
        "aware_of_convertible_heels": aware,
        "features_interested": "; ".join(features) if features else "None",
        "concerns": "; ".join(concerns) if concerns else "None",
        "trust_d2c_brands": trust, "influencer_influence": influence,
        "wtp_base_inr": wtp, "wtp_extra_heel_inr": wtp_extra,
        "purchase_intent": intent, "open_feedback": txt,
    })

df = pd.DataFrame(rows)

# ===================================================== inject data-quality issues
def idx(frac=None, n=None):
    n = n if n is not None else int(len(df) * frac)
    return rng.choice(df.index, size=n, replace=False)

# 1) Missing values (people skip questions)
for col, frac in [("age", .02), ("monthly_income_inr", .045), ("wtp_base_inr", .03),
                  ("pain_severity", .02), ("city", .01), ("preferred_channel", .02),
                  ("hours_on_feet", .015), ("wtp_extra_heel_inr", .025)]:
    df.loc[idx(frac), col] = np.nan

# 2) Outliers / impossible values (typos, joke answers)
df.loc[idx(n=8), "monthly_income_inr"] = df["monthly_income_inr"].median() * rng.choice([25, 40, 60], 8)
df.loc[idx(n=5), "age"] = rng.choice([4, 9, 150, 120, 99], 5)
df.loc[idx(n=10), "wtp_base_inr"] = rng.choice([99999, 50000, 75000, 25, 10], 10)
df.loc[idx(n=6), "hours_on_feet"] = rng.choice([30, 24, 48], 6)
df.loc[idx(n=5), "annual_footwear_spend_inr"] = -df["annual_footwear_spend_inr"].sample(5, random_state=1).values
df.loc[idx(n=4), "heel_pairs_owned"] = rng.choice([150, 300], 4)

# 3) Invalid Likert values (outside 1-5)
for col in ["carry_extra_pair", "importance_comfort", "trust_d2c_brands", "influencer_influence"]:
    df.loc[idx(.006), col] = rng.choice([0, 6, 7], int(len(df) * .006))

# 4) Inconsistent text labels
city_variants = {"Mumbai": ["mumbai", "Bombay", " Mumbai "], "Bengaluru": ["Bangalore", "bengaluru"],
                 "Delhi NCR": ["Delhi", "New Delhi", "Gurgaon"], "Chennai": ["Madras"],
                 "Kolkata": ["Calcutta"], "Pune": ["pune "]}
for canon, variants in city_variants.items():
    m = df.index[df["city"] == canon]
    chosen = rng.choice(m, size=int(len(m) * .12), replace=False)
    df.loc[chosen, "city"] = rng.choice(variants, len(chosen))
intent_variants = {"Yes": ["yes", "Y", "YES ", "Definitely"], "No": ["no", "N", "Nope"],
                   "Maybe": ["maybe", "May be", "Not sure"]}
for canon, variants in intent_variants.items():
    m = df.index[df["purchase_intent"] == canon]
    chosen = rng.choice(m, size=int(len(m) * .06), replace=False)
    df.loc[chosen, "purchase_intent"] = rng.choice(variants, len(chosen))
m = idx(.03)
df.loc[m, "occupation"] = df.loc[m, "occupation"].str.lower().str.strip()

# 5) Duplicate submissions (same person pressed submit twice)
dups = df.loc[idx(n=35)].copy()
df = pd.concat([df, dups]).sample(frac=1, random_state=SEED).reset_index(drop=True)

os.makedirs(OUT_DIR, exist_ok=True)
out = os.path.join(OUT_DIR, "raw_survey_responses.csv")
df.to_csv(out, index=False)
print(f"Saved {len(df):,} rows x {df.shape[1]} columns -> {out}")
print(df["purchase_intent"].value_counts().head(10))
