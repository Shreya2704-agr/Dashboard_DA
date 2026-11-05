# 🎯 Campaign Intelligence Studio

An interactive Streamlit dashboard that explains **why supermarket customers stop responding to marketing campaigns** and **whom to target next**. It combines RFM modelling, CLTV segmentation, a campaign-fatigue framework and machine learning on 6 past campaigns (2,240 customers).

> **Headline:** none of the 6 campaigns covered its costs. At a cost of 3 per contact and revenue of 11 per response, a campaign has to reach **27.3%** response to break even. The best one reached 14.9%. Re-running Campaign 6 on the **top 15% of customers ranked by the ML model** would turn a **-$3,034 loss into a +$1,250 profit**.

---

## 🚀 Run it locally

```bash
git clone <your-repo-url>
cd Dashboard_DA
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```
Open http://localhost:8501. The first load takes about 20 seconds while the models train; after that, results are cached.

**Deploy free:** push to GitHub, go to [share.streamlit.io](https://share.streamlit.io), choose the repo, and set `app.py` as the main file.

---

## 📁 Project structure

| File | Purpose |
|---|---|
| `data/marketing_campaign.csv` | Raw data: 2,240 customers, demographics, spend, channels, 6 campaign responses |
| `analysis.py` | Analytics engine: cleaning, RFM, CLTV, K-Means, fatigue, ML models, gains, association rules |
| `app.py` | Streamlit UI: 7 tabs, 36 interactive Plotly charts, sidebar filters, CSV export |
| `.streamlit/config.toml` | Dark theme |

---

## 🧭 Step-by-step: from raw data to insight

### Step 1: Data cleaning and feature engineering (`load_data`)
* Filled 24 missing **Income** values with the median for that education level.
* Removed 4 bad rows: birth years 1893/1899/1900 and one income of 666,666.
* Merged unclear marital statuses ("Alone", "Absurd", "YOLO") into **Single**, and merged "2n Cycle" into **Master**.
* Dropped `Z_CostContact` (=3) and `Z_Revenue` (=11). They are constants, used as the **cost and revenue per contact**.
* Engineered features: **Age, Tenure, Children, Couple, Monetary** (total spend across 6 categories), **Frequency** (web + catalog + store purchases), **AOV** (average order value), **Deal share**, and **Prior / Total accepts**.

### Step 2: Campaign performance (`campaign_summary`, `acceptor_profile`, `co_acceptance`)
For each campaign: response rate, responders, cost, revenue, profit and ROI against the **break-even rate = cost ÷ revenue = 27.3%**. An *acceptor profile index* compares each campaign's acceptors with the average customer (100 = average). A *co-acceptance matrix* gives P(accept B | accepted A).

### Step 3: RFM modelling and CLTV (`add_rfm_cltv`)
* **R, F and M scores (1–5)**, assigned by quintile: Recency (lower is better), Frequency and Monetary.
* **CLTV** uses the standard retention-and-discount formula, driven by RFM:
  ```
  annual value = Monetary / tenure_years            ← M and F
  retention r  = 0.50 + 0.10 × (R_score − 1)        ← R  (50% … 90%)
  CLTV         = annual value × margin × r / (1 + d − r)
  ```
  Margin (default 25%) and discount rate (default 10%) can be changed live in the sidebar.

### Step 4: CLTV segmentation with K-Means (`segment_kmeans`)
K-Means clustering on standardised [Recency, log Frequency, log Monetary, log CLTV]. The **elbow** and **silhouette** charts check that k = 3 is a sensible choice. The clusters are ranked by mean CLTV and named **High / Medium / Low CLTV**.

### Step 5: Campaign-fatigue framework (`add_fatigue`)
Campaigns are treated as a timeline (Cmp 1 → Cmp 6). The **ignore streak** is the number of campaigns in a row ignored since the customer's last "yes".

| Level | Rule |
|---|---|
| 🟢 Engaged | accepted ≥ 2 campaigns, including one of the last 2 |
| 🟡 Low Fatigue | last "yes" within the last 3 campaigns |
| 🟠 Moderate Fatigue | said yes before, then ignored ≥ 3 in a row (lapsed) |
| 🔴 High Fatigue | ignored all 6 campaigns |

**Fatigue Index (0–100)** = 60% ignore streak + 25% lifetime non-acceptance + 15% purchase recency.
**Silent loyalists** are highly fatigued customers who still buy recently and often. They are loyal to the store, not to the campaigns.

### Step 6: Machine-learning propensity models (`train_models`)
Three models predict *"will this customer accept a campaign like Cmp 6?"* using demographics, spend, channels and Cmp 1–5 history:

| Model | ROC-AUC | PR-AUC |
|---|---|---|
| **XGBoost** (gradient-boosted trees) | **0.907** | **0.679** |
| Logistic Regression | 0.895 | 0.626 |
| Random Forest | 0.892 | 0.634 |

Every score is **out-of-fold** (5-fold stratified cross-validation), so each customer is scored by a model that never saw them. **SHAP** explains which features push each prediction up or down.

### Step 7: Gains, lift and ROI simulator (`gains_table`)
Customers are ranked by propensity, and the table shows what share of responders you reach by contacting the top X%, the lift over random targeting, and the profit. The slider in the dashboard re-calculates all of these live.

### Step 8: Association rules (`association_rules_`, Apriori)
Customer traits are converted to yes/no flags (high income, wine/meat/gold buyer, catalog buyer, deal seeker, no kids, past acceptances…). Apriori then finds rules of the form **{traits} → responded to Cmp 6**, ranked by lift.

---

## 📊 Dashboard tour (7 tabs, 36 charts)

| Tab | Advanced charts |
|---|---|
| 🏠 **Executive Overview** | KPI cards · response rate vs break-even bar · donut of # campaigns accepted · P&L bar · strategy comparison |
| 📣 **Campaign Performance** | Acceptor-profile diverging heatmap · co-acceptance heatmap · **Sankey** response journey · demographic drill-down · **sunburst** · trajectory lines |
| 💎 **RFM & CLTV** | **3-D RFM scatter** (rotatable) · **treemap** of CLTV by segment → fatigue · RFM grid heatmap · **violin** plots · **radar** segment DNA · wallet-share stacked bars · **Pareto** curve · elbow & silhouette |
| 😴 **Campaign Fatigue** | Fatigue-index histogram · segment × fatigue heatmap (3 metrics) · fatigue-over-time heatmap · **value-at-risk quadrant** scatter |
| 🤖 **AI Predictive Lab** | Model leaderboard · ROC curves · SHAP importance · **SHAP beeswarm** · cumulative gains · lift · profit curve with optimum · **ROI simulator** slider · association-rule bubble chart |
| 🎯 **Strategy & Targeting** | Segment × fatigue **action matrix** · 6-step playbook · channel mix · **target-list builder** with CSV download |
| 🔍 **Customer Explorer** | Customer 360°: propensity **gauge**, campaign history, spend radar vs average, recommended play |

Global sidebar filters (segment, fatigue, education, marital status, income, age, kids) and business assumptions (cost, revenue, margin, discount) update every tab.

---

## 💡 Key insights

### 1. How are the campaigns performing?
* **All 6 campaigns lost money.** Response rates were 6.4%, 1.3%, 7.3%, 7.5%, 7.2% and 14.9%, against a 27.3% break-even. Total loss: about **-$29k**.
* **73% of customers never accepted any campaign**, 16.5% accepted exactly one, and only 0.4% accepted 5.
* **Each campaign reached a different persona.** Cmp 1, 2 and 5 reached affluent, childless, heavy wine and meat buyers (spend index 216–266). Cmp 3 reached deal-driven families. Cmp 6 reached long-tenured customers.
* **Responders repeat.** 67% of Cmp 2 acceptors and 55% of Cmp 1 acceptors also accepted Cmp 6, against a 15% base rate.

### 2. RFM-based CLTV segments
| Segment | Customers | Avg CLTV | Share of value | Cmp 6 response | Profile |
|---|---|---|---|---|---|
| **High** | ~640 | ~$920 | ~71% | 26% | Recent (~22 days), frequent, high spend |
| **Medium** | ~710 | ~$290 | ~25% | 13% | Same spend as High, but **~73 days since last purchase**: at-risk VIPs |
| **Low** | ~885 | ~$42 | ~4% | 8% | Half the income, more kids, 40% deal purchases, store-first, more gold and sweets |

* The **top 20% of customers hold ~66% of lifetime value**.

### 3. Campaign fatigue
* **73% are highly fatigued** (ignored all 6), 5% moderate (lapsed), 12% low, and 10% engaged.
* **~400 High-CLTV customers are moderately or highly fatigued.** This is the largest pool of value at risk.
* **~300 "silent loyalists"** keep buying but ignore every campaign. The fix is the channel, not a discount.
* Engaged customers convert at 85% and carry the highest CLTV. Limit how often you contact them so they don't tire too.

### 4. Strategy for a better response rate
1. **Stop mass mailing.** Contact the model's top 15–25%: 61–79% of responders are reached at 47–61% response, which turns a loss into a profit.
2. **Suppress never-responders** for 2–3 cycles. Reach silent loyalists in-store or through the loyalty programme.
3. **Win back Medium CLTV** customers (at-risk VIPs) with a time-bound, personalised wine and meat offer via catalog plus email.
4. **Persona-specific creative.** Premium offers for affluent childless buyers; value bundles for families and deal-seekers.
5. **Use acceptance history.** Past acceptors are 3–4× more likely to say yes, so build a "responder club".
6. **Cap frequency by fatigue level, keep a 10% control group** to measure true uplift, and retrain the model after every campaign.

---

## ⚠️ Assumptions and caveats
* Campaign order is assumed to be chronological (Cmp 1 → Cmp 5 → Response = Cmp 6).
* CLTV is a scenario estimate. Margin and discount rate are adjustable assumptions, not values taken from the data.
* Fatigue levels use all 6 campaigns, including Cmp 6. When the target-list builder excludes High-Fatigue customers, its "expected responders" figure is therefore optimistic.
* A probabilistic BG/NBD + Gamma-Gamma CLTV model was tested but rejected. The data is an aggregated snapshot rather than dated transactions, and frequency and monetary value are correlated (r = 0.64), which violates the Gamma-Gamma assumption.
