# 🏦 Universal Bank: Personal Loan Campaign Intelligence

A Streamlit dashboard that uses **classification, clustering and association rule mining** to answer four questions about the last personal-loan campaign (5,000 existing customers, 9.6% acceptance):

| Layer | Question | Main visuals |
|---|---|---|
| Descriptive | What happened? | KPI cards, outcome donut, product ownership, click-to-drill acceptance bars |
| Diagnostic | Why? | Two-way acceptance heatmap, *stacking-signals* waterfall, sunburst drill-down, acceptor vs decliner ratios |
| Predictive | Who will say yes? | Model comparison, ROC, confusion matrix, feature importance, plain-English tree rules, decile lift, what-if scorer |
| Prescriptive | What should we do? | Budget slider with spend and loans waterfalls, budget curve, segment playbook, downloadable target list |

Plus a **Segments** tab (K-Means or Gaussian Mixture, LCA-style) and a **Product Affinity** tab (Apriori rules, lift heatmaps, per-segment rules).

> **Headline (back-test on the last campaign):** contacting only the top ~11% of customers ranked by the Random Forest score would have won **95% of the loans** at **~11% of the cost**. Contacting the top 50% (the goal) keeps **100%** of loans and doubles conversion to **19.2%**.

## Run

From the repository root:

```bash
.venv\Scripts\streamlit run personal_loan/app.py --server.port 8502
```

Or double-click `run_loan_dashboard.bat`. First load takes about 5 seconds while the models train; after that everything is cached.

## Files

| File | Purpose |
|---|---|
| `data/UniversalBank.csv` | Raw data (5,000 × 14) |
| `engine.py` | All analytics, with **no UI code**, ready to reuse in an API or batch job: cleaning, models, clustering, rules, budget simulation |
| `app.py` | Streamlit UI (8 tabs, sidebar filters, CSV exports) |

## Method in brief

* **Cleaning:** 52 negative `Experience` values clipped to 0. `Experience` (r = 0.99 with Age) and `ZIP Code` are excluded from models. ZIP is mapped to a region for filtering.
* **Classification:** Logistic Regression, Decision Tree (depth 4, readable rules) and Random Forest on a 70/30 stratified hold-out. The best model by AUC (Random Forest, 0.998) produces **5-fold out-of-fold** scores, so every customer is scored by a model that never saw them.
* **Clustering:** on profile only (income, card spend, family, education, mortgage), so product ownership and loan uptake can be compared across segments afterwards. Segments are auto-named and ordered by acceptance rate.
* **Association rules:** Apriori over product flags plus optional profile flags. Loan-focused rules and cross-sell rules, filterable by segment.
* **Budget simulation:** rank by score, contact the top *x%*, and compare spend, loans, conversion, cost per loan and profit with "contact everyone". Cost per contact and value per loan are editable assumptions; all % results are independent of them.

Total compute is under 10 seconds on a laptop, light enough to retrain after every campaign.
