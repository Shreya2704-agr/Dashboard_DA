"""
STEP 3 - Run every algorithm without the dashboard and save the results.

Useful for checking the numbers quickly, and as a backup if Streamlit
cannot be run. Output tables go to the outputs/ folder.
Run:  python 03_run_models.py
"""
import os
import sys
import warnings
import pandas as pd
import analytics as A

warnings.filterwarnings("ignore")
# Print the ₹ sign safely even when output is redirected to a file on Windows (default cp1252 encoding)
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)
df = pd.read_csv(os.path.join(HERE, "data", "clean_survey_data.csv"))
pd.set_option("display.width", 200, "display.max_columns", 20)

print("\n=== CLASSIFICATION (target: purchase_intent Yes/Maybe/No) ===")
C = A.run_classification(df)
print(C["results"].round(3).to_string(index=False))
C["results"].to_csv(os.path.join(OUT, "classification_results.csv"), index=False)
for name, cm in C["cms"].items():
    pd.DataFrame(cm, index=[f"Actual {c}" for c in A.CLASSES], columns=[f"Pred {c}" for c in A.CLASSES]) \
        .to_csv(os.path.join(OUT, f"confusion_matrix_{name.replace(' ', '_').lower()}.csv"))
print(f"\nBest model: {C['best']}\nConfusion matrix:\n{C['cms'][C['best']]}")
B = A.run_binary(df)
print(f"\nYes-vs-not (Gradient Boosting): accuracy {B['accuracy']:.3f}, precision {B['precision']:.3f}, "
      f"recall {B['recall']:.3f}, AUC {B['auc']:.3f}  (TP={B['tp']} FP={B['fp']} FN={B['fn']} TN={B['tn']})")

print("\n=== CLUSTERING (K-Means) ===")
print(A.cluster_scan(df).round(3).to_string(index=False))
K = A.run_kmeans(df, 4)
prof = K["profile"].sort_values("Yes_pct", ascending=False)
print(prof[["Name", "Size", "Avg_age", "Median_income", "Heel_freq_0_5", "Drives_pct", "Yes_pct",
            "Median_WTP", "Conversion_tier"]].round(2).to_string())
prof.to_csv(os.path.join(OUT, "kmeans_cluster_profiles.csv"))
L = A.run_lca(df, 4)
print(f"\nLatent Class Analysis (4 classes): BIC {L['bic']:.0f}, entropy {L['entropy']:.2f}, "
      f"class sizes {pd.Series(L['labels']).value_counts(normalize=True).round(2).to_dict()}")

print("\n=== ASSOCIATION RULES (likely buyers, features) ===")
buyers = df[df.purchase_intent != "No"]
sup, names, fi = A.apriori(A.baskets(buyers, ("features",)), 0.05, 3)
rules = A.association_rules(sup, names, 0.5).drop(columns=["_ante", "_cons"])
print(rules.head(15).round(3).to_string(index=False))
rules.to_csv(os.path.join(OUT, "association_rules.csv"), index=False)
seq = A.launch_sequence(buyers)
print("\nLaunch sequence:\n" + seq.to_string(index=False))
seq.to_csv(os.path.join(OUT, "launch_sequence.csv"), index=False)

print("\n=== REGRESSION (target: wtp_base_inr) ===")
R = A.run_regression(df)
print(R["results"].round(3).to_string(index=False))
R["results"].to_csv(os.path.join(OUT, "regression_results.csv"), index=False)
R["coefs"].to_csv(os.path.join(OUT, "regression_coefficients.csv"))

print("\n=== PRICING & BREAK-EVEN (default assumptions) ===")
dc = A.demand_curve(df, 0.3)
print("Revenue-maximising price:", int(dc.loc[dc["Revenue index"].idxmax(), "Price"]))
sim, unit, be, need = A.break_even(2499, 900, 699, 180, 0.35, 0.05, 0.20, 0.08, 120, 400, 350000, 2500000, 300, 0.10)
print(unit.round(0).to_string())
print(f"Orders/month to cover fixed costs: {need:,.0f}; cumulative break-even month: {be}")
sim.to_csv(os.path.join(OUT, "break_even_simulation.csv"), index=False)
print(f"\nAll tables saved to {OUT}")
