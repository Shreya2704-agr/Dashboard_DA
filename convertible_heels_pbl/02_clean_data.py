"""
STEP 2 - Clean & transform the raw survey export.

Input : data/raw_survey_responses.csv   (synthetic now, real survey later)
Output: data/clean_survey_data.csv
        data/cleaning_log.csv          (every action, reason and rows affected)
Run   : python 02_clean_data.py
"""
import os
import pandas as pd
from pipeline import clean

HERE = os.path.dirname(os.path.abspath(__file__))
raw = pd.read_csv(os.path.join(HERE, "data", "raw_survey_responses.csv"))
clean_df, log = clean(raw)

clean_df.to_csv(os.path.join(HERE, "data", "clean_survey_data.csv"), index=False)
log.to_csv(os.path.join(HERE, "data", "cleaning_log.csv"), index=False)

pd.set_option("display.max_colwidth", 80, "display.width", 200)
print(log[["step", "action", "records_affected"]].to_string(index=False))
print("\nMissing values left in numeric columns:", int(clean_df.select_dtypes("number").isna().sum().sum()))
print(clean_df["purchase_intent"].value_counts())
