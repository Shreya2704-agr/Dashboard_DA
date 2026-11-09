# Convertto: data-driven validation of a convertible-heel sandal for India

**Idea:** a sandal body that clicks onto interchangeable heels (kitten, stiletto, block, wedge) and turns into a flat anytime, anywhere. Launch market: India.

**Goal of this project:** validate the idea before launch. The analysis works out who will buy, which customer segments to target, which features to launch first, what price to charge, and when the business breaks even. It uses a simulated consumer survey (2,000 respondents, 30 questions) and four families of algorithms.

**Report:** `Convertto_Report.docx` (5 pages; a PDF copy is included as `Convertto_Report.pdf`).

## What is in this folder

| File / folder | What it does |
|---|---|
| `Convertto_Report.docx` / `.pdf` | The written report, with dashboard screenshots |
| `survey_design.py` | The 30-question survey and its answer options (single source of truth) |
| `SURVEY_QUESTIONNAIRE.md` | Readable questionnaire, with the algorithm each question feeds |
| `01_generate_synthetic_data.py` | Creates 2,000 realistic mock respondents (+35 duplicates) with intentional noise, missing values, outliers and typos -> `data/raw_survey_responses.csv` |
| `pipeline.py` | Cleaning and transformation steps, each with a logged reason |
| `02_clean_data.py` | Runs the pipeline -> `data/clean_survey_data.csv` and `data/cleaning_log.csv` |
| `analytics.py` | Classification, clustering (K-Means + Latent Class Analysis), Apriori association rules, Linear/Ridge/Lasso regression, pricing curve, break-even model |
| `03_run_models.py` | Runs every model in the terminal and saves result tables to `outputs/` |
| `app.py` | **Streamlit dashboard** (see below) |
| `.streamlit/config.toml` | Dashboard colours and theme |
| `data/` | Raw survey, clean survey and the cleaning log |
| `outputs/` | Result tables written by `03_run_models.py` |
| `requirements.txt` | Python packages needed |

## How to run (Windows / Mac)

```bash
# 1. open a terminal inside this folder
python -m venv venv
# Windows: venv\Scripts\activate      Mac: source venv/bin/activate
pip install -r requirements.txt

# 2. (optional) regenerate and re-clean the data - the CSVs are already included
python 01_generate_synthetic_data.py
python 02_clean_data.py

# 3. optional: see all model results in the terminal
python 03_run_models.py

# 4. launch the dashboard (opens in the browser at http://localhost:8501)
streamlit run app.py
```

Tested with Python 3.12. The first visit to the Classification and Clustering views takes 10-20 seconds while the models train; after that, results are cached.

**Windows tip:** extract the zip to a short path such as `C:\Convertto` or the Desktop. The installed packages contain very long file names, and `pip install` can fail with "filename too long" if the folder is buried deep inside other folders (Windows' 260-character path limit).

## The dashboard

The Home page has four buttons, one for each level of analytics, plus a Survey & data page. Every chart is followed by an **Insight** and a **Decision** computed from the data.

| Button | Question it answers | Views inside |
|---|---|---|
| **Descriptive** | What happened? | Who they are; Pains & needs; Money & channels; Customer segments |
| **Diagnostic** | Why did it happen? | Drivers of intent (correlations); Behaviour vs intent; Price & income (ANOVA); Significance tests (chi-square); Why "Maybe"? |
| **Predictive** | What will happen? | Algorithm map (inputs, targets and outputs of every model); Classification (KNN, Decision Tree, Random Forest, Gradient Boosting: accuracy, precision, recall, F1, ROC-AUC, confusion matrices); Regression (Linear, Ridge, Lasso); Clustering (K-Means and Latent Class Analysis); Association rules (Apriori); Score a prospect |
| **Prescriptive** | What should we do? | Launch plan; Price; Who to target; Break-even, followed by limitations and next steps |
| **Survey & data** | How was the data made and cleaned? | Questionnaire; Raw data quality; Cleaning before vs after; Cleaning log and clean data |

The Home page also has a "How the survey data was created" panel.

## Using real survey data later
Export the real survey with the same column names as `data/raw_survey_responses.csv` and upload it in the dashboard sidebar. It goes through the same cleaning pipeline and every model retrains automatically.
