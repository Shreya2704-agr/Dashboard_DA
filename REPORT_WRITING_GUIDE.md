# Report writing guide (for you, not for submission)

Your professor said the report must be **in your own words**: screenshots from the dashboard plus your own explanation of what you understood. This guide follows the structure from the assignment brief (section 5). For each section it gives the points to cover, the screenshots to take and questions to answer in your own words. The dashboard text under every chart (*What it shows / Insight / So what*) is there to help you understand each chart. Rephrase it in your own voice; don't copy it.

---

## 1. Abstract (5-6 lines, write it last)
Answer in your own words: What is the product? What did you want to find out? How (survey design, synthetic data, 4 algorithm families)? What did you find (one number each for demand, best segment, launch features, price, break-even)? Is the idea sustainable?

## 2. Introduction and business idea
- Your personal motivation. You called this idea "your own child", so say why it matters to you.
- The customer pain: carrying an extra pair, driving in heels, office-to-party changes, knee/calf/plantar fasciitis pain.
- Why India first: few local players, low awareness. Mention the global brands (Pashion, Mime et Moi, Tanya Heath, Mestrae at roughly $100-$300) and the Indian footwear market (about USD 22.7 bn in 2026, women about 54%, per IMARC).
- 📸 Dashboard page 1: the KPI row and the "Business question → Technique" table.

## 3. Domain of dataset and objectives
- Domain: consumer survey, women's footwear, India.
- Objectives (list them): (1) predict who will buy, (2) segment customers, (3) decide which features to launch first, (4) estimate willingness to pay, (5) price and break-even.
- Explain the four levels: descriptive, diagnostic, predictive, prescriptive, and which page covers each.

## 4. Framework
Draw (by hand or in PowerPoint) the flow: secondary research → survey design → synthetic data → cleaning → EDA → 4 algorithms → strategy → plug in real data later.
Explain **why synthetic data**: a real survey takes about a month, so the model is built and tested now and is ready when real data arrives.

## 5. Data collected (survey design)
- 📸 Page 2, "Questionnaire" tab.
- Explain why you asked **exact numbers instead of ranges** (the professor stressed this in class).
- Explain which question is the classification label (Q29), the regression target (Q27) and the association baskets (Q14, Q15, Q18, Q23, Q24).
- Mention 2,000 respondents across 12 cities and that the data was simulated from 5 realistic consumer types with random noise.

## 6. Data cleaning and transformation (10 marks)
- 📸 Page 2: the "missing answers" chart, the "messy answers" table and the cleaning log.
- For each problem, write one line in your words: what it was, what you did and **why**. The problems are duplicates, city spellings, intent label variants, impossible ages and WTP, out-of-range Likert values, outliers capped at Q3 + 3×IQR, and missing values filled by median (income by occupation median).
- Transformations: one-hot encoding of multi-selects, ordinal heel frequency, bands, log income, intent score.

## 7. Descriptive and diagnostic analytics (30 marks, the biggest part)
The brief says *"every graph should be explained logically"*. Pick **8-12 charts** from pages 3 and 4. For each one write 2-4 sentences: what the chart shows, what you notice (with a number), and what decision it leads to.
Strong choices:
- Purchase intent split · Yes-rate by occupation · heel frequency · pains · features wanted · concerns · WTP histogram (page 3)
- Correlation matrix · intent by "carry an extra pair" · intent by commute (self-drivers) · WTP by income · chi-square table · concerns of Maybe vs Yes (page 4)

## 8. Algorithms applied
For each algorithm write: **why this algorithm** (what business question, and why supervised or unsupervised), **which columns**, **how you evaluated it**, **result**, **decision**.

| Algorithm | Screenshot | Points you must be able to explain |
|---|---|---|
| Classification | Page 5: metrics table, confusion matrices, TP/FP/FN/TN matrix, ROC, feature importance | Why compare 4 models; why accuracy alone is not enough; what FP and FN cost *this* business; why Random Forest train accuracy (96%) vs test shows overfitting; why "Maybe" is hard to predict |
| Clustering | Page 6: elbow + silhouette, profile table, conversion chart, LCA table | Why k = 4; why intent was **not** an input; persona names; high/medium/low conversion tiers; how LCA differs from K-Means |
| Association rules | Page 7: rules table, top rules chart, launch sequence | Support vs confidence vs lift with one real example; why the travel pouch is bundled at launch even though it isn't top-4 |
| Regression | Page 8: results table, predicted vs actual, coefficients | R² and MAE in plain words; what Ridge and Lasso add; which features Lasso dropped |

## 9. Results and strategy
- 📸 Page 9: demand/revenue curve, targeting table, break-even chart, unit economics, sensitivity grid.
- Write your own go-to-market plan: who first, which kit, what price (mention the GST step at ₹2,500), which channels, what message, how to convert the Maybes.
- State your verdict on sustainability and the conditions it depends on (CAC, attach rate, funding needed). Change some break-even sliders and describe what happens. That shows understanding.

## 10. Conclusion and limitations
- Your answer: is the idea worth pursuing?
- Limitations: synthetic data (it carries the assumptions it was built on), stated intent overstates real buying, mostly metro sample, cost figures are assumptions.
- Next steps: run the real survey (upload in the sidebar), do a pre-order pilot, analyse open-text feedback with sentiment analysis, and validate in Tier-2 cities.

---

## Key numbers from the current dataset (to quote)
| Metric | Value |
|---|---|
| Respondents after cleaning | 2,000 (35 duplicates removed) |
| Intent | Yes 35% · Maybe 30% · No 35% |
| Best 3-class classifier | Gradient Boosted Trees, accuracy ≈ 67%, macro-F1 0.65 |
| Yes-vs-not model | accuracy 82%, precision 73%, recall 75%, AUC 0.90 |
| Top segment | "Driving daily heel-wearers", ≈27% of market, 75% Yes, median WTP ≈ ₹4,540 |
| Launch features | Orthopaedic insole, click-lock, kitten heel, block heel + travel pouch (bundled) |
| Strongest rule | Kitten + Block heel → Travel pouch: confidence 67%, lift 2.1 |
| Regression | Lasso R² 0.86, MAE ≈ ₹452, uses 24 of 62 inputs |
| Revenue-maximising price | ≈ ₹3,199 (₹2,499 entry kit stays in the 5% GST slab) |
| Break-even (default assumptions) | ≈ 879 orders/month to cover fixed costs; cumulative break-even ≈ month 25 |

## Questions your professor may ask (prepare answers in your own words)
1. Why did you use exact numbers instead of ranges in the survey?
2. Your accuracy is 67%. Would you deploy this model? (Hint: 3 classes, Maybe is ambiguous, the Yes/No confusion is small, and the binary model's AUC is 0.90.)
3. In your business, which is worse: a false positive or a false negative?
4. Why didn't you use purchase intent as an input to clustering?
5. What does a lift of 2.1 mean?
6. Why Lasso over Linear regression if R² is nearly the same?
7. Your data is synthetic. How do you know the models will work on real data?
8. What is the single biggest risk to break-even?
