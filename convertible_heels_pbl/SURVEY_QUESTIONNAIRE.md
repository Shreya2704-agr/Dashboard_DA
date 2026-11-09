# Survey questionnaire: Convertible Heels consumer study (India)

**Target respondents:** women aged 16-75 in Indian Tier-1 and Tier-2 cities. **Length:** 30 questions, about 7 minutes.

**Design rules used**
- Age, income and ₹ amounts are asked as **exact numbers, not ranges**, so the models can work at the individual level. Bands are created later only for charts.
- Every question feeds at least one algorithm (see the last column).
- Q29 (purchase intent) is the **classification label** and Q27 (willingness to pay) is the **regression target**.
- Q14, Q15, Q18, Q23 and Q24 are **multi-select baskets** for association rule mining.
- Q30 (open text) is kept for future sentiment / text analytics.

Legend: CLS = Classification, CLU = Clustering, ARM = Association rule mining, REG = Regression, EDA = descriptive/diagnostic only.

| Q | Question | Type | Options / range | Used for | Column in data |
|---|---|---|---|---|---|
| Q1 | What is your age? (in years) | Numeric | 16-75 | CLU, CLS, REG | `age` |
| Q2 | Which city do you live in? | Single choice | Mumbai, Delhi NCR, Bengaluru, Hyderabad, Chennai, Pune, Kolkata, Ahmedabad, Jaipur, Lucknow, Chandigarh, Indore | CLS, EDA | `city` |
| Q3 | What is your occupation? | Single choice | Corporate professional, Business owner / Entrepreneur, Student, Homemaker, Healthcare professional, Teacher / Academic, Creative / Media, Government employee | CLU, CLS, REG | `occupation` |
| Q4 | What is your monthly personal income / allowance? (₹) | Numeric | ₹ | CLU, CLS, REG | `monthly_income_inr` |
| Q5 | Marital status | Single choice | Single, Married | EDA | `marital_status` |
| Q6 | How do you usually commute? | Single choice | Self-drive car, Two-wheeler, Cab / Auto, Metro / Public transport, Walk | CLS, CLU | `commute_mode` |
| Q7 | On a typical day, how many hours are you on your feet? | Numeric | 0-16 | CLU, CLS | `hours_on_feet` |
| Q8 | How often do you wear heels? | Ordinal | Never, Rarely (few times a year), Monthly, Weekly, Several times a week, Daily | CLS, CLU, REG | `heel_frequency` |
| Q9 | When you wear heels, for how many hours at a stretch? | Numeric | 0-16 | CLU, CLS | `heel_hours_per_wear` |
| Q10 | How many pairs of heels do you own? | Numeric | 0-40 | CLU, REG | `heel_pairs_owned` |
| Q11 | Roughly how much do you spend on footwear in a year? (₹) | Numeric | ₹ | CLU, REG | `annual_footwear_spend_inr` |
| Q12 | What is the most you have paid for a single pair of heels/sandals? (₹) | Numeric | ₹ | REG | `max_price_paid_inr` |
| Q13 | Where do you usually buy footwear? | Single choice | Myntra / Ajio, Nykaa Fashion, Amazon / Flipkart, Brand website / Instagram, Mall / Brand store, Local market | EDA, Strategy | `preferred_channel` |
| Q14 | Which heel types do you like wearing? (select all) | Multi-select | Kitten, Stiletto, Block, Wedge, Platform, Cone | ARM | `heel_types_preferred` |
| Q15 | Which of these have you experienced after wearing heels? (select all) | Multi-select | Knee pain, Calf pain, Heel pain / Plantar fasciitis, Lower back pain, Ankle strain, Blisters / Shoe bite | ARM, CLS, CLU | `pains_experienced` |
| Q16 | How severe is the discomfort after a long day in heels? (1 = none, 10 = extreme) | Scale 1-10 | 1-10 | CLS, CLU, REG | `pain_severity` |
| Q17 | How often do you carry an extra pair of footwear to switch? | Likert 1-5 | Never, Rarely, Sometimes, Often, Always | CLS, CLU, REG | `carry_extra_pair` |
| Q18 | In which situations do you need to switch between heels and flats? (select all) | Multi-select | Driving, Office to party, Weddings / Functions, Travel, Long standing events, Walking commute | ARM, CLS | `switch_situations` |
| Q19 | How important is comfort when buying footwear? (1-5) | Likert 1-5 | 1-5 | CLU, CLS | `importance_comfort` |
| Q20 | How important is style / trend? (1-5) | Likert 1-5 | 1-5 | CLU, REG | `importance_style` |
| Q21 | How important is durability / safety? (1-5) | Likert 1-5 | 1-5 | CLU | `importance_durability` |
| Q22 | Before today, had you heard of shoes whose heel can be removed or changed? | Yes / No | Yes, No | CLS, EDA | `aware_of_convertible_heels` |
| Q23 | Which features of a convertible heel sandal interest you? (select all) | Multi-select | Kitten heel module, Stiletto heel module, Block heel module, Wedge heel module, Orthopaedic cushioned insole, Click-lock safety mechanism, Travel pouch for heels, Interchangeable straps, Custom colours / prints, Heel replacement subscription | ARM (launch sequence) | `features_interested` |
| Q24 | What would worry you about buying it? (select all) | Multi-select | Heel detaching / Safety, Price, Limited designs, Durability, Sizing / Fit, Mechanism too complicated | ARM, EDA | `concerns` |
| Q25 | How much do you trust new online (D2C) brands? (1-5) | Likert 1-5 | 1-5 | CLS | `trust_d2c_brands` |
| Q26 | How much do influencers / social media affect your purchases? (1-5) | Likert 1-5 | 1-5 | CLS, Strategy | `influencer_influence` |
| Q27 | How much would you pay for a convertible sandal with ONE pair of detachable heels? (₹) | Numeric | ₹ | REG (target) | `wtp_base_inr` |
| Q28 | How much would you pay for each ADDITIONAL heel module? (₹) | Numeric | ₹ | REG, Strategy | `wtp_extra_heel_inr` |
| Q29 | Given the opportunity, would you buy this product in the next 6 months? | Single choice | Yes, Maybe, No | CLS (target label) | `purchase_intent` |
| Q30 | What would make you (more) likely to buy it? | Open text | Free text | Text analytics (future) | `open_feedback` |

## Which columns feed which algorithm

| Algorithm | Input columns | Target / output |
|---|---|---|
| Classification (KNN, Decision Tree, Random Forest, Gradient Boosting) | Q1-Q4, Q6-Q11, Q15-Q22, Q24-Q26 (one-hot pains, situations, concerns, occupation, commute; count of features wanted) | Q29 purchase intent (Yes / Maybe / No) |
| Clustering (K-Means) | age, log income, heel frequency, heel hours, carry-extra-pair, pain count, pain severity, comfort and style importance, influencer influence, drives self, number of switch situations, hours on feet | Cluster label; Q29 used only *afterwards* to rank clusters by conversion |
| Latent Class Analysis | Heel frequency, carries extra pair, pains, income band, commute, style importance, occupation (all categorical) | Latent class membership probabilities |
| Association rules (Apriori) | Q23 features basket (optionally with Q15 pains, Q18 situations, Q24 concerns) | Rules: support, confidence, lift; launch sequence |
| Regression (Linear, Ridge, Lasso) | Classification inputs + Q12 max price paid + Q23 feature flags | Q27 willingness to pay (₹) |
