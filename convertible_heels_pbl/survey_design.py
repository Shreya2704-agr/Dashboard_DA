"""
Survey questionnaire for the Convertible Heels consumer study (India).

Single source of truth: the data generator, the cleaning pipeline and the
dashboard all read the question list and the answer options from here.

Every question is tagged with the algorithm(s) it feeds:
    CLS = Classification (predict purchase intent)
    CLU = Clustering     (segment customers / personas)
    ARM = Association Rule Mining (which features / pains / situations go together)
    REG = Regression     (predict willingness to pay)
    EDA = Descriptive & diagnostic analytics only
"""

CITIES_TIER1 = ["Mumbai", "Delhi NCR", "Bengaluru", "Hyderabad", "Chennai", "Pune", "Kolkata"]
CITIES_TIER2 = ["Ahmedabad", "Jaipur", "Lucknow", "Chandigarh", "Indore"]
CITIES = CITIES_TIER1 + CITIES_TIER2

OCCUPATIONS = [
    "Corporate professional", "Business owner / Entrepreneur", "Student",
    "Homemaker", "Healthcare professional", "Teacher / Academic",
    "Creative / Media", "Government employee",
]
MARITAL = ["Single", "Married"]
COMMUTE = ["Self-drive car", "Two-wheeler", "Cab / Auto", "Metro / Public transport", "Walk"]

HEEL_FREQ = ["Never", "Rarely (few times a year)", "Monthly", "Weekly",
             "Several times a week", "Daily"]          # ordinal 0..5
CARRY_FREQ = ["Never", "Rarely", "Sometimes", "Often", "Always"]  # Likert 1..5

CHANNELS = ["Myntra / Ajio", "Nykaa Fashion", "Amazon / Flipkart",
            "Brand website / Instagram", "Mall / Brand store", "Local market"]

HEEL_TYPES = ["Kitten", "Stiletto", "Block", "Wedge", "Platform", "Cone"]

PAINS = ["Knee pain", "Calf pain", "Heel pain / Plantar fasciitis",
         "Lower back pain", "Ankle strain", "Blisters / Shoe bite"]

SITUATIONS = ["Driving", "Office to party", "Weddings / Functions",
              "Travel", "Long standing events", "Walking commute"]

CONCERNS = ["Heel detaching / Safety", "Price", "Limited designs",
            "Durability", "Sizing / Fit", "Mechanism too complicated"]

FEATURES = [
    "Kitten heel module", "Stiletto heel module", "Block heel module",
    "Wedge heel module", "Orthopaedic cushioned insole", "Click-lock safety mechanism",
    "Travel pouch for heels", "Interchangeable straps", "Custom colours / prints",
    "Heel replacement subscription",
]

INTENT = ["Yes", "Maybe", "No"]

# (code, column name, question text, type, options/range, algorithms)
QUESTIONS = [
    # ---------- Section A: Profile ----------
    ("Q1", "age", "What is your age? (in years)", "Numeric", "16-75", "CLU, CLS, REG"),
    ("Q2", "city", "Which city do you live in?", "Single choice", ", ".join(CITIES), "CLS, EDA"),
    ("Q3", "occupation", "What is your occupation?", "Single choice", ", ".join(OCCUPATIONS), "CLU, CLS, REG"),
    ("Q4", "monthly_income_inr", "What is your monthly personal income / allowance? (₹)", "Numeric", "₹", "CLU, CLS, REG"),
    ("Q5", "marital_status", "Marital status", "Single choice", ", ".join(MARITAL), "EDA"),
    ("Q6", "commute_mode", "How do you usually commute?", "Single choice", ", ".join(COMMUTE), "CLS, CLU"),
    ("Q7", "hours_on_feet", "On a typical day, how many hours are you on your feet?", "Numeric", "0-16", "CLU, CLS"),
    # ---------- Section B: Footwear behaviour ----------
    ("Q8", "heel_frequency", "How often do you wear heels?", "Ordinal", ", ".join(HEEL_FREQ), "CLS, CLU, REG"),
    ("Q9", "heel_hours_per_wear", "When you wear heels, for how many hours at a stretch?", "Numeric", "0-16", "CLU, CLS"),
    ("Q10", "heel_pairs_owned", "How many pairs of heels do you own?", "Numeric", "0-40", "CLU, REG"),
    ("Q11", "annual_footwear_spend_inr", "Roughly how much do you spend on footwear in a year? (₹)", "Numeric", "₹", "CLU, REG"),
    ("Q12", "max_price_paid_inr", "What is the most you have paid for a single pair of heels/sandals? (₹)", "Numeric", "₹", "REG"),
    ("Q13", "preferred_channel", "Where do you usually buy footwear?", "Single choice", ", ".join(CHANNELS), "EDA, Strategy"),
    ("Q14", "heel_types_preferred", "Which heel types do you like wearing? (select all)", "Multi-select", ", ".join(HEEL_TYPES), "ARM"),
    # ---------- Section C: Pain points ----------
    ("Q15", "pains_experienced", "Which of these have you experienced after wearing heels? (select all)", "Multi-select", ", ".join(PAINS), "ARM, CLS, CLU"),
    ("Q16", "pain_severity", "How severe is the discomfort after a long day in heels? (1 = none, 10 = extreme)", "Scale 1-10", "1-10", "CLS, CLU, REG"),
    ("Q17", "carry_extra_pair", "How often do you carry an extra pair of footwear to switch?", "Likert 1-5", ", ".join(CARRY_FREQ), "CLS, CLU, REG"),
    ("Q18", "switch_situations", "In which situations do you need to switch between heels and flats? (select all)", "Multi-select", ", ".join(SITUATIONS), "ARM, CLS"),
    ("Q19", "importance_comfort", "How important is comfort when buying footwear? (1-5)", "Likert 1-5", "1-5", "CLU, CLS"),
    ("Q20", "importance_style", "How important is style / trend? (1-5)", "Likert 1-5", "1-5", "CLU, REG"),
    ("Q21", "importance_durability", "How important is durability / safety? (1-5)", "Likert 1-5", "1-5", "CLU"),
    # ---------- Section D: Product concept ----------
    ("Q22", "aware_of_convertible_heels", "Before today, had you heard of shoes whose heel can be removed or changed?", "Yes / No", "Yes, No", "CLS, EDA"),
    ("Q23", "features_interested", "Which features of a convertible heel sandal interest you? (select all)", "Multi-select", ", ".join(FEATURES), "ARM (launch sequence)"),
    ("Q24", "concerns", "What would worry you about buying it? (select all)", "Multi-select", ", ".join(CONCERNS), "ARM, EDA"),
    ("Q25", "trust_d2c_brands", "How much do you trust new online (D2C) brands? (1-5)", "Likert 1-5", "1-5", "CLS"),
    ("Q26", "influencer_influence", "How much do influencers / social media affect your purchases? (1-5)", "Likert 1-5", "1-5", "CLS, Strategy"),
    ("Q27", "wtp_base_inr", "How much would you pay for a convertible sandal with ONE pair of detachable heels? (₹)", "Numeric", "₹", "REG (target)"),
    ("Q28", "wtp_extra_heel_inr", "How much would you pay for each ADDITIONAL heel module? (₹)", "Numeric", "₹", "REG, Strategy"),
    ("Q29", "purchase_intent", "Given the opportunity, would you buy this product in the next 6 months?", "Single choice", "Yes, Maybe, No", "CLS (target label)"),
    ("Q30", "open_feedback", "What would make you (more) likely to buy it?", "Open text", "Free text", "Text analytics (future)"),
]

MULTI_SELECT = {
    "heel_types_preferred": HEEL_TYPES,
    "pains_experienced": PAINS,
    "switch_situations": SITUATIONS,
    "features_interested": FEATURES,
    "concerns": CONCERNS,
}
