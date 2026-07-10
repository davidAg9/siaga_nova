"""Central configuration for the SIAGA pipeline: paths, schema, feature spec.

Everything the pipeline needs to be reproducible lives here, so retraining on new
data is a matter of dropping files into data/raw and re-running the pipeline.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]        # siaga/
RAW = ROOT / "data" / "raw"
CLEAN = ROOT / "data" / "clean"
MODELS = ROOT / "models"
API_DATA = ROOT / "api" / "data"

# One canonical name + ISO3 per country, keyed by every raw-file variant.
COUNTRY_MAP = {
    "Brunei Darussalam": ("Brunei Darussalam", "BRN"), "Brunnei Darussalam": ("Brunei Darussalam", "BRN"),
    "Cambodia": ("Cambodia", "KHM"), "Indonesia": ("Indonesia", "IDN"),
    "Lao's PDR": ("Lao PDR", "LAO"), "Lao PDR": ("Lao PDR", "LAO"),
    "Lao People's Democratic Republic": ("Lao PDR", "LAO"), "Malaysia": ("Malaysia", "MYS"),
    "Myanmar": ("Myanmar", "MMR"), "Philippines": ("Philippines", "PHL"),
    "Singapore": ("Singapore", "SGP"), "Thailand": ("Thailand", "THA"), "Viet Nam": ("Vietnam", "VNM"),
}
ISO3 = {v[0]: v[1] for v in COUNTRY_MAP.values()}

# Wide raw files: (file, indicator name, year-column suffix to strip)
WIDE_FILES = [
    ("crude_birth_ratio.csv", "crude_birth_rate", ""), ("crude_death_ratio.csv", "crude_death_rate", ""),
    ("Infant_mortality_rate.csv", "infant_mortality", ""), ("under_5_mortality_rate.csv", "under5_mortality", ""),
    ("maternal_mortality_rate.csv", "maternal_mortality", ""), ("TB_Prevalence.csv", "tb_prevalence", "TBC"),
    ("malaria_prevalence.csv", "malaria_cases", "Malaria"), ("immunization_DPT.csv", "immunization_dpt", "DPT"),
    ("immunization_measless.csv", "immunization_measles", "Measles"), ("HIV_Prevalence.csv", "hiv_prevalence", ""),
    ("death_by_HIV_ AIDS.csv", "hiv_deaths", ""), ("undernourished_population.csv", "undernourished_pct", ""),
    ("underweight_children.csv", "underweight_children_pct", ""),
]

# World Bank Open Data indicators (API v2). Used for validation, per-capita
# normalization, and the contextual predictors (GDP per capita, sanitation).
WB_INDICATORS = {
    "SP.DYN.LE00.IN": "wb_life_expectancy", "SH.MED.PHYS.ZS": "wb_physicians_per_1000",
    "SH.MED.NUMW.P3": "wb_nurses_midwives_per_1000", "SH.XPD.CHEX.PC.CD": "wb_health_exp_per_capita_usd",
    "SH.IMM.IDPT": "wb_immunization_dpt", "SH.IMM.MEAS": "wb_immunization_measles",
    "SN.ITK.DEFC.ZS": "wb_undernourished_pct", "SH.TBS.INCD": "wb_tb_incidence_per_100k",
    "SP.POP.TOTL": "wb_population", "SP.DYN.CBRT.IN": "wb_crude_birth_rate",
    "NY.GDP.PCAP.CD": "wb_gdp_per_capita", "SH.STA.BASS.ZS": "wb_sanitation_pct",
}

TARGET = "life_expectancy"
TRAIN_YEARS = (2004, 2014)      # modeling window from the hackathon data
VALIDATION_YEARS = (2015, 2019)  # out-of-source window scored against World Bank

# The model feature set, with monotonic priors:
#   +1  more of this can never LOWER predicted life expectancy
#   -1  more of this can never RAISE it
#    0  unconstrained
# Actionable levers and contextual determinants are both included; the simulator
# exposes only the actionable ones and holds context constant.
FEATURE_SPEC = [
    ("physicians_per_1000", 1, "actionable"),
    ("nurses_midwives_per_1000", 1, "actionable"),
    ("pharma_workers_per_1000", 1, "actionable"),
    ("immunization_dpt", 1, "actionable"),
    ("immunization_measles", 1, "actionable"),
    ("tb_prevalence", -1, "actionable"),
    ("hiv_prevalence", -1, "actionable"),
    ("undernourished_pct", -1, "actionable"),
    ("crude_birth_rate", 0, "context"),
    ("log_capex_per_capita", 1, "actionable"),
    ("log_gdp_per_capita", 1, "context"),
    ("sanitation_pct", 1, "actionable"),
    ("year", 0, "context"),
]
FEATURES = [f for f, _, _ in FEATURE_SPEC]
MONO = [m for _, m, _ in FEATURE_SPEC]
ACTIONABLE = [f for f, _, kind in FEATURE_SPEC if kind == "actionable"]

# Tuned hyperparameters (from workbench/11 grouped search, chosen for the best
# out-of-source generalization, not just in-sample fit).
GBM_PARAMS = dict(max_depth=3, learning_rate=0.08, max_iter=400, random_state=42)
