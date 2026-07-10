"""Feature build, model training, validation, explainability, and forecasting.

The reference model is a monotonicity-constrained gradient-boosting regressor.
A sign-constrained linear surrogate is exported for edge inference in the API.
"""
import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from statsmodels.tsa.exponential_smoothing.ets import ETSModel

from . import config as C


def build_frame(panel, wb):
    """Merge World Bank context, interpolate interior gaps, engineer features.

    Returns the full annual frame (all years) with the model feature columns.
    """
    wb_cols = ["iso3", "year", "wb_population", "wb_gdp_per_capita", "wb_sanitation_pct", "wb_agriculture_pct_gdp"]
    df = panel.merge(wb[wb_cols], on=["iso3", "year"], how="left")
    interp = ["physicians_per_1000", "nurses_midwives_per_1000", "pharma_workers_per_1000",
              "immunization_dpt", "immunization_measles", "tb_prevalence", "hiv_prevalence",
              "undernourished_pct", "crude_birth_rate", "govt_health_capex_musd",
              "wb_population", "wb_gdp_per_capita", "wb_sanitation_pct", "wb_agriculture_pct_gdp"]
    out = []
    for _, g in df.groupby("country"):
        g = g.sort_values("year").set_index("year")
        g[interp] = g[interp].interpolate(method="linear", limit_area="inside")
        out.append(g.reset_index())
    data = pd.concat(out)
    # Singapore undernourishment reported by FAO as "<2.5%" (the floor).
    data.loc[data.country == "Singapore", "undernourished_pct"] = (
        data.loc[data.country == "Singapore", "undernourished_pct"].fillna(2.5))
    data["log_capex_per_capita"] = np.log1p(data["govt_health_capex_musd"] * 1e6 / data["wb_population"])
    data["log_gdp_per_capita"] = np.log1p(data["wb_gdp_per_capita"])
    data["sanitation_pct"] = data["wb_sanitation_pct"]
    data["agriculture_pct_gdp"] = data["wb_agriculture_pct_gdp"]
    return data


def make_model(**overrides):
    params = {**C.GBM_PARAMS, **overrides}
    return HistGradientBoostingRegressor(monotonic_cst=C.MONO, **params)


def training_frame(data):
    lo, hi = C.TRAIN_YEARS
    return data[data.year.between(lo, hi)].dropna(subset=[C.TARGET]).reset_index(drop=True)


def validate(train, panel, wb):
    """Three honest tests: unseen country, future years, out-of-source vs World Bank."""
    X, y, groups = train[C.FEATURES], train[C.TARGET], train["country"]
    metrics = {}

    loco = cross_val_predict(make_model(), X, y, cv=GroupKFold(10), groups=groups)
    metrics["loco_cv"] = _score(y, loco)

    tr, te = train.year <= 2011, train.year >= 2012
    p = make_model().fit(X[tr], y[tr]).predict(X[te])
    metrics["temporal_2012_2014"] = _score(y[te], p)

    oot_X, oot_y = _oot_frame(train, panel, wb)
    op = make_model().fit(X, y).predict(oot_X)
    ok = oot_y.notna()
    metrics["worldbank_oot_2015_2019"] = {**_score(oot_y[ok], op[ok]), "n": int(ok.sum())}
    return metrics


def _score(y, p):
    return {"r2": round(r2_score(y, p), 3),
            "rmse_years": round(float(np.sqrt(np.mean((y - p) ** 2))), 2),
            "mae_years": round(mean_absolute_error(y, p), 2)}


def _oot_frame(train, panel, wb):
    """Build the 2015-2019 feature frame from World Bank, mapped to our schema."""
    tb_ratio = (panel.merge(wb[["iso3", "year", "wb_tb_incidence_per_100k"]], on=["iso3", "year"])
                .dropna(subset=["tb_prevalence", "wb_tb_incidence_per_100k"])
                .groupby("iso3").apply(lambda g: g["tb_prevalence"].mean() / g["wb_tb_incidence_per_100k"].mean(),
                                       include_groups=False))
    lo, hi = C.VALIDATION_YEARS
    oot = wb[wb.year.between(lo, hi)].copy()
    hiv16 = panel[panel.year == 2016].set_index("iso3")["hiv_prevalence"]
    colmap = {
        "physicians_per_1000": oot.wb_physicians_per_1000, "nurses_midwives_per_1000": oot.wb_nurses_midwives_per_1000,
        "pharma_workers_per_1000": np.nan, "immunization_dpt": oot.wb_immunization_dpt,
        "immunization_measles": oot.wb_immunization_measles,
        "tb_prevalence": oot.wb_tb_incidence_per_100k * oot.iso3.map(tb_ratio).values,
        "hiv_prevalence": oot.iso3.map(hiv16).values, "undernourished_pct": oot.wb_undernourished_pct,
        "crude_birth_rate": oot.wb_crude_birth_rate, "log_capex_per_capita": np.nan,
        "log_gdp_per_capita": np.log1p(oot.wb_gdp_per_capita), "sanitation_pct": oot.wb_sanitation_pct,
        "agriculture_pct_gdp": oot.wb_agriculture_pct_gdp if "wb_agriculture_pct_gdp" in oot.columns else np.nan,
        "year": oot.year}
    return pd.DataFrame({f: colmap[f] for f in C.FEATURES}), oot["wb_life_expectancy"].reset_index(drop=True)


def linear_surrogate(train):
    """Sign-constrained standardized linear approximation for edge inference."""
    X = train[C.FEATURES].apply(lambda c: c.fillna(c.median()))
    y = train[C.TARGET]
    mean, std = X.mean(), X.std().replace(0, 1)
    Z = (X - mean) / std
    lo = [0.0 if m == 1 else -np.inf for m in C.MONO]
    hi = [0.0 if m == -1 else np.inf for m in C.MONO]
    fit = lsq_linear(Z.to_numpy(), (y - y.mean()).to_numpy(), bounds=(lo, hi))
    return {"features": C.FEATURES, "mean": mean.round(6).tolist(), "std": std.round(6).tolist(),
            "coef": np.round(fit.x, 6).tolist(), "intercept": round(float(y.mean()), 6),
            "note": "sign-constrained standardized linear surrogate of the reference GBM",
            "in_sample_r2": round(float(r2_score(y, float(y.mean()) + Z.to_numpy() @ fit.x)), 3)}


def shap_importance(model, train):
    import shap
    X = train[C.FEATURES].fillna(train[C.FEATURES].median())
    sv = shap.TreeExplainer(model).shap_values(X)
    imp = pd.Series(np.abs(sv).mean(0), index=C.FEATURES)
    return imp.round(4).to_dict()


def _forecast_one(series, horizon=5):
    s = series.dropna()
    if len(s) < 6:
        return None
    ls = np.log1p(s)

    def ets(train_s, h):
        fit = ETSModel(train_s, error="add", trend="add", damped_trend=True).fit(disp=False)
        pr = fit.get_prediction(start=len(train_s), end=len(train_s) + h - 1)
        return np.expm1(pr.summary_frame(alpha=0.2))

    ets_mae = mean_absolute_error(s.iloc[-3:], ets(ls.iloc[:-3], 3)["mean"])
    naive_mae = mean_absolute_error(s.iloc[-3:], [s.iloc[-4]] * 3)
    if ets_mae <= naive_mae:
        return ets(ls, horizon), ets_mae, naive_mae, "ets_log"
    sd = s.diff().std()
    frame = pd.DataFrame({"mean": [s.iloc[-1]] * horizon,
                          "pi_lower": [max(0, s.iloc[-1] - 1.28 * sd * np.sqrt(h + 1)) for h in range(horizon)],
                          "pi_upper": [s.iloc[-1] + 1.28 * sd * np.sqrt(h + 1) for h in range(horizon)]})
    return frame, ets_mae, naive_mae, "naive"


def forecast_diseases(data):
    lo, hi = C.VALIDATION_YEARS
    rows = []
    for disease, col in [("tb", "tb_prevalence"), ("malaria", "malaria_cases")]:
        for country, g in data.groupby("country"):
            s = g.set_index("year")[col]
            s.index = pd.PeriodIndex(s.index, freq="Y")
            res = _forecast_one(s[s.index.year <= C.TRAIN_YEARS[1]])
            if res is None:
                continue
            frame, e, n, method = res
            for yr, r in zip(range(lo, hi + 1), frame.itertuples()):
                rows.append((disease, country, yr, r.mean, r.pi_lower, r.pi_upper, e, n, method))
    return pd.DataFrame(rows, columns=["disease", "country", "year", "forecast", "lo80", "hi80",
                                       "ets_backtest_mae", "naive_backtest_mae", "method"])
