"""Ingest and clean: raw ASEAN files and World Bank data into one harmonized panel.

Stages:
  pull_worldbank()  fetch (or load cached) World Bank indicators
  build_panel()     parse every raw file, harmonize, adjudicate outliers, pivot

All cleaning decisions are logged to a quirks list returned alongside the panel.
"""
import json
import time
import urllib.request

import numpy as np
import pandas as pd

from . import config as C


def canon_country(raw):
    name = str(raw).strip().rstrip("*").strip()
    return C.COUNTRY_MAP.get(name, (None,))[0]


def to_number(v):
    """Parse numbers that may carry spaces or commas as thousands separators."""
    if pd.isna(v):
        return np.nan
    s = str(v).replace("\xa0", " ").replace(",", "").replace(" ", "").strip()
    try:
        return float(s)
    except ValueError:
        return np.nan


def _melt_wide(df, indicator, suffix=""):
    country_col = df.columns[0]
    year_cols = {}
    for c in df.columns[1:]:
        stem = str(c).strip()
        if suffix and stem.endswith(suffix):
            stem = stem[: -len(suffix)]
        if len(stem) == 4 and stem.isdigit():
            year_cols[c] = int(stem)
    rows = []
    for _, r in df.iterrows():
        country = canon_country(r[country_col])
        if country is None:
            continue
        for c, year in year_cols.items():
            rows.append((country, year, indicator, to_number(r[c])))
    return pd.DataFrame(rows, columns=["country", "year", "indicator", "value"])


def _life_expectancy(quirks):
    df = pd.read_csv(C.RAW / "life_expentancy_rate.csv").rename(columns={"2004M.1": "2004F"})
    quirks.append("life_expectancy: duplicate 2004M header, second column is female (renamed)")
    rows = []
    for _, r in df.iterrows():
        country = canon_country(r[df.columns[0]])
        if country is None:
            continue
        for c in (str(x).strip() for x in df.columns):
            if len(c) == 5 and c[:4].isdigit() and c[4] in "MF":
                sex = "male" if c[4] == "M" else "female"
                rows.append((country, int(c[:4]), f"life_expectancy_{sex}", to_number(r[c])))
    return pd.DataFrame(rows, columns=["country", "year", "indicator", "value"])


def _long_density(fname, indicator):
    df = pd.read_csv(C.RAW / fname)
    df.columns = ["country", "year", "value"]
    df["country"] = df["country"].map(canon_country)
    df["indicator"] = indicator
    df["value"] = df["value"].map(to_number)
    return df.dropna(subset=["country"])[["country", "year", "indicator", "value"]]


def _adjudicate_outliers(tidy, quirks):
    """Flag local spikes (>3x from both neighbours) as suspected entry errors."""
    suspect = []
    for (country, ind), grp in tidy.groupby(["country", "indicator"]):
        grp = grp.sort_values("year")
        v = grp["value"].to_numpy()
        for i in range(1, len(grp) - 1):
            cur, prev, nxt = v[i], v[i - 1], v[i + 1]
            if min(cur, prev, nxt) <= 0:
                continue
            if (cur < prev / 3 and cur < nxt / 3) or (cur > prev * 3 and cur > nxt * 3):
                idx = grp.index[i]
                suspect.append(idx)
                quirks.append(f"{ind} {country} {int(grp.loc[idx,'year'])}: {cur:g} spikes vs "
                              f"{prev:g}/{nxt:g}, nulled as suspected entry error")
    edge = tidy[(tidy.indicator == "tb_prevalence") & (tidy.country == "Brunei Darussalam") & (tidy.year == 2004)]
    suspect.extend(edge.index)
    quirks.append("tb_prevalence Brunei 2004: edge value nulled (54 vs 224-311 later)")
    return tidy.drop(index=suspect)


def build_panel():
    """Return (panel DataFrame, quirks list). Panel is one row per country-year."""
    quirks = []
    frames = [_melt_wide(pd.read_csv(C.RAW / f), ind, suf) for f, ind, suf in C.WIDE_FILES]
    frames.append(_life_expectancy(quirks))
    gov = pd.read_csv(C.RAW / "goverment_expence_in_health.csv").drop(columns=["Indicators", "Unit"])
    frames.append(_melt_wide(gov, "govt_health_capex_musd"))
    frames.append(_long_density("physicans_density.csv", "physicians_per_1000"))
    frames.append(_long_density("nurses _ midwife_density.csv", "nurses_midwives_per_1000"))
    frames.append(_long_density("pharmaceutical_worker_density.csv", "pharma_workers_per_1000"))
    quirks.append("hiv_deaths: space thousands separators parsed")

    tidy = pd.concat(frames, ignore_index=True).dropna(subset=["value"])
    tidy = _adjudicate_outliers(tidy, quirks)

    le = tidy[tidy.indicator.str.startswith("life_expectancy_")]
    le_avg = le.pivot_table(index=["country", "year"], columns="indicator", values="value").mean(axis=1).reset_index()
    le_avg = le_avg.rename(columns={0: "value"})
    le_avg["indicator"] = "life_expectancy"
    tidy = pd.concat([tidy, le_avg], ignore_index=True)

    panel = tidy.pivot_table(index=["country", "year"], columns="indicator", values="value").reset_index()
    panel.insert(1, "iso3", panel["country"].map(C.ISO3))

    # Merge weather data if available
    weather_path = C.RAW / "weather.csv"
    if weather_path.exists():
        weather = pd.read_csv(weather_path)
        panel = panel.merge(weather, on=["iso3", "year"], how="left")

    return panel.sort_values(["country", "year"]).reset_index(drop=True), quirks


def pull_worldbank(use_cache=True):
    """Fetch World Bank indicators, or load the cached snapshot if offline.

    Caches to data/clean/worldbank.csv so the pipeline is reproducible without a
    network connection (important for the disconnected environments in the brief).
    """
    cache = C.CLEAN / "worldbank.csv"
    iso = list(C.ISO3.values())

    def fetch(code):
        url = (f"https://api.worldbank.org/v2/country/{';'.join(iso)}/indicator/{code}"
               "?format=json&per_page=500&date=2000:2019")
        for attempt in range(3):
            try:
                with urllib.request.urlopen(url, timeout=30) as r:
                    _, data = json.load(r)
                return pd.DataFrame(
                    [(d["countryiso3code"], int(d["date"]), d["value"]) for d in data if d["value"] is not None],
                    columns=["iso3", "year", C.WB_INDICATORS[code]])
            except OSError:
                if attempt == 2:
                    raise
                time.sleep(2 * (attempt + 1))

    try:
        frames = None
        for code in C.WB_INDICATORS:
            df = fetch(code)
            frames = df if frames is None else frames.merge(df, on=["iso3", "year"], how="outer")
        wb = frames.sort_values(["iso3", "year"])
        C.CLEAN.mkdir(parents=True, exist_ok=True)
        wb.to_csv(cache, index=False)
        return wb
    except OSError:
        if use_cache and cache.exists():
            print("World Bank fetch failed, using cached snapshot")
            return pd.read_csv(cache)
        raise
