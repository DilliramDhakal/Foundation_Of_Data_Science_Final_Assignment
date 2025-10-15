# =========================================================
# Bat–Rat Investigations (A, B, C) + Rich Visualizations
# =========================================================

import math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import binomtest
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import statsmodels.api as sm

# --------------------------
# Paths & globals
# --------------------------
DATA1 = Path("dataset1.csv")
DATA2 = Path("dataset2.csv")
OUTDIR = Path("model_summaries")
CHARTS = Path("charts")
OUTDIR.mkdir(exist_ok=True)
CHARTS.mkdir(exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid")
sns.set_context("talk")

# --------------------------
# Helpers
# --------------------------
def to_dt(series):
    return pd.to_datetime(series, errors="coerce", dayfirst=True, infer_datetime_format=True)

def col_any(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def wilson_ci(k, n, alpha=0.05):
    if n == 0:
        return (np.nan, np.nan)
    z = 1.959963984540054
    p = k / n
    den = 1 + (z**2)/n
    centre = p + (z**2)/(2*n)
    rad = z * math.sqrt((p*(1-p) + (z**2)/(4*n))/n)
    return (centre - rad)/den, (centre + rad)/den

def savefig(name: str, tight=True):
    path = CHARTS / f"{name}.png"
    if tight: plt.tight_layout()
    plt.savefig(path, dpi=160)
    print(f"[saved] charts/{name}.png")

def print_model_sklearn(label, model, Xcols, y_true, y_pred):
    print(f"\n[{label}] Linear Regression (scikit-learn)")
    for name, coef in zip(Xcols, model.coef_):
        print(f"  coef[{name:>20}] = {coef: .4f}")
    print(f"  intercept               = {model.intercept_: .4f}")
    print(f"  R^2 (test)              = {r2_score(y_true, y_pred): .3f}")

    # Pred vs Actual
    plt.figure(figsize=(7,5))
    plt.scatter(y_true, y_pred, alpha=0.7)
    lo, hi = float(np.min(y_true)), float(np.max(y_true))
    plt.plot([lo, hi], [lo, hi], "r--")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"{label} — Predicted vs Actual")
    savefig(f"{label.lower().replace(' ', '_')}_pred_vs_actual")
    plt.show()

    # Residuals
    residuals = y_true - y_pred
    plt.figure(figsize=(7,5))
    sns.histplot(residuals, bins=20, kde=True)
    plt.title(f"{label} — Residuals")
    plt.xlabel("Residual")
    savefig(f"{label.lower().replace(' ', '_')}_residuals")
    plt.show()

def print_and_save_ols(label, X, y, out_name):
    Xc = sm.add_constant(X)
    model = sm.OLS(y, Xc).fit()
    txt = model.summary().as_text()
    print(f"\n--- OLS Summary: {label} ---")
    print(txt)
    (OUTDIR / out_name).write_text(txt, encoding="utf-8")
    return model

# --------------------------
# Load & preprocess dataset1
# --------------------------
d1 = pd.read_csv(DATA1)

for col in ["start_time", "rat_period_start", "rat_period_end", "sunset_time"]:
    if col in d1.columns:
        d1[col] = to_dt(d1[col])

if "start_time" in d1.columns and "bat_landing_to_food" in d1.columns:
    d1["eat_ts"] = d1["start_time"] + pd.to_timedelta(d1["bat_landing_to_food"], unit="s")

if "rat_period_end" in d1.columns and "eat_ts" in d1.columns:
    d1["secs_before_rat_left"] = (d1["rat_period_end"] - d1["eat_ts"]).dt.total_seconds()
    d1["ate_with_rats"] = (d1["secs_before_rat_left"] > 0).astype(int)

if "sunset_time" in d1.columns and "start_time" in d1.columns:
    d1["hours_after_sunset"] = (d1["start_time"] - d1["sunset_time"]).dt.total_seconds() / 3600

season_col_d1 = col_any(d1, ["season", "SEASON"])
month_col_d1  = col_any(d1, ["month", "MONTH"])
d1["season_"] = d1[season_col_d1].astype("Int64") if season_col_d1 else pd.Series([pd.NA]*len(d1), dtype="Int64")
d1["month_"]  = d1[month_col_d1].astype("Int64") if month_col_d1 else (d1["start_time"].dt.month if "start_time" in d1.columns else pd.Series([pd.NA]*len(d1), dtype="Int64"))
if "start_time" in d1.columns:
    d1["date_"] = d1["start_time"].dt.normalize()

# --------------------------
# Load & preprocess dataset2
# --------------------------
d2 = pd.read_csv(DATA2)
for col in ["start_time", "observation_time", "timestamp"]:
    if col in d2.columns:
        d2[col] = to_dt(d2[col])

season_col_d2 = col_any(d2, ["season", "SEASON"])
month_col_d2  = col_any(d2, ["month", "MONTH"])
d2["season_"] = d2[season_col_d2].astype("Int64") if season_col_d2 else pd.Series([pd.NA]*len(d2), dtype="Int64")
d2["month_"]  = d2[month_col_d2].astype("Int64") if month_col_d2 else pd.Series([pd.NA]*len(d2), dtype="Int64")

if "start_time" in d2.columns:
    d2["date_"] = d2["start_time"].dt.normalize()
elif "observation_time" in d2.columns:
    d2["date_"] = d2["observation_time"].dt.normalize()

if "rat_arrival_number" in d2.columns:
    d2["rat_present"] = (d2["rat_arrival_number"] > 0).astype(int)

# --------------------------
# Quick EDA & proportions
# --------------------------
if "bat_landing_to_food" in d1.columns:
    plt.figure(figsize=(7,5))
    sns.histplot(d1["bat_landing_to_food"], bins=20, kde=True)
    plt.title("Distribution: Bat Landing-to-Food (Dataset 1)")
    plt.xlabel("Seconds")
    savefig("d1_dist_landing_to_food")
    plt.show()

if {"ate_with_rats", "bat_landing_to_food"}.issubset(d1.columns):
    plt.figure(figsize=(7,5))
    sns.boxplot(x="ate_with_rats", y="bat_landing_to_food", data=d1)
    plt.title("Landing-to-Food vs Ate-With-Rats (D1)")
    plt.xlabel("Ate With Rats (0/1)")
    plt.ylabel("Seconds")
    savefig("d1_box_ate_with_rats_vs_landing_to_food")
    plt.show()

if "ate_with_rats" in d1.columns:
    k = int(d1["ate_with_rats"].sum())
    n = int(d1["ate_with_rats"].count())
    bt = binomtest(k, n, p=0.5, alternative="greater")
    ci_lo, ci_hi = wilson_ci(k, n)
    print(f"\nBinomial Test (D1 ate_with_rats): k={k}, n={n}, p̂={k/n:.3f}, p={bt.pvalue:.4f}, 95% CI=({ci_lo:.3f},{ci_hi:.3f})")

# Dataset2 distributions
if "bat_landing_number" in d2.columns:
    plt.figure(figsize=(7,5))
    sns.histplot(d2["bat_landing_number"], bins=20, kde=True)
    plt.title("Distribution: Bat Landing Number (Dataset 2)")
    plt.xlabel("Landings")
    savefig("d2_dist_landing_number")
    plt.show()

# Seasonal/monthly bar
if {"season_", "month_", "bat_landing_number"}.issubset(d2.columns):
    plt.figure(figsize=(9,5))
    sns.barplot(x="month_", y="bat_landing_number", hue="season_", data=d2, ci="sd")
    plt.title("Avg Bat Landings by Month × Season (D2)")
    plt.xlabel("Month")
    plt.ylabel("Avg Bat Landings")
    savefig("d2_bar_month_season_landings")
    plt.show()

# Correlations
corr_cols1 = [c for c in ["bat_landing_to_food", "secs_before_rat_left", "hours_after_sunset", "ate_with_rats"] if c in d1.columns]
if len(corr_cols1) >= 2:
    plt.figure(figsize=(7,5))
    sns.heatmap(d1[corr_cols1].corr(numeric_only=True), annot=True, cmap="vlag")
    plt.title("Correlation Heatmap (Dataset 1)")
    savefig("d1_corr_heatmap")
    plt.show()

corr_cols2 = [c for c in ["bat_landing_number", "food_availability", "rat_present", "hours_after_sunset"] if c in d2.columns]
if len(corr_cols2) >= 2:
    plt.figure(figsize=(7,5))
    sns.heatmap(d2[corr_cols2].corr(numeric_only=True), annot=True, cmap="vlag")
    plt.title("Correlation Heatmap (Dataset 2)")
    savefig("d2_corr_heatmap")
    plt.show()

# --------------------------
# Investigation A — Dataset 1
# --------------------------
print("\n===============================")
print("INVESTIGATION A – Dataset 1")
print("===============================")
XA = [c for c in ["secs_before_rat_left", "hours_after_sunset"] if c in d1.columns]
yA = "bat_landing_to_food"

modelA_ols = None
dfA = d1.dropna(subset=XA + [yA]) if XA and (yA in d1.columns) else pd.DataFrame()
if not dfA.empty:
    # Relationship plots
    if "secs_before_rat_left" in XA:
        plt.figure(figsize=(7,5))
        sns.regplot(data=d1, x="secs_before_rat_left", y="bat_landing_to_food", scatter_kws={"alpha":0.6})
        plt.title("D1: Landing-to-Food ~ Secs Before Rat Left")
        savefig("d1_reg_secs_before_rat_left_vs_landing_to_food")
        plt.show()
    if "hours_after_sunset" in XA:
        plt.figure(figsize=(7,5))
        sns.regplot(data=d1, x="hours_after_sunset", y="bat_landing_to_food", scatter_kws={"alpha":0.6}, color="orange")
        plt.title("D1: Landing-to-Food ~ Hours After Sunset")
        savefig("d1_reg_hours_after_sunset_vs_landing_to_food")
        plt.show()

    Xtr, Xte, ytr, yte = train_test_split(dfA[XA], dfA[yA], test_size=0.25, random_state=42)
    mA = LinearRegression().fit(Xtr, ytr)
    yhat = mA.predict(Xte)
    print_model_sklearn("Investigation A", mA, XA, yte, yhat)

    modelA_ols = print_and_save_ols("Investigation A", dfA[XA], dfA[yA], "investigationA_OLS.txt")
else:
    print("Not enough data for Investigation A.")

# --------------------------
# Investigation B — Dataset 2
# --------------------------
print("\n===============================")
print("INVESTIGATION B – Dataset 2")
print("===============================")
XB = [c for c in ["food_availability", "rat_present", "hours_after_sunset"] if c in d2.columns]
yB = "bat_landing_number"

modelB_ols = None
dfB = d2.dropna(subset=XB + [yB]) if XB and (yB in d2.columns) else pd.DataFrame()
if not dfB.empty:
    # Relation visuals
    if {"food_availability", "bat_landing_number"}.issubset(d2.columns):
        sns.lmplot(x="food_availability", y="bat_landing_number", hue="rat_present", data=d2,
                   height=5, aspect=1.2, scatter_kws={"alpha":0.6})
        plt.title("D2: Landings ~ Food Availability (by Rat Presence)")
        savefig("d2_lm_food_vs_landings_by_rat")
        plt.show()

    Xtr, Xte, ytr, yte = train_test_split(dfB[XB], dfB[yB], test_size=0.25, random_state=42)
    mB = LinearRegression().fit(Xtr, ytr)
    yhat = mB.predict(Xte)
    print_model_sklearn("Investigation B", mB, XB, yte, yhat)

    modelB_ols = print_and_save_ols("Investigation B", dfB[XB], dfB[yB], "investigationB_OLS.txt")
else:
    print("Not enough data for Investigation B.")

# =========================================================
# JOIN: Build a combined dataset (d1 ⨝ d2) for Investigation C
# =========================================================
print("\n===============================")
print("JOIN STEP – Combining dataset1 & dataset2")
print("===============================")

def try_exact_id_join(df1, df2):
    id_candidates = ["event_id", "EventID", "session_id", "record_id", "id", "ID"]
    left_id  = col_any(df1, id_candidates)
    right_id = col_any(df2, id_candidates)
    if left_id and right_id:
        merged = df1.merge(df2, left_on=left_id, right_on=right_id, how="inner", suffixes=("_d1", "_d2"))
        if len(merged):
            print(f"Exact ID join on {left_id} ~ {right_id}: matched {len(merged)} rows.")
            return merged
    return None

def try_exact_time_join(df1, df2):
    if "start_time" in df1.columns and "start_time" in df2.columns:
        merged = df1.merge(df2, on="start_time", how="inner", suffixes=("_d1", "_d2"))
        if len(merged):
            print(f"Exact time join on start_time: matched {len(merged)} rows.")
            return merged
    return None

def try_keyed_date_join(df1, df2):
    keys = [k for k in ["date_", "season_", "month_"] if (k in df1.columns and k in df2.columns)]
    if not keys:
        return None
    site_key = col_any(df1, ["site", "location", "Site", "Location"])
    if site_key and (site_key in df2.columns):
        keys = keys + [site_key]
    merged = df1.merge(df2, on=keys, how="inner", suffixes=("_d1", "_d2"))
    if len(merged):
        print(f"Keyed date join on {keys}: matched {len(merged)} rows.")
        return merged
    return None

def try_nearest_time_join(df1, df2, tolerance="30min"):
    tleft  = col_any(df1, ["start_time", "eat_ts"])
    tright = col_any(df2, ["start_time", "observation_time", "timestamp"])
    if not (tleft and tright):
        return None
    a = df1.sort_values(tleft).copy()
    b = df2.sort_values(tright).copy()
    merged = pd.merge_asof(
        a, b, left_on=tleft, right_on=tright,
        direction="nearest", tolerance=pd.Timedelta(tolerance),
        suffixes=("_d1", "_d2")
    )
    merged = merged.dropna(subset=[tright])  # keep matched
    if len(merged):
        print(f"Nearest-time join on {tleft} ~ {tright} within {tolerance}: matched {len(merged)} rows.")
        return merged
    return None

merged = (
    try_exact_id_join(d1, d2)
    or try_exact_time_join(d1, d2)
    or try_keyed_date_join(d1, d2)
    or try_nearest_time_join(d1, d2, tolerance="30min")
)

if merged is None or merged.empty:
    print("No overlap detected; Investigation C will be skipped.")
else:
    # Harmonize features
    if "rat_present" not in merged.columns:
        if "rat_arrival_number" in merged.columns:
            merged["rat_present"] = (merged["rat_arrival_number"] > 0).astype(int)
        else:
            merged["rat_present"] = 0

    if "hours_after_sunset" not in merged.columns:
        ha_cols = [c for c in merged.columns if c.endswith("hours_after_sunset")]
        if ha_cols:
            merged["hours_after_sunset"] = merged[ha_cols[0]]

    XC_candidates = [
        "food_availability",
        "rat_present",
        "hours_after_sunset",
        "ate_with_rats",
        "secs_before_rat_left",
    ]
    XC = [c for c in XC_candidates if c in merged.columns]

    if "bat_landing_number" in merged.columns:
        yC = "bat_landing_number"
        yC_label = "bat_landing_number"
    elif "bat_landing_to_food" in merged.columns:
        yC = "bat_landing_to_food"
        yC_label = "bat_landing_to_food"
    else:
        yC = None
        yC_label = ""

    print("\nColumns available to Investigation C")
    print("Predictors (XC):", XC)
    print("Outcome (yC):", yC)

    if XC and yC:
        dfC = merged.dropna(subset=XC + [yC])
        if dfC.empty:
            print("Not enough overlapping rows for Investigation C.")
        else:
            print(f"Investigation C dataset size: {len(dfC)} rows")

            # Quick corr heatmap
            if len(XC) >= 2:
                plt.figure(figsize=(7,5))
                sns.heatmap(dfC[XC].corr(numeric_only=True), annot=True, cmap="vlag")
                plt.title("Investigation C — Predictor Correlations")
                savefig("invC_corr_heatmap")
                plt.show()

            # Pairplot (sample if large)
            sampleC = dfC.sample(min(len(dfC), 800), random_state=42)
            num_cols_C = [c for c in sampleC.columns if sampleC[c].dtype in ["float64", "int64"]][:6]
            if len(num_cols_C) >= 3:
                sns.pairplot(sampleC[num_cols_C].dropna(), diag_kind="kde")
                plt.suptitle("Investigation C — Pairwise Relationships", y=1.02)
                savefig("invC_pairplot")
                plt.show()

            # Train/test & scikit-learn LR
            Xtr, Xte, ytr, yte = train_test_split(dfC[XC], dfC[yC], test_size=0.25, random_state=42)
            mC = LinearRegression().fit(Xtr, ytr)
            yhat = mC.predict(Xte)
            print_model_sklearn("Investigation C (Joined)", mC, XC, yte, yhat)

            # Full OLS with statsmodels
            modelC_ols = print_and_save_ols("Investigation C (Joined)", dfC[XC], dfC[yC],
                                            f"investigationC_{yC_label}_OLS.txt")

            # Diagnostics for OLS
            def regression_diagnostics(model, label):
                infl = model.get_influence()
                sm_fr = infl.summary_frame()
                resid = model.resid
                fitted = model.fittedvalues

                # Residual vs Fitted
                plt.figure(figsize=(7,5))
                plt.scatter(fitted, resid, alpha=0.6)
                plt.axhline(0, color="red", linestyle="--")
                plt.title(f"{label} – Residuals vs Fitted")
                plt.xlabel("Fitted")
                plt.ylabel("Residual")
                savefig(f"{label.lower().replace(' ', '_')}_resid_vs_fitted")
                plt.show()

                # QQ Plot
                sm.qqplot(resid, line="45", fit=True)
                plt.title(f"{label} – QQ Plot")
                savefig(f"{label.lower().replace(' ', '_')}_qqplot")
                plt.show()

                # Leverage vs Residuals
                plt.figure(figsize=(7,5))
                plt.scatter(sm_fr["hat_diag"], resid, alpha=0.6)
                plt.title(f"{label} – Leverage vs Residuals")
                plt.xlabel("Leverage (hat)")
                plt.ylabel("Residual")
                savefig(f"{label.lower().replace(' ', '_')}_leverage_residual")
                plt.show()

            # Run diagnostics
            if 'modelA_ols' in globals() and modelA_ols is not None:
                regression_diagnostics(modelA_ols, "Investigation A OLS")
            if 'modelB_ols' in globals() and modelB_ols is not None:
                regression_diagnostics(modelB_ols, "Investigation B OLS")
            if 'modelC_ols' in locals():
                regression_diagnostics(modelC_ols, "Investigation C OLS")

    else:
        print("Investigation C skipped (missing predictors or outcome).")

print("\n Done. OLS summaries in 'model_summaries/' and charts in 'charts/'.")
