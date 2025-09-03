#!/usr/bin/env python3
import os, argparse, math
import polars as pl
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt

# ----------------- helpers -----------------
def log1p_series(x: pd.Series) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce")
    x = x.mask(x < 0, np.nan)
    return np.log1p(x)

def make_feature_matrix(df_in: pd.DataFrame, num_cols, log_cols):
    cols = [c for c in num_cols if c in df_in.columns]
    log_cols = [c for c in log_cols if c in cols]
    lin_cols = [c for c in cols if c not in log_cols]

    X_parts, feat_names = [], []
    for c in log_cols:
        X_parts.append(log1p_series(df_in[c]))
        feat_names.append(f"{c}_log1p")
    for c in lin_cols:
        col = pd.to_numeric(df_in[c], errors="coerce")
        col = col.mask(col < 0, np.nan)
        X_parts.append(col)
        feat_names.append(c)

    X = np.column_stack([s.values for s in X_parts]) if X_parts else np.empty((len(df_in), 0))
    for j in range(X.shape[1]):
        col = X[:, j]
        m = np.nanmedian(col)
        if not np.isfinite(m): m = 0.0
        col[np.isnan(col)] = m
        X[:, j] = col

    X_scaled = RobustScaler().fit_transform(X) if X.shape[1] else X
    return X_scaled, feat_names

def run_iso_forest(X_scaled, contamination="auto", random_state=42):
    iso = IsolationForest(
        n_estimators=400,
        max_samples="auto",
        contamination=contamination,     # e.g. 0.01 for ~1% outliers
        random_state=random_state,
        n_jobs=-1,
        bootstrap=False,
        verbose=0,
    )
    iso.fit(X_scaled)
    pred = iso.predict(X_scaled)                 # -1 = outlier
    scores = iso.decision_function(X_scaled)     # higher = more normal
    return (pred == -1), scores, iso

def plot_hist(series, title, outpath, bins=60, log1p=False, sample_n=None, seed=42):
    vals = pd.to_numeric(series, errors="coerce").dropna().astype(float)
    if sample_n and len(vals) > sample_n:
        vals = vals.sample(sample_n, random_state=seed)
    if log1p:
        vals = np.log1p(vals)
        title = f"{title} (log1p)"
    plt.figure(figsize=(6,4))
    plt.hist(vals, bins=bins)
    plt.title(title); plt.xlabel("value"); plt.ylabel("count")
    plt.tight_layout(); plt.savefig(outpath, dpi=120); plt.close()

def plot_distributions(df_pd, tag, out_dir, num_candidates, log_cols, bins, sample_n, seed):
    os.makedirs(out_dir, exist_ok=True)
    for c in [c for c in num_candidates if c in df_pd.columns]:
        use_log = c in log_cols
        plot_hist(
            df_pd[c], f"[{tag}] {c}",
            os.path.join(out_dir, f"{tag}__{c}.png"),
            bins=bins, log1p=use_log, sample_n=sample_n, seed=seed
        )

# ----------------- main -----------------
def main():
    ap = argparse.ArgumentParser(description="Distribution plots + two-pass IsolationForest (group-wise, then global).")
    ap.add_argument("--input", required=True, help="Input parquet path")
    ap.add_argument("--out-gw", default="/content/campaign_level__final__no_outliers__groupwise.parquet")
    ap.add_argument("--out-final", default="/content/campaign_level__final__no_outliers__gw_then_global.parquet")
    ap.add_argument("--plot-dir", default="/content/plots_isoforest")
    ap.add_argument("--mask-prefix", default="/content/isoforest_mask", help="Prefix for CSV masks")
    ap.add_argument("--engine", choices=["cpu","gpu"], default="cpu", help="Polars collect engine (affects read only)")
    ap.add_argument("--min-group", type=int, default=150)
    ap.add_argument("--contamination", default="auto", help="'auto' or float like 0.01")
    ap.add_argument("--plot-sample", type=int, default=200000, help="Max rows to sample for plotting (None = all)")
    ap.add_argument("--bins", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)

    # numeric candidates (you can override via env if needed)
    ap.add_argument("--num-cols", nargs="*", default=[
        "campaign_duration", "num_ads", "cost",
        "impressions", "reach", "clicks", "actions", "conversion_value",
        "btype_confidence",
    ])
    ap.add_argument("--log-cols", nargs="*", default=[
        "cost","impressions","reach","clicks","actions","conversion_value"
    ])
    args = ap.parse_args()

    # parse contamination
    cont = args.contamination
    try:
        if isinstance(cont, str) and cont != "auto":
            cont = float(cont)
    except Exception:
        raise ValueError("--contamination must be 'auto' or a float, e.g. 0.01")

    # derive mask paths
    MASK_GW     = f"{args.mask_prefix}_groupwise.csv"
    MASK_GLOBAL = f"{args.mask_prefix}_global_after_groupwise.csv"
    MASK_COMBO  = f"{args.mask_prefix}_combined.csv"

    print("Polars:", pl.__version__)
    print("Config:", vars(args))

    # Read with LazyFrame -> collect to honor GPU engine if requested
    lf = pl.scan_parquet(args.input)
    if args.engine == "gpu":
        df_pl = lf.collect(engine="gpu")
    else:
        df_pl = lf.collect()
    df = df_pl.to_pandas()
    df.insert(0, "row_index", np.arange(len(df), dtype=int))
    print("Loaded:", df.shape, list(df.columns)[:12], "…")
    assert "business_type" in df.columns, "business_type missing (needed for group-wise pass)."

    # -------- BEFORE plots --------
    plot_distributions(df, "before", args.plot_dir, args.num_cols, args.log_cols, args.bins, args.plot_sample, args.seed)

    # -------- Group-wise IF --------
    rng = np.random.default_rng(args.seed)
    mask_gw = np.zeros(len(df), dtype=bool)
    score_gw = np.full(len(df), np.nan, dtype=float)
    summary_rows = []

    for bt, g in df.groupby("business_type", dropna=False):
        idx = g.index.values
        n = len(g)
        if n < args.min_group:
            summary_rows.append({"business_type": bt, "size": n, "outliers": 0, "pct": 0.0, "note": "skipped_too_small"})
            continue

        Xg, feats = make_feature_matrix(g, args.num_cols, args.log_cols)
        if Xg.shape[1] == 0:
            summary_rows.append({"business_type": bt, "size": n, "outliers": 0, "pct": 0.0, "note": "no_numeric_features"})
            continue

        out_g, sc_g, _ = run_iso_forest(Xg, contamination=cont, random_state=args.seed)
        mask_gw[idx] = out_g
        score_gw[idx] = sc_g
        summary_rows.append({
            "business_type": bt, "size": n,
            "outliers": int(out_g.sum()), "pct": 100*out_g.mean(), "note": ""
        })

    gw_summary = pd.DataFrame(summary_rows).sort_values("size", ascending=False)
    print("\nGroup-wise summary:\n", gw_summary.head(10))
    gw_summary.to_csv(os.path.join(args.plot_dir, "groupwise_summary.csv"), index=False)

    df["is_outlier_gw"] = mask_gw
    df["anomaly_score_gw"] = score_gw
    print(f"Group-wise flagged: {mask_gw.sum()} / {len(df)} ({100*mask_gw.mean():.2f}%)")

    pd.DataFrame({
        "row_index": df["row_index"],
        "business_type": df["business_type"],
        "is_outlier_gw": df["is_outlier_gw"],
        "anomaly_score_gw": df["anomaly_score_gw"],
    }).to_csv(MASK_GW, index=False)
    print("Saved →", MASK_GW)

    df_gw_clean = df.loc[~mask_gw].copy()
    print("After group-wise clean:", df_gw_clean.shape)
    pl.DataFrame(df_gw_clean.drop(columns=["is_outlier_gw","anomaly_score_gw"])).write_parquet(
        args.out_gw, compression="zstd", statistics=True
    )
    print("Saved group-wise cleaned parquet →", args.out_gw)

    # -------- AFTER group-wise plots --------
    plot_distributions(df_gw_clean, "after_groupwise", args.plot_dir, args.num_cols, args.log_cols, args.bins, args.plot_sample, args.seed)

    # -------- Global IF on group-cleaned --------
    X2, feats2 = make_feature_matrix(df_gw_clean, args.num_cols, args.log_cols)
    print("\nGlobal pass features:", len(feats2), feats2[:8], "…")

    out2, sc2, _ = run_iso_forest(X2, contamination=cont, random_state=args.seed)
    df_gw_clean["is_outlier_global"] = out2
    df_gw_clean["anomaly_score_global"] = sc2
    print(f"Global pass flagged: {out2.sum()} / {len(df_gw_clean)} ({100*out2.mean():.2f}%)")

    pd.DataFrame({
        "row_index": df_gw_clean["row_index"],
        "is_outlier_global": df_gw_clean["is_outlier_global"],
        "anomaly_score_global": df_gw_clean["anomaly_score_global"],
    }).to_csv(MASK_GLOBAL, index=False)
    print("Saved →", MASK_GLOBAL)

    df_final = df_gw_clean.loc[~df_gw_clean["is_outlier_global"]].drop(columns=["is_outlier_global","anomaly_score_global"])
    print("Final cleaned shape:", df_final.shape)

    # Combined mask
    m_combined = pd.DataFrame({"row_index": df["row_index"]})
    m_combined = m_combined.merge(pd.read_csv(MASK_GW), on="row_index", how="left")
    m_combined = m_combined.merge(pd.read_csv(MASK_GLOBAL), on="row_index", how="left")
    m_combined["is_outlier_combined"] = (
        m_combined["is_outlier_gw"].fillna(False) | m_combined["is_outlier_global"].fillna(False)
    )
    m_combined.to_csv(MASK_COMBO, index=False)
    print("Saved →", MASK_COMBO)

    pl.DataFrame(df_final).write_parquet(args.out_final, compression="zstd", statistics=True)
    print("Saved final parquet →", args.out_final)

    # -------- AFTER final plots --------
    plot_distributions(df_final, "after_final", args.plot_dir, args.num_cols, args.log_cols, args.bins, args.plot_sample, args.seed)

    # anomaly score hist
    sc = pd.to_numeric(df_gw_clean["anomaly_score_global"], errors="coerce").dropna()
    if len(sc) > 0:
        vals = sc
        if args.plot_sample and len(vals) > args.plot_sample:
            vals = vals.sample(args.plot_sample, random_state=args.seed)
        plt.figure(figsize=(6,4))
        plt.hist(vals, bins=args.bins)
        plt.title("IsolationForest anomaly_score (higher = more normal)")
        plt.xlabel("score"); plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(os.path.join(args.plot_dir, "global_anomaly_score_hist.png"), dpi=120)
        plt.close()

if __name__ == "__main__":
    main()
