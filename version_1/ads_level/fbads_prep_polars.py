#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
fbads_prep_polars.py
--------------------
Prepare Facebook Ads campaign-level table with Polars (GPU-first, CPU fallback).

Pipeline (RAM-friendly):
1) Lazy scan -> tolerant rename
2) Deduplicate on ad/adset/date/metric key (keep date strings)
3) Aggregate to Campaign (numeric metrics)
4) Aggregate campaign_start/end from lf_dedup
5) Collect (GPU if available), compute campaign_duration from campaign_start/end (Date arithmetic)
6) Map business_type at FINAL step (vectorized str.contains; no explode) and add btype_confidence
7) Drop ['campaign_id','campaign_start_str','campaign_end_str'] and save parquet
"""

import os, re, time, argparse, sys, traceback
import polars as pl

# =========================
# YOUR business_type mapping
# =========================
business_type_mapping = {
    # Shopping Malls & Retail Chains
    'long lasting': 'retail_brand',
    'nitori': 'retail_brand',
    'zhulian': 'retail_brand',
    'ielleair': 'retail_brand',
    'vetz petz': 'retail_brand',
    'kanekoji': 'retail_brand',
    'kamedis': 'retail_brand',

    # Real Estate & Property Development
    'sena development': 'real_estate',
    'sena developmant': 'real_estate',
    'onerealestate': 'real_estate',
    'jsp property': 'real_estate',
    'jsp property x sena': 'real_estate',
    'nusasiri': 'real_estate',
    'premium place': 'real_estate',
    'urban': 'real_estate',
    'pieamsuk': 'real_estate',
    'asakan': 'real_estate',
    'cpn': 'real_estate',
    'wt land development (2024)': 'real_estate',
    'property client (ex.the fine)': 'real_estate',
    'the fine': 'real_estate',
    'colour development': 'real_estate',
    'banmae villa': 'real_estate',
    'varintorn l as vibhavadi l brand': 'real_estate',
    'inspired': 'real_estate',
    'goldenduck': 'real_estate',
    'chewa x c - pk': 'real_estate',

    # Fashion & Lifestyle
    'samsonite': 'fashion_lifestyle',
    'do day dream': 'fashion_lifestyle',
    'ido day dream': 'fashion_lifestyle',
    'ido day dream i valeraswiss thailand i brand [2025]': 'fashion_lifestyle',
    'fila': 'fashion_lifestyle',
    'playboy': 'fashion_lifestyle',
    'what a girl want': 'fashion_lifestyle',
    'rich sport': 'fashion_lifestyle',
    'heydude': 'fashion_lifestyle',

    # Beauty & Cosmetics
    'bb care': 'beauty_cosmetics',
    'reuse': 'beauty_cosmetics',
    'dedvelvet': 'beauty_cosmetics',
    'riobeauty': 'beauty_cosmetics',
    'kameko': 'beauty_cosmetics',
    'befita': 'beauty_cosmetics',
    'vitablend': 'beauty_cosmetics',

    # Healthcare & Medical
    'abl clinic': 'healthcare_medical',
    'luxury clinic': 'healthcare_medical',
    'mane clinic': 'healthcare_medical',
    'dentalme clinic': 'healthcare_medical',
    'mild clinic': 'healthcare_medical',
    'aestheta wellness': 'healthcare_medical',
    'luxury club skin': 'healthcare_medical',

    # Technology & Electronics
    'kangyonglaundry': 'technology_electronics',
    'bosch': 'technology_electronics',
    'amazfit': 'technology_electronics',
    'panduit': 'technology_electronics',
    'mitsubishi electric x digimusketeers': 'technology_electronics',
    'asiasoft digital marketing (center)': 'technology_electronics',
    'at home thailand': 'technology_electronics',
    'sinthanee group': 'technology_electronics',
    'noventiq th': 'technology_electronics',
    'blaupunk l blaupunk l brand': 'technology_electronics',
    'yip in tsoi': 'technology_electronics',

    # Digital Marketing & Agencies
    'digimusketeers': 'digital_marketing',
    'set x digimusketeers': 'digital_marketing',
    'we are innosense co., ltd. v.2': 'digital_marketing',

    # Software Development
    'dspace': 'software_development',
    'midas': 'software_development',
    'launch platform': 'software_development',

    # Financial Services
    'cimb': 'financial_services',
    'tisco ppk': 'financial_services',
    'tisco - insure': 'financial_services',
    'gsb society': 'financial_services',
    'aslan investor': 'financial_services',
    'aeon': 'financial_services',
    'proprakan': 'financial_services',

    # Entertainment & Media
    'donut bangkok': 'entertainment_media',
    'i have ticket': 'entertainment_media',
    'ondemand l ondemand l brand': 'entertainment_media',

    # Food & Beverage
    'ramendesu': 'food_beverage',
    'nomimashou': 'food_beverage',
    'oakberry': 'food_beverage',

    # Transportation & Logistics
    'paypoint': 'transportation_logistics',
    'asia cab': 'transportation_logistics',
    'uac': 'transportation_logistics',
    'artralux': 'transportation_logistics',
    'artralux (social media project)': 'transportation_logistics',
    'siamwatercraft': 'transportation_logistics',

    # Pharmaceuticals & Health Products
    'inpac pharma': 'pharmaceuticals',

    # Non-Profit & Organizations
    'unhcr': 'non_profit',

    # Construction & Manufacturing
    'arun plus ptt': 'industrial_manufacturing',
    'scg': 'industrial_manufacturing',

    # Others/Uncategorized
    'free 657,00 thb': 'other',
}

# ------------------ column candidates (tolerant rename) ------------------
CANDIDATES = {
    # IDs
    "campaign_id":  ["campaign_id","campaignid","campaignID"],
    "ad_group_id":  ["ad_group_id","adgroup_id","adset_id","adsetid"],

    # Row-level/ad-group dates (not for duration, only for dedupe key)
    "date_start":   ["ad_group_start_date","adgroup_start_date","date_start","start_time","ad_delivery_start_time"],
    "date_stop":    ["ad_group_end_date","adgroup_end_date","date_stop","end_date","ad_delivery_stop_time"],

    # Campaign-level dates (used ONLY for duration)
    "campaign_start": ["campaign_start_date","campaign_start","campaign_start_time"],
    "campaign_end":   ["campaign_end_date","campaign_end","campaign_end_time"],

    # Metrics
    "impressions":  ["impressions"],
    "reach":        ["reach","unique_impressions"],
    "clicks":       ["clicks","link_clicks"],
    "actions":      ["actions","conversions","total_actions"],
    "conversion_value": ["conversion_value","value","purchase_value"],
    "cost":         ["cost","spend","amount_spent","total_spent"],

    # Profile (optional, for business type mapping)
    "profile":      ["profile","page_name","account_name","advertiser_name"],
}

# ------------------ utilities ------------------
def std(s: str) -> str:
    s = re.sub(r"\s+", "_", s.strip().lower())
    s = re.sub(r"[^\w]+", "_", s)
    return re.sub(r"_+", "_", s).strip("_")

def build_business_type_expr(mapping: dict, col: str = "profile_norm") -> pl.Expr:
    """Vectorized contains-based mapper (no explode)."""
    expr = pl.lit(None, dtype=pl.Utf8)
    for k, v in list(mapping.items())[::-1]:
        pat = rf"\b{re.escape(k.lower().strip())}\b"
        expr = pl.when(pl.col(col).str.contains(pat)).then(pl.lit(v)).otherwise(expr)
    return expr.fill_null("unknown")

def safe_collect(lf: pl.LazyFrame, prefer_gpu: bool = True) -> pl.DataFrame:
    """Collect with GPU if possible, else CPU (rust)."""
    try:
        if prefer_gpu:
            return lf.collect(engine="gpu")
    except Exception:
        pass
    return lf.collect(engine="rust")

# ------------------ main ------------------
def main(args):
    t0 = time.time()
    if not os.path.exists(args.input):
        raise FileNotFoundError(args.input)

    print("Polars:", pl.__version__)
    print("Input :", args.input)
    print("Output:", args.output)
    print("Prefer GPU:", args.engine == "gpu")

    # 1) scan + tolerant rename (lazy)
    lf_raw = pl.scan_parquet(args.input)
    schema = lf_raw.collect_schema()
    schema_norm = {name: std(name) for name in schema.names()}

    rename_map = {}
    for canon, opts in CANDIDATES.items():
        for o in opts:
            so = std(o)
            match = [actual for actual, nm in schema_norm.items() if nm == so]
            if match:
                rename_map[match[0]] = canon
                break

    for req in ["campaign_id", "ad_group_id"]:
        if req not in rename_map.values():
            raise RuntimeError(f"Required column '{req}' not found in input.")

    keep_cols = [c for c in [
        "campaign_id","ad_group_id",
        "date_start","date_stop",
        "campaign_start","campaign_end",
        "impressions","reach","clicks","actions","conversion_value","cost",
        "profile"
    ] if c in rename_map.values()]

    lf = lf_raw.rename(rename_map).select([pl.col(c) for c in keep_cols])

    # 2) deduplicate (on 10-field row key)
    dedup_subset = [c for c in [
        "campaign_id","ad_group_id","date_start","date_stop",
        "impressions","reach","clicks","actions","conversion_value","cost"
    ] if c in keep_cols]
    rows_before = safe_collect(lf.select(pl.len().alias("rows")),
                               prefer_gpu=(args.engine=="gpu")).item()
    lf_dedup = lf.unique(subset=dedup_subset, keep="first")
    rows_after  = safe_collect(lf_dedup.select(pl.len().alias("rows")),
                               prefer_gpu=(args.engine=="gpu")).item()
    print(f"Dedup → before {rows_before:,} | after {rows_after:,} | removed {rows_before-rows_after:,}")

    # 3) campaign numeric metrics (lazy)
    num_aggs = [pl.col("ad_group_id").n_unique().alias("num_ads")]
    for col in ["cost","impressions","reach","clicks","actions","conversion_value"]:
        if col in keep_cols:
            num_aggs.append(pl.col(col).sum())
    campaign_metrics_lf = lf_dedup.group_by("campaign_id").agg(num_aggs)

    # 4) campaign start/end strings/dates from lf_dedup (lazy)
    have_start = "campaign_start" in keep_cols
    have_end   = "campaign_end"   in keep_cols
    if have_start or have_end:
        date_aggs = []
        if have_start:
            date_aggs.append(pl.col("campaign_start").min().alias("campaign_start_str"))
        if have_end:
            date_aggs.append(pl.col("campaign_end").max().alias("campaign_end_str"))
        campaign_dates_lf = lf_dedup.group_by("campaign_id").agg(date_aggs)
    else:
        campaign_dates_lf = pl.LazyFrame({"campaign_id": pl.Series([], dtype=pl.Utf8)})

    # 5) business_type mapping (final step; lazy)
    if "profile" in keep_cols and len(business_type_mapping) > 0:
        prof_counts_lf = (
            lf_dedup
            .with_columns(pl.col("profile").cast(pl.Utf8)
                          .str.to_lowercase()
                          .str.replace_all(r"\s+", " ")
                          .alias("profile_norm"))
            .group_by(["campaign_id","profile_norm"]).len().rename({"len":"pf_count"})
        )
        prof_max_lf = prof_counts_lf.group_by("campaign_id") \
                                    .agg(pl.col("pf_count").max().alias("pf_count_max"))
        prof_top_lf = (
            prof_counts_lf.join(prof_max_lf, on="campaign_id", how="inner")
                          .filter(pl.col("pf_count") == pl.col("pf_count_max"))
                          .group_by("campaign_id")
                          .agg(pl.col("profile_norm").min().alias("profile_norm"))
        )
        bt_per_campaign_lf = prof_top_lf.with_columns(
            build_business_type_expr(business_type_mapping, "profile_norm").alias("business_type")
        ).select(["campaign_id","business_type"])

        # confidence
        lf_bt_rows_lf = (
            lf_dedup
            .with_columns(pl.col("profile").cast(pl.Utf8)
                          .str.to_lowercase()
                          .str.replace_all(r"\s+", " ")
                          .alias("profile_norm"))
            .with_columns(build_business_type_expr(business_type_mapping, "profile_norm")
                          .alias("business_type_row"))
            .group_by(["campaign_id","business_type_row"]).len().rename({"len":"bt_count"})
        )
        bt_totals_lf = lf_dedup.group_by("campaign_id").len().rename({"len":"total_rows"})

        bt_conf_lf = (
            bt_per_campaign_lf
            .join(lf_bt_rows_lf,
                  left_on=["campaign_id","business_type"],
                  right_on=["campaign_id","business_type_row"], how="left")
            .join(bt_totals_lf, on="campaign_id", how="left")
            .with_columns(
                (pl.col("bt_count") / pl.col("total_rows"))
                .cast(pl.Float64).fill_null(0.0).alias("btype_confidence")
            )
            .select(["campaign_id","business_type","btype_confidence"])
        )
    else:
        bt_conf_lf = pl.LazyFrame({
            "campaign_id": pl.Series([], dtype=pl.Utf8),
            "business_type": pl.Series([], dtype=pl.Utf8),
            "btype_confidence": pl.Series([], dtype=pl.Float64),
        })

    # 6) assemble base LF and collect (no duration yet)
    campaign_base_lf = campaign_metrics_lf.join(bt_conf_lf, on="campaign_id", how="left")
    if "campaign_start_str" in campaign_dates_lf.collect_schema().names() or \
       "campaign_end_str"   in campaign_dates_lf.collect_schema().names():
        campaign_base_lf = campaign_base_lf.join(campaign_dates_lf, on="campaign_id", how="left")

    campaign_df = safe_collect(
        campaign_base_lf.with_columns([
            pl.col("num_ads").fill_null(0).cast(pl.Int64),
            pl.when(pl.col("cost").is_not_null())
              .then(pl.col("cost").cast(pl.Float64))
              .otherwise(pl.lit(0.0)).alias("cost"),
            pl.col("business_type").fill_null("unknown").cast(pl.Utf8),
            pl.col("btype_confidence").fill_null(0.0).cast(pl.Float64),
        ]),
        prefer_gpu=(args.engine == "gpu")
    )

    # 7) compute campaign_duration from campaign_start_str / campaign_end_str ONLY (post-collect)
    has_dates = {"campaign_start_str","campaign_end_str"}.issubset(set(campaign_df.columns))
    if has_dates:
        start_dt, end_dt = "campaign_start_str", "campaign_end_str"
        start_ty = campaign_df.schema.get(start_dt)
        end_ty   = campaign_df.schema.get(end_dt)

        if start_ty in (pl.Date, pl.Datetime) and end_ty in (pl.Date, pl.Datetime):
            campaign_df = (
                campaign_df
                .with_columns(
                    (pl.col(end_dt).cast(pl.Date) - pl.col(start_dt).cast(pl.Date))
                    .dt.total_days()
                    .clip(0, None)
                    .cast(pl.Int64)
                    .alias("campaign_duration")
                )
            )
        else:
            campaign_df = (
                campaign_df
                .with_columns([
                    pl.col(start_dt).cast(pl.Utf8).str.strptime(pl.Datetime, strict=False, exact=False).alias("_start_dt"),
                    pl.col(end_dt).cast(pl.Utf8).str.strptime(pl.Datetime, strict=False, exact=False).alias("_end_dt"),
                ])
                .with_columns(
                    ((pl.col("_end_dt") - pl.col("_start_dt")).dt.total_days())
                    .fill_null(0).clip(0, None).cast(pl.Int64).alias("campaign_duration")
                )
                .drop(["_start_dt","_end_dt"])
            )
    else:
        campaign_df = campaign_df.with_columns(pl.lit(0).cast(pl.Int64).alias("campaign_duration"))

    # 8) drop requested cols, order & save
    to_drop = ["campaign_id", "campaign_start_str", "campaign_end_str"]
    drop_now = [c for c in to_drop if c in campaign_df.columns]
    campaign_df = campaign_df.drop(drop_now)

    want_first = [
        "campaign_duration","num_ads","cost",
        "impressions","reach","clicks","actions","conversion_value",
        "business_type","btype_confidence",
    ]
    cols = [c for c in want_first if c in campaign_df.columns] + \
           [c for c in campaign_df.columns if c not in want_first]
    campaign_df = campaign_df.select(cols)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    campaign_df.write_parquet(args.output, compression="zstd", statistics=True)

    print(f"\n✅ Done in {time.time()-t0:.1f}s | rows: {campaign_df.height:,} | cols: {campaign_df.width}")
    print("Dropped columns:", drop_now)
    print("Saved →", args.output)
    print("\nPreview:\n", campaign_df.head(10))

# ------------------ CLI ------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Prep FB Ads campaign-level table with Polars (GPU-first).")
    ap.add_argument("--input",  "-i", default="/content/facebook_ads__20210101_20250630.parquet",
                    help="Path to merged Parquet (ad/adset-level).")
    ap.add_argument("--output", "-o", default="/content/campaign_level__final.parquet",
                    help="Destination Parquet path.")
    ap.add_argument("--engine", choices=["gpu","cpu"], default="gpu",
                    help="Prefer GPU collect if available.")
    args = ap.parse_args()
    try:
        main(args)
    except Exception as e:
        print("\n[x] ERROR:", type(e).__name__, str(e))
        traceback.print_exc()
        sys.exit(1)
