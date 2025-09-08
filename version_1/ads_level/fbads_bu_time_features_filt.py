#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Feature engineering for FB Ads (campaign/ad-group time features + efficiencies).

Input  : a parquet with at least these columns:
  ['profile','campaign_objective','creative_call_to_action_type','impression_device',
   'ad_group_start_date','ad_group_end_date','campaign_start_date','campaign_end_date',
   'cost','actions','clicks','impressions','conversion_value','reach']

Output : enriched parquet with:
  - X features: time features, durations, relationships, interactions, categorical norms
  - Y targets: log1p transformed metrics (impressions, clicks, actions, reach, conversion_value)
"""

import argparse
import os
import re
import time
import numpy as np
import polars as pl

# =========================
# Business type mapping (from fbads_prep_polars.py)
# =========================
business_type_mapping = {
    # Shopping Malls & Retail Chains
    'central': 'retail_brand',
    'bebeplay': 'retail_brand',
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


NEED_COLS = [
    "profile",
    "campaign_objective",
    "creative_call_to_action_type",
    "impression_device",
    "ad_group_start_date",
    "ad_group_end_date",
    "campaign_start_date",
    "campaign_end_date",
    "cost",
    "actions",
    "clicks",
    "impressions",
    "conversion_value",
    "reach",  # Added reach for reach_log1p target
]

# Columns to drop after feature engineering
COLS_TO_DROP = [
    'profile',
    'campaign_objective',
    'creative_call_to_action_type',
    'impression_device',
    'ad_group_start_date',
    'ad_group_end_date',
    'campaign_start_date',
    'campaign_end_date',
    'ctr',
    'cvr',
    'cpm',
    'cpc',
    'roas',
    'acos',
    'btype_confidence'
]

# X features to keep
X_FEATURES = [
    'ad_group_duration',
    'campaign_duration',
    'adg_start_month',
    'adg_start_year4',
    'adg_start_dow',
    'adg_start_weekofyear',
    'adg_start_doy',
    'adg_start_days_in_month',
    'adg_start_dom',
    'adg_start_month_progress',
    'adg_start_quarter',
    'adg_start_is_weekend',
    'adg_start_is_month_start',
    'adg_start_is_month_end',
    'adg_start_month_sin',
    'adg_start_month_cos',
    'adg_start_dow_sin',
    'adg_start_dow_cos',
    'adg_start_doy_sin',
    'adg_start_doy_cos',
    'adg_end_month',
    'adg_end_year4',
    'adg_end_dow',
    'adg_end_weekofyear',
    'adg_end_doy',
    'adg_end_days_in_month',
    'adg_end_dom',
    'adg_end_month_progress',
    'adg_end_quarter',
    'adg_end_is_weekend',
    'adg_end_is_month_start',
    'adg_end_is_month_end',
    'adg_end_month_sin',
    'adg_end_month_cos',
    'adg_end_dow_sin',
    'adg_end_dow_cos',
    'adg_end_doy_sin',
    'adg_end_doy_cos',
    'camp_start_month',
    'camp_start_year4',
    'camp_start_dow',
    'camp_start_weekofyear',
    'camp_start_doy',
    'camp_start_days_in_month',
    'camp_start_dom',
    'camp_start_month_progress',
    'camp_start_quarter',
    'camp_start_is_weekend',
    'camp_start_is_month_start',
    'camp_start_is_month_end',
    'camp_start_month_sin',
    'camp_start_month_cos',
    'camp_start_dow_sin',
    'camp_start_dow_cos',
    'camp_start_doy_sin',
    'camp_start_doy_cos',
    'camp_end_month',
    'camp_end_year4',
    'camp_end_dow',
    'camp_end_weekofyear',
    'camp_end_doy',
    'camp_end_days_in_month',
    'camp_end_dom',
    'camp_end_month_progress',
    'camp_end_quarter',
    'camp_end_is_weekend',
    'camp_end_is_month_start',
    'camp_end_is_month_end',
    'camp_end_month_sin',
    'camp_end_month_cos',
    'camp_end_dow_sin',
    'camp_end_dow_cos',
    'camp_end_doy_sin',
    'camp_end_doy_cos',
    'adg_start_minus_camp_start_days',
    'camp_end_minus_adg_end_days',
    'adg_start_offset_days',
    'adg_end_offset_days',
    'adg_inside_campaign',
    'adg_duration_ratio',
    'adg_mid_month_sin',
    'adg_mid_month_cos',
    'camp_mid_month_sin',
    'camp_mid_month_cos',
    'adg_cost_per_day',
    'camp_cost_per_day',
    'cost_x_adg_start_month_sin',
    'cost_x_adg_start_month_cos',
    'cost_x_camp_start_month_sin',
    'cost_x_camp_start_month_cos',
    'campaign_objective_norm',
    'cta_type_norm',
    'impression_device_norm',
    'business_type',
    'cost'
]

# Y target features to keep
Y_TARGETS = [
    'impressions_log1p',
    'clicks_log1p',
    'actions_log1p',
    'reach_log1p',
    'conversion_value_log1p',
]


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


def safe_div(num: pl.Expr, den: pl.Expr) -> pl.Expr:
    return (num.cast(pl.Float64) / pl.max_horizontal(den.cast(pl.Float64), pl.lit(1e-9))).cast(pl.Float64)


def log1p_safe(x: pl.Expr) -> pl.Expr:
    return (pl.max_horizontal(x.cast(pl.Float64), pl.lit(0.0)) + 1.0).log()


# ---- robust trig helpers (work across Polars builds) ----
_HAS_EXPR_SIN = hasattr(pl.Expr, "sin")
_HAS_EXPR_COS = hasattr(pl.Expr, "cos")

def sin_cycle(x: pl.Expr, period: int | float) -> pl.Expr:
    ang = (2.0 * np.pi * x.cast(pl.Float64) / float(period))
    if _HAS_EXPR_SIN:
        return ang.sin()
    return ang.map_elements(np.sin, return_dtype=pl.Float64)

def cos_cycle(x: pl.Expr, period: int | float) -> pl.Expr:
    ang = (2.0 * np.pi * x.cast(pl.Float64) / float(period))
    if _HAS_EXPR_COS:
        return ang.cos()
    return ang.map_elements(np.cos, return_dtype=pl.Float64)


def days_between(end_date: pl.Expr, start_date: pl.Expr) -> pl.Expr:
    # (end - start) in days; negatives→0; null→0; int64
    return (
        pl.when(end_date.is_not_null() & start_date.is_not_null())
          .then((end_date - start_date).dt.total_days().clip(lower_bound=0))
          .otherwise(None)
          .fill_null(0)
          .cast(pl.Int64)
    )


def time_block(prefix: str, d: pl.Expr) -> list[pl.Expr]:
    """Feature block for a Date expression `d` with prefix."""
    dom  = d.dt.day()                   # 1..31
    mon  = d.dt.month()                 # 1..12
    yr4  = d.dt.year()
    dow  = d.dt.weekday()               # 0..6 (Mon=0)
    woy  = d.dt.week()
    doy  = d.dt.ordinal_day()           # 1..365/366
    dim  = d.dt.month_end().dt.day()    # days in month

    return [
        # raw calendar
        mon.cast(pl.Int8).alias(f"{prefix}_month"),
        yr4.cast(pl.Int16).alias(f"{prefix}_year4"),
        dow.cast(pl.Int8).alias(f"{prefix}_dow"),
        woy.cast(pl.Int16).alias(f"{prefix}_weekofyear"),
        doy.cast(pl.Int16).alias(f"{prefix}_doy"),
        dim.cast(pl.Int8).alias(f"{prefix}_days_in_month"),
        dom.cast(pl.Int8).alias(f"{prefix}_dom"),
        (((dom - 1) / dim).cast(pl.Float64)).alias(f"{prefix}_month_progress"),

        # quarter & flags
        (((mon - 1) // 3) + 1).cast(pl.Int8).alias(f"{prefix}_quarter"),
        (dow >= 5).cast(pl.Int8).alias(f"{prefix}_is_weekend"),
        (dom == 1).cast(pl.Int8).alias(f"{prefix}_is_month_start"),
        (dom == dim).cast(pl.Int8).alias(f"{prefix}_is_month_end"),

        # cyclical encodings
        sin_cycle(mon, 12).alias(f"{prefix}_month_sin"),
        cos_cycle(mon, 12).alias(f"{prefix}_month_cos"),
        sin_cycle(dow, 7).alias(f"{prefix}_dow_sin"),
        cos_cycle(dow, 7).alias(f"{prefix}_dow_cos"),
        sin_cycle(doy, 366).alias(f"{prefix}_doy_sin"),
        cos_cycle(doy, 366).alias(f"{prefix}_doy_cos"),
    ]


def norm_text(colname: str, outname: str) -> pl.Expr:
    return (
        pl.col(colname).cast(pl.Utf8)
        .str.to_lowercase()
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
        .fill_null("unknown")
        .alias(outname)
    )


def main(args: argparse.Namespace) -> None:
    t0 = time.time()
    print(f"Polars: {pl.__version__}")
    print(f"Input : {args.input}")
    print(f"Output: {args.output}")
    print(f"Engine: {args.engine} (prefer GPU: {args.engine == 'gpu'})")

    # --- scan & choose available columns (LAZY)
    lf = pl.scan_parquet(args.input)
    schema = lf.collect_schema()
    cols_present = set(schema.names())
    use_cols = [c for c in NEED_COLS if c in cols_present]
    missing  = [c for c in NEED_COLS if c not in cols_present]
    print("Using cols:", use_cols)
    if missing:
        print("Missing   :", missing)

    lf = lf.select([pl.col(c) for c in use_cols])

    # --- numeric coercions (LAZY)
    num_cols = [c for c in ["cost","actions","clicks","impressions","conversion_value","reach"] if c in use_cols]
    if num_cols:
        lf = lf.with_columns([pl.col(c).cast(pl.Float64, strict=False).alias(c) for c in num_cols])

    # --- standardize date columns directly to Date (LAZY)
    def _date_or_null(name: str, alias: str) -> pl.Expr:
        if name in use_cols:
            return pl.col(name).cast(pl.Date).alias(alias)
        return pl.lit(None).cast(pl.Date).alias(alias)

    lf = lf.with_columns([
        _date_or_null("ad_group_start_date", "_adg_start"),
        _date_or_null("ad_group_end_date",   "_adg_end"),
        _date_or_null("campaign_start_date", "_camp_start"),
        _date_or_null("campaign_end_date",   "_camp_end"),
    ])

    # --- durations (LAZY)
    lf = lf.with_columns([
        days_between(pl.col("_adg_end"),  pl.col("_adg_start")).alias("ad_group_duration"),
        days_between(pl.col("_camp_end"), pl.col("_camp_start")).alias("campaign_duration"),
    ])

    # --- rich time features per anchor date + relations (LAZY)
    lf = (
        lf.with_columns(
            time_block("adg_start", pl.col("_adg_start"))
          + time_block("adg_end",   pl.col("_adg_end"))
          + time_block("camp_start",pl.col("_camp_start"))
          + time_block("camp_end",  pl.col("_camp_end"))
        )
        .with_columns([
            # signed offsets
            (pl.col("_adg_start") - pl.col("_camp_start")).dt.total_days().alias("adg_start_minus_camp_start_days"),
            (pl.col("_camp_end")  - pl.col("_adg_end")).dt.total_days().alias("camp_end_minus_adg_end_days"),

            # non-negative offsets
            pl.max_horizontal((pl.col("_adg_start") - pl.col("_camp_start")).dt.total_days(), pl.lit(0)).alias("adg_start_offset_days"),
            pl.max_horizontal((pl.col("_camp_end")  - pl.col("_adg_end")).dt.total_days(),   pl.lit(0)).alias("adg_end_offset_days"),

            # containment + ratios
            ((pl.col("_adg_start") >= pl.col("_camp_start")) & (pl.col("_adg_end") <= pl.col("_camp_end"))).cast(pl.Int8).alias("adg_inside_campaign"),
            (pl.col("ad_group_duration") / pl.max_horizontal(pl.col("campaign_duration"), pl.lit(1))).cast(pl.Float64).alias("adg_duration_ratio"),

            # midpoints
            (pl.col("_adg_start") + pl.duration(days=(pl.col("ad_group_duration") // 2))).alias("_adg_mid"),
            (pl.col("_camp_start") + pl.duration(days=(pl.col("campaign_duration") // 2))).alias("_camp_mid"),
        ])
        .with_columns([
            sin_cycle(pl.col("_adg_mid").dt.month(), 12).alias("adg_mid_month_sin"),
            cos_cycle(pl.col("_adg_mid").dt.month(), 12).alias("adg_mid_month_cos"),
            sin_cycle(pl.col("_camp_mid").dt.month(), 12).alias("camp_mid_month_sin"),
            cos_cycle(pl.col("_camp_mid").dt.month(), 12).alias("camp_mid_month_cos"),
        ])
        .drop(["_adg_mid","_camp_mid"])
    )

    # --- efficiency metrics (will drop most, but needed for interactions) + Y targets (LAZY)
    lf = (
        lf.with_columns([
            # Y targets (log1p transformations)
            log1p_safe(pl.col("cost")).alias("cost_log1p"),
            log1p_safe(pl.col("impressions")).alias("impressions_log1p"),
            log1p_safe(pl.col("clicks")).alias("clicks_log1p"),
            log1p_safe(pl.col("actions")).alias("actions_log1p"),
            log1p_safe(pl.col("conversion_value")).alias("conversion_value_log1p"),
        ] + ([log1p_safe(pl.col("reach")).alias("reach_log1p")] if "reach" in use_cols else []) +
        [
            # Efficiency metrics (most will be dropped)
            safe_div(pl.col("clicks"), pl.col("impressions")).alias("ctr"),
            safe_div(pl.col("actions"), pl.col("clicks")).alias("cvr"),
            safe_div(pl.col("cost"), (pl.col("impressions") / 1000.0)).alias("cpm"),
            safe_div(pl.col("cost"), pl.col("clicks")).alias("cpc"),
            safe_div(pl.col("conversion_value"), pl.col("cost")).alias("roas"),
            safe_div(pl.col("cost"), pl.col("conversion_value")).alias("acos"),

            safe_div(pl.col("cost"), pl.max_horizontal(pl.col("ad_group_duration"),  pl.lit(1))).alias("adg_cost_per_day"),
            safe_div(pl.col("cost"), pl.max_horizontal(pl.col("campaign_duration"), pl.lit(1))).alias("camp_cost_per_day"),
        ])
        .with_columns([
            (pl.col("cost_log1p") * pl.col("adg_start_month_sin")).alias("cost_x_adg_start_month_sin"),
            (pl.col("cost_log1p") * pl.col("adg_start_month_cos")).alias("cost_x_adg_start_month_cos"),
            (pl.col("cost_log1p") * pl.col("camp_start_month_sin")).alias("cost_x_camp_start_month_sin"),
            (pl.col("cost_log1p") * pl.col("camp_start_month_cos")).alias("cost_x_camp_start_month_cos"),
        ])
    )

    # --- normalized categoricals (LAZY)
    cat_norm_exprs = []
    for raw, out in [
        ("campaign_objective", "campaign_objective_norm"),
        ("creative_call_to_action_type", "cta_type_norm"),
        ("impression_device", "impression_device_norm"),
        ("profile", "profile_norm"),
    ]:
        if raw in cols_present:
            cat_norm_exprs.append(norm_text(raw, out))

    if cat_norm_exprs:
        lf = lf.with_columns(cat_norm_exprs)

    # --- business type mapping (LAZY)
    if "profile" in cols_present and len(business_type_mapping) > 0:
        print("Adding business type mapping...")
        lf = lf.with_columns([
            build_business_type_expr(business_type_mapping, "profile_norm").alias("business_type"),
            # Simplified confidence - will be dropped anyway
            pl.lit(1.0).alias("btype_confidence")
        ])
    else:
        lf = lf.with_columns([
            pl.lit("unknown").alias("business_type"),
            pl.lit(0.0).alias("btype_confidence")
        ])

    # --- SINGLE COLLECT with GPU/CPU preference
    print("Collecting with engine preference:", args.engine)
    df = safe_collect(lf, prefer_gpu=(args.engine == "gpu"))
    
    # --- drop internal temp date anchors (post-collect)
    drop_tmp = [c for c in ["_adg_start","_adg_end","_camp_start","_camp_end"] if c in df.columns]
    df = df.drop(drop_tmp)

    # --- Select only X features and Y targets
    all_keep_cols = X_FEATURES + Y_TARGETS
    available_cols = [c for c in all_keep_cols if c in df.columns]
    missing_cols = [c for c in all_keep_cols if c not in df.columns]
    
    if missing_cols:
        print(f"⚠️  Missing expected columns: {missing_cols}")
    
    print(f"Keeping {len(available_cols)} columns: {len(X_FEATURES)} X features + {len(Y_TARGETS)} Y targets")
    df = df.select(available_cols)

    # --- save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    df.write_parquet(args.output, compression="zstd", statistics=True)

    print(f"✅ Done in {time.time()-t0:.1f}s | rows={df.height:,} cols={df.width}")
    print("Saved →", args.output)

    # Show final column summary
    print(f"\nFinal dataset:")
    print(f"- X features: {len([c for c in df.columns if c in X_FEATURES])}")
    print(f"- Y targets: {len([c for c in df.columns if c in Y_TARGETS])}")
    
    # Show business type distribution if available
    if "business_type" in df.columns:
        bt_dist = df.group_by("business_type").len().sort("len", descending=True)
        print(f"\nBusiness type distribution:")
        print(bt_dist)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Feature engineering for FB Ads with GPU/CPU support")
    ap.add_argument("--input",  required=True, help="Path to input parquet")
    ap.add_argument("--output", required=True, help="Path to output parquet")
    ap.add_argument("--engine", choices=["gpu", "cpu"], default="gpu", 
                    help="Prefer GPU collect if available (default: gpu)")
    args = ap.parse_args()
    main(args)