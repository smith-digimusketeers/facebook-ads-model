#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Facebook Ads Performance Prediction - Complete Inference Script
================================================================

Takes user inputs and predicts FB ads performance using trained ensemble models.
Allows separate specification of standardization parameters file location.

CRITICAL: Models expect standardized features. This script loads and applies
the same standardization parameters used during training.
"""

import argparse
import json
import os
import pickle
import re
import warnings
from datetime import datetime, date, timedelta
from typing import Dict, List, Tuple, Optional, Union

import numpy as np
import pandas as pd
import joblib

# Suppress the feature name warnings
warnings.filterwarnings('ignore', message='X does not have valid feature names')

# =========================
# Business type mapping (same as training)
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

# Expected feature names (160 features total)
EXPECTED_FEATURES = [
    'ad_group_duration', 'campaign_duration', 'adg_start_month', 'adg_start_year4',
    'adg_start_dow', 'adg_start_weekofyear', 'adg_start_doy', 'adg_start_days_in_month',
    'adg_start_dom', 'adg_start_month_progress', 'adg_start_quarter', 'adg_start_is_weekend',
    'adg_start_is_month_start', 'adg_start_is_month_end', 'adg_start_month_sin',
    'adg_start_month_cos', 'adg_start_dow_sin', 'adg_start_dow_cos', 'adg_start_doy_sin',
    'adg_start_doy_cos', 'adg_end_month', 'adg_end_year4', 'adg_end_dow',
    'adg_end_weekofyear', 'adg_end_doy', 'adg_end_days_in_month', 'adg_end_dom',
    'adg_end_month_progress', 'adg_end_quarter', 'adg_end_is_weekend', 'adg_end_is_month_start',
    'adg_end_is_month_end', 'adg_end_month_sin', 'adg_end_month_cos', 'adg_end_dow_sin',
    'adg_end_dow_cos', 'adg_end_doy_sin', 'adg_end_doy_cos', 'camp_start_month',
    'camp_start_year4', 'camp_start_dow', 'camp_start_weekofyear', 'camp_start_doy',
    'camp_start_days_in_month', 'camp_start_dom', 'camp_start_month_progress',
    'camp_start_quarter', 'camp_start_is_weekend', 'camp_start_is_month_start',
    'camp_start_is_month_end', 'camp_start_month_sin', 'camp_start_month_cos',
    'camp_start_dow_sin', 'camp_start_dow_cos', 'camp_start_doy_sin', 'camp_start_doy_cos',
    'camp_end_month', 'camp_end_year4', 'camp_end_dow', 'camp_end_weekofyear',
    'camp_end_doy', 'camp_end_days_in_month', 'camp_end_dom', 'camp_end_month_progress',
    'camp_end_quarter', 'camp_end_is_weekend', 'camp_end_is_month_start', 'camp_end_is_month_end',
    'camp_end_month_sin', 'camp_end_month_cos', 'camp_end_dow_sin', 'camp_end_dow_cos',
    'camp_end_doy_sin', 'camp_end_doy_cos', 'adg_start_minus_camp_start_days',
    'camp_end_minus_adg_end_days', 'adg_start_offset_days', 'adg_end_offset_days',
    'adg_inside_campaign', 'adg_duration_ratio', 'adg_mid_month_sin', 'adg_mid_month_cos',
    'camp_mid_month_sin', 'camp_mid_month_cos', 'adg_cost_per_day', 'camp_cost_per_day',
    'cost_x_adg_start_month_sin', 'cost_x_adg_start_month_cos', 'cost_x_camp_start_month_sin',
    'cost_x_camp_start_month_cos', 'cost',
    # One-hot encoded features (campaign_objective_norm)
    'campaign_objective_norm_app_installs', 'campaign_objective_norm_brand_awareness',
    'campaign_objective_norm_conversions', 'campaign_objective_norm_event_responses',
    'campaign_objective_norm_lead_generation', 'campaign_objective_norm_link_clicks',
    'campaign_objective_norm_messages', 'campaign_objective_norm_outcome_app_promotion',
    'campaign_objective_norm_outcome_awareness', 'campaign_objective_norm_outcome_engagement',
    'campaign_objective_norm_outcome_leads', 'campaign_objective_norm_outcome_sales',
    'campaign_objective_norm_outcome_traffic', 'campaign_objective_norm_page_likes',
    'campaign_objective_norm_post_engagement', 'campaign_objective_norm_product_catalog_sales',
    'campaign_objective_norm_reach', 'campaign_objective_norm_store_visits',
    'campaign_objective_norm_video_views',
    # One-hot encoded features (cta_type_norm)
    'cta_type_norm_', 'cta_type_norm_apply_now', 'cta_type_norm_book_travel',
    'cta_type_norm_buy_tickets', 'cta_type_norm_call_now', 'cta_type_norm_contact_us',
    'cta_type_norm_download', 'cta_type_norm_event_rsvp', 'cta_type_norm_get_directions',
    'cta_type_norm_get_offer', 'cta_type_norm_get_offer_view', 'cta_type_norm_get_promotions',
    'cta_type_norm_instagram_message', 'cta_type_norm_install_mobile_app',
    'cta_type_norm_learn_more', 'cta_type_norm_like_page', 'cta_type_norm_listen_now',
    'cta_type_norm_message_page', 'cta_type_norm_no_button', 'cta_type_norm_order_now',
    'cta_type_norm_play_game', 'cta_type_norm_shop_now', 'cta_type_norm_sign_up',
    'cta_type_norm_subscribe', 'cta_type_norm_view_instagram_profile', 'cta_type_norm_watch_more',
    'cta_type_norm_whatsapp_message',
    # One-hot encoded features (impression_device_norm)
    'impression_device_norm_', 'impression_device_norm_android_smartphone',
    'impression_device_norm_android_tablet', 'impression_device_norm_desktop',
    'impression_device_norm_ipad', 'impression_device_norm_iphone', 'impression_device_norm_ipod',
    'impression_device_norm_other',
    # One-hot encoded features (business_type)
    'business_type_beauty_cosmetics', 'business_type_digital_marketing',
    'business_type_entertainment_media', 'business_type_fashion_lifestyle',
    'business_type_financial_services', 'business_type_food_beverage',
    'business_type_healthcare_medical', 'business_type_industrial_manufacturing',
    'business_type_other', 'business_type_real_estate', 'business_type_retail_brand',
    'business_type_software_development', 'business_type_technology_electronics',
    'business_type_transportation_logistics', 'business_type_unknown'
]

TARGET_NAMES = ['impressions', 'clicks', 'actions', 'reach', 'conversion_value']

# Expected reasonable ranges for predictions (log scale)
REASONABLE_LOG_RANGES = {
    'impressions_log1p': (-1, 15),    # ~0 to ~3M impressions
    'clicks_log1p': (-1, 12),         # ~0 to ~160K clicks  
    'actions_log1p': (-1, 10),        # ~0 to ~22K actions
    'reach_log1p': (-1, 15),          # ~0 to ~3M reach
    'conversion_value_log1p': (-1, 12) # ~$0 to ~$160K
}


def parse_date(date_input: Union[str, date, datetime]) -> date:
    """Parse various date formats to date object"""
    if isinstance(date_input, date):
        return date_input
    elif isinstance(date_input, datetime):
        return date_input.date()
    elif isinstance(date_input, str):
        # Try common date formats
        formats = ['%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%Y/%m/%d', '%d-%m-%Y']
        for fmt in formats:
            try:
                return datetime.strptime(date_input, fmt).date()
            except ValueError:
                continue
        raise ValueError(f"Unable to parse date: {date_input}")
    else:
        raise ValueError(f"Invalid date type: {type(date_input)}")


def safe_div(num: float, den: float) -> float:
    """Safe division with epsilon to avoid division by zero"""
    return num / max(den, 1e-9)


def log1p_safe(x: float) -> float:
    """Safe log1p transformation"""
    return np.log1p(max(x, 0.0))


def sin_cycle(x: float, period: Union[int, float]) -> float:
    """Sine cycle encoding"""
    return np.sin(2.0 * np.pi * x / period)


def cos_cycle(x: float, period: Union[int, float]) -> float:
    """Cosine cycle encoding"""
    return np.cos(2.0 * np.pi * x / period)


def days_between(end_date: date, start_date: date) -> int:
    """Calculate days between dates, non-negative"""
    return max((end_date - start_date).days, 0)


def extract_time_features(d: date, prefix: str) -> Dict[str, float]:
    """Extract time features from a date with given prefix"""
    dom = d.day                                    # 1..31
    mon = d.month                                  # 1..12
    yr4 = d.year
    dow = d.weekday()                              # 0..6 (Mon=0)
    woy = d.isocalendar()[1]                       # Week of year
    doy = d.timetuple().tm_yday                    # 1..365/366
    
    # Days in month calculation
    if mon == 12:
        next_month_start = date(yr4 + 1, 1, 1)
    else:
        next_month_start = date(yr4, mon + 1, 1)
    
    current_month_start = date(yr4, mon, 1)
    dim = (next_month_start - current_month_start).days
    
    return {
        f"{prefix}_month": float(mon),
        f"{prefix}_year4": float(yr4),
        f"{prefix}_dow": float(dow),
        f"{prefix}_weekofyear": float(woy),
        f"{prefix}_doy": float(doy),
        f"{prefix}_days_in_month": float(dim),
        f"{prefix}_dom": float(dom),
        f"{prefix}_month_progress": float((dom - 1) / dim),
        f"{prefix}_quarter": float(((mon - 1) // 3) + 1),
        f"{prefix}_is_weekend": float(dow >= 5),
        f"{prefix}_is_month_start": float(dom == 1),
        f"{prefix}_is_month_end": float(dom == dim),
        f"{prefix}_month_sin": sin_cycle(mon, 12),
        f"{prefix}_month_cos": cos_cycle(mon, 12),
        f"{prefix}_dow_sin": sin_cycle(dow, 7),
        f"{prefix}_dow_cos": cos_cycle(dow, 7),
        f"{prefix}_doy_sin": sin_cycle(doy, 366),
        f"{prefix}_doy_cos": cos_cycle(doy, 366),
    }


def normalize_text(text: str) -> str:
    """Normalize text similar to training pipeline"""
    if pd.isna(text) or text is None:
        return "unknown"
    return str(text).lower().strip().replace(r'\s+', ' ')


def map_business_type(profile: str) -> str:
    """Map profile to business type using the same logic as training"""
    profile_norm = normalize_text(profile)
    
    for key, business_type in business_type_mapping.items():
        key_pattern = key.lower().strip()
        if key_pattern in profile_norm:
            return business_type
    
    return "unknown"


def create_one_hot_features(value: str, category_name: str, all_categories: List[str]) -> Dict[str, float]:
    """Create one-hot encoded features for a categorical value"""
    features = {}
    value_norm = normalize_text(value)
    
    for category in all_categories:
        feature_name = f"{category_name}_{category}"
        features[feature_name] = 1.0 if value_norm == category else 0.0
    
    return features


def engineer_features(
    campaign_start_date: Union[str, date],
    campaign_end_date: Union[str, date], 
    ad_group_start_date: Union[str, date],
    ad_group_end_date: Union[str, date],
    cost: float,
    campaign_objective: str,
    cta_type: str,
    impression_device: str,
    business_type: str
) -> Dict[str, float]:
    """
    Engineer features from raw inputs using the same logic as training pipeline
    """
    
    # Parse dates
    camp_start = parse_date(campaign_start_date)
    camp_end = parse_date(campaign_end_date)
    adg_start = parse_date(ad_group_start_date)
    adg_end = parse_date(ad_group_end_date)
    
    # Basic durations
    features = {
        'ad_group_duration': float(days_between(adg_end, adg_start)),
        'campaign_duration': float(days_between(camp_end, camp_start)),
        'cost': float(cost)
    }
    
    # Time features for each date
    features.update(extract_time_features(adg_start, 'adg_start'))
    features.update(extract_time_features(adg_end, 'adg_end'))
    features.update(extract_time_features(camp_start, 'camp_start'))
    features.update(extract_time_features(camp_end, 'camp_end'))
    
    # Date relationships
    features.update({
        'adg_start_minus_camp_start_days': float((adg_start - camp_start).days),
        'camp_end_minus_adg_end_days': float((camp_end - adg_end).days),
        'adg_start_offset_days': float(max((adg_start - camp_start).days, 0)),
        'adg_end_offset_days': float(max((camp_end - adg_end).days, 0)),
        'adg_inside_campaign': float(adg_start >= camp_start and adg_end <= camp_end),
        'adg_duration_ratio': safe_div(features['ad_group_duration'], 
                                     max(features['campaign_duration'], 1.0))
    })
    
    # Midpoint calculations
    adg_mid_days = int(features['ad_group_duration'] // 2)
    camp_mid_days = int(features['campaign_duration'] // 2)
    
    adg_mid = adg_start + timedelta(days=adg_mid_days)
    camp_mid = camp_start + timedelta(days=camp_mid_days)
    
    features.update({
        'adg_mid_month_sin': sin_cycle(adg_mid.month, 12),
        'adg_mid_month_cos': cos_cycle(adg_mid.month, 12),
        'camp_mid_month_sin': sin_cycle(camp_mid.month, 12),
        'camp_mid_month_cos': cos_cycle(camp_mid.month, 12)
    })
    
    # Cost-related features
    features.update({
        'adg_cost_per_day': safe_div(cost, max(features['ad_group_duration'], 1.0)),
        'camp_cost_per_day': safe_div(cost, max(features['campaign_duration'], 1.0))
    })
    
    # Cost interactions with time cycles (using log1p of cost)
    cost_log1p = log1p_safe(cost)
    features.update({
        'cost_x_adg_start_month_sin': cost_log1p * features['adg_start_month_sin'],
        'cost_x_adg_start_month_cos': cost_log1p * features['adg_start_month_cos'],
        'cost_x_camp_start_month_sin': cost_log1p * features['camp_start_month_sin'],
        'cost_x_camp_start_month_cos': cost_log1p * features['camp_start_month_cos']
    })
    
    # Normalize categorical values
    campaign_objective_norm = normalize_text(campaign_objective)
    cta_type_norm = normalize_text(cta_type)
    impression_device_norm = normalize_text(impression_device)
    business_type_final = business_type if business_type in [
        'beauty_cosmetics', 'digital_marketing', 'entertainment_media', 'fashion_lifestyle',
        'financial_services', 'food_beverage', 'healthcare_medical', 'industrial_manufacturing',
        'other', 'real_estate', 'retail_brand', 'software_development', 'technology_electronics',
        'transportation_logistics', 'unknown'
    ] else map_business_type(business_type)
    
    # One-hot encoding (based on the expected features list)
    # Campaign objective categories
    campaign_objective_categories = [
        'app_installs', 'brand_awareness', 'conversions', 'event_responses', 'lead_generation',
        'link_clicks', 'messages', 'outcome_app_promotion', 'outcome_awareness', 
        'outcome_engagement', 'outcome_leads', 'outcome_sales', 'outcome_traffic',
        'page_likes', 'post_engagement', 'product_catalog_sales', 'reach', 'store_visits',
        'video_views'
    ]
    
    # CTA type categories  
    cta_type_categories = [
        '', 'apply_now', 'book_travel', 'buy_tickets', 'call_now', 'contact_us', 'download',
        'event_rsvp', 'get_directions', 'get_offer', 'get_offer_view', 'get_promotions',
        'instagram_message', 'install_mobile_app', 'learn_more', 'like_page', 'listen_now',
        'message_page', 'no_button', 'order_now', 'play_game', 'shop_now', 'sign_up',
        'subscribe', 'view_instagram_profile', 'watch_more', 'whatsapp_message'
    ]
    
    # Impression device categories
    impression_device_categories = [
        '', 'android_smartphone', 'android_tablet', 'desktop', 'ipad', 'iphone', 'ipod', 'other'
    ]
    
    # Business type categories  
    business_type_categories = [
        'beauty_cosmetics', 'digital_marketing', 'entertainment_media', 'fashion_lifestyle',
        'financial_services', 'food_beverage', 'healthcare_medical', 'industrial_manufacturing',
        'other', 'real_estate', 'retail_brand', 'software_development', 'technology_electronics',
        'transportation_logistics', 'unknown'
    ]
    
    # Create one-hot features
    features.update(create_one_hot_features(campaign_objective_norm, 'campaign_objective_norm', campaign_objective_categories))
    features.update(create_one_hot_features(cta_type_norm, 'cta_type_norm', cta_type_categories))
    features.update(create_one_hot_features(impression_device_norm, 'impression_device_norm', impression_device_categories))
    features.update(create_one_hot_features(business_type_final, 'business_type', business_type_categories))
    
    return features


def calculate_prediction_range(prediction: float, mae: float, confidence_level: float = 0.95) -> Dict[str, float]:
    """Calculate prediction range based on MAE and confidence level"""
    
    # For MAE-based confidence intervals, we use empirical rules:
    # - 68% confidence: ±1 MAE
    # - 95% confidence: ±2 MAE  
    # - 99% confidence: ±3 MAE
    
    if confidence_level == 0.68:
        multiplier = 1.0
    elif confidence_level == 0.95:
        multiplier = 2.0
    elif confidence_level == 0.99:
        multiplier = 3.0
    else:
        # Interpolate for other confidence levels
        multiplier = 2.0 * confidence_level / 0.95
    
    margin = mae * multiplier
    
    return {
        'lower_bound': max(0.0, prediction - margin),
        'upper_bound': prediction + margin,
        'margin': margin,
        'confidence_level': confidence_level
    }


def calculate_performance_metrics(predictions: Dict[str, Dict], budget: float, campaign_duration_days: int = 13) -> Dict:
    """Calculate comprehensive performance metrics"""
    
    metrics = {
        'cost_efficiency': {},
        'reach_metrics': {},
        'engagement_metrics': {},
        'conversion_metrics': {},
        'quality_metrics': {}
    }
    
    # Extract predictions
    impressions = predictions.get('impressions', {}).get('predicted_value', 0)
    clicks = predictions.get('clicks', {}).get('predicted_value', 0)
    actions = predictions.get('actions', {}).get('predicted_value', 0)
    reach = predictions.get('reach', {}).get('predicted_value', 0)
    conversion_value = predictions.get('conversion_value', {}).get('predicted_value', 0)
    
    # Cost efficiency metrics
    if impressions > 0:
        metrics['cost_efficiency']['cost_per_impression'] = float(budget / impressions)
        metrics['cost_efficiency']['impressions_per_dollar'] = float(impressions / budget)
        metrics['cost_efficiency']['impressions_per_day'] = float(impressions / campaign_duration_days)
    
    if clicks > 0:
        metrics['cost_efficiency']['cost_per_click'] = float(budget / clicks)
        metrics['cost_efficiency']['clicks_per_dollar'] = float(clicks / budget)
        metrics['cost_efficiency']['clicks_per_day'] = float(clicks / campaign_duration_days)
    
    if actions > 0:
        metrics['cost_efficiency']['cost_per_action'] = float(budget / actions)
        metrics['cost_efficiency']['actions_per_dollar'] = float(actions / budget)
        metrics['cost_efficiency']['actions_per_day'] = float(actions / campaign_duration_days)
    
    # Reach metrics
    if reach > 0:
        metrics['reach_metrics']['estimated_reach'] = float(reach)
        metrics['reach_metrics']['reach_per_dollar'] = float(reach / budget)
        metrics['reach_metrics']['reach_per_day'] = float(reach / campaign_duration_days)
        
        if impressions > 0:
            metrics['reach_metrics']['frequency'] = float(impressions / reach)
            metrics['reach_metrics']['reach_efficiency'] = float(reach / impressions)
    
    # Engagement metrics
    if impressions > 0 and clicks > 0:
        ctr = clicks / impressions
        metrics['engagement_metrics']['click_through_rate'] = float(ctr)
        metrics['engagement_metrics']['ctr_percentage'] = float(ctr * 100)
        
        # CTR benchmarks
        if ctr > 0.05:  # 5%
            metrics['engagement_metrics']['ctr_benchmark'] = "Excellent"
        elif ctr > 0.02:  # 2%
            metrics['engagement_metrics']['ctr_benchmark'] = "Good"
        elif ctr > 0.01:  # 1%
            metrics['engagement_metrics']['ctr_benchmark'] = "Average"
        else:
            metrics['engagement_metrics']['ctr_benchmark'] = "Below Average"
    
    if clicks > 0 and actions > 0:
        conversion_rate = actions / clicks
        metrics['engagement_metrics']['conversion_rate'] = float(conversion_rate)
        metrics['engagement_metrics']['conversion_rate_percentage'] = float(conversion_rate * 100)
    
    # Conversion metrics
    if actions > 0 and conversion_value > 0:
        metrics['conversion_metrics']['value_per_conversion'] = float(conversion_value / actions)
        metrics['conversion_metrics']['conversions_per_dollar'] = float(actions / budget)
    
    if conversion_value > 0:
        roas = conversion_value / budget
        metrics['conversion_metrics']['return_on_ad_spend'] = float(roas)
        metrics['conversion_metrics']['roas_percentage'] = float(roas * 100)
        
        # ROAS benchmarks
        if roas > 4.0:  # 400%
            metrics['conversion_metrics']['roas_benchmark'] = "Excellent"
        elif roas > 2.0:  # 200%
            metrics['conversion_metrics']['roas_benchmark'] = "Good"
        elif roas > 1.0:  # 100%
            metrics['conversion_metrics']['roas_benchmark'] = "Break-even"
        else:
            metrics['conversion_metrics']['roas_benchmark'] = "Loss"
    
    # Quality metrics
    if impressions > 0 and reach > 0:
        frequency = impressions / reach
        if frequency > 3.0:
            metrics['quality_metrics']['frequency_benchmark'] = "High (may cause fatigue)"
        elif frequency > 1.5:
            metrics['quality_metrics']['frequency_benchmark'] = "Optimal"
        else:
            metrics['quality_metrics']['frequency_benchmark'] = "Low (may need more exposure)"
    
    return metrics


def safe_inverse_transform(log_pred: float, target: str) -> float:
    """Safely convert log prediction back to original scale with bounds checking"""
    
    # Get reasonable range for this target
    min_log, max_log = REASONABLE_LOG_RANGES.get(target, (-1, 10))
    
    # Clamp the log prediction to reasonable range
    log_pred_clamped = np.clip(log_pred, min_log, max_log)
    
    # If we had to clamp, warn the user
    if abs(log_pred - log_pred_clamped) > 0.1:
        print(f"⚠️ Warning: {target} prediction was clamped from {log_pred:.2f} to {log_pred_clamped:.2f} (log scale)")
    
    # Convert back to original scale
    original_pred = np.expm1(log_pred_clamped)
    
    # Additional safety: ensure non-negative
    return max(original_pred, 0.0)


class SimpleEnsembleReconstructor:
    """Simple ensemble reconstructor for inference without custom classes"""
    
    def __init__(self, base_models: Dict, ensemble_type: str, weights: Dict = None, meta_model = None):
        self.base_models = base_models
        self.ensemble_type = ensemble_type
        self.weights = weights
        self.meta_model = meta_model
    
    def predict(self, X):
        """Make ensemble predictions"""
        # Get predictions from all base models
        predictions = {}
        for name, model in self.base_models.items():
            predictions[name] = model.predict(X)
        
        if self.ensemble_type == 'voting':
            # Simple average
            return np.mean(list(predictions.values()), axis=0)
        
        elif self.ensemble_type == 'weighted_voting':
            # Weighted average
            if self.weights is None:
                return np.mean(list(predictions.values()), axis=0)
            
            weighted_sum = np.zeros_like(list(predictions.values())[0])
            total_weight = 0
            for name, pred in predictions.items():
                weight = self.weights.get(name, 0)
                weighted_sum += weight * pred
                total_weight += weight
            
            return weighted_sum / total_weight if total_weight > 0 else weighted_sum
        
        elif self.ensemble_type == 'stacking':
            # Use meta-model to combine predictions
            if self.meta_model is None:
                # Fallback to simple average
                return np.mean(list(predictions.values()), axis=0)
            
            X_meta = np.column_stack(list(predictions.values()))
            return self.meta_model.predict(X_meta)
        
        else:
            # Default to simple average
            return np.mean(list(predictions.values()), axis=0)


def load_models_and_metadata(models_dir: str, standardization_path: Optional[str] = None) -> Tuple[Dict, Dict, Dict]:
    """Load trained models and preprocessing metadata with ensemble reconstruction"""
    
    print(f"🔍 Loading models from: {models_dir}")
    
    # Load model info
    best_models_info = joblib.load(os.path.join(models_dir, 'best_models_info.joblib'))
    training_metadata = joblib.load(os.path.join(models_dir, 'training_metadata.joblib'))
    
    # Load individual best models
    models = {}
    for target, info in best_models_info.items():
        target_clean = target.replace('_log1p', '')
        model_path = os.path.join(models_dir, f'best_model_{target_clean}.joblib')
        
        if os.path.exists(model_path):
            loaded_model = joblib.load(model_path)
            
            # Check if this is an ensemble reconstruction dictionary
            if isinstance(loaded_model, dict) and 'ensemble_type' in loaded_model:
                print(f"🔄 Reconstructing ensemble model: {target_clean} ({loaded_model['ensemble_type']})")
                
                # Load base models
                base_models = {}
                base_model_paths = loaded_model.get('base_model_paths', {})
                
                if base_model_paths:
                    # Use explicit base model paths
                    for name, base_path in base_model_paths.items():
                        if os.path.exists(base_path):
                            base_models[name] = joblib.load(base_path)
                            print(f"   ✅ Loaded base model: {name}")
                        else:
                            print(f"   ⚠️ Base model not found: {base_path}")
                else:
                    # Try to find base models by naming convention
                    base_model_names = loaded_model.get('base_model_names', [])
                    for name in base_model_names:
                        base_path = os.path.join(models_dir, f'base_model_{target_clean}_{name}.joblib')
                        if os.path.exists(base_path):
                            base_models[name] = joblib.load(base_path)
                            print(f"   ✅ Loaded base model: {name}")
                        else:
                            print(f"   ⚠️ Base model not found: {base_path}")
                
                # Load meta-model if stacking
                meta_model = None
                if loaded_model['ensemble_type'] == 'stacking':
                    meta_model_path = loaded_model.get('meta_model_path')
                    if meta_model_path and os.path.exists(meta_model_path):
                        meta_model = joblib.load(meta_model_path)
                        print(f"   ✅ Loaded meta-model for stacking")
                    else:
                        print(f"   ⚠️ Meta-model not found for stacking ensemble")
                
                # Create ensemble reconstructor
                if base_models:
                    ensemble_model = SimpleEnsembleReconstructor(
                        base_models=base_models,
                        ensemble_type=loaded_model['ensemble_type'],
                        weights=loaded_model.get('weights'),
                        meta_model=meta_model
                    )
                    models[target] = ensemble_model
                    print(f"✅ Reconstructed ensemble model: {target_clean}")
                else:
                    print(f"❌ No base models found for ensemble: {target_clean}")
            
            else:
                # Regular individual model
                models[target] = loaded_model
                print(f"✅ Loaded individual model: {target_clean}")
        else:
            print(f"⚠️ Warning: Model file not found: {model_path}")
    
    # Load preprocessing parameters
    preprocessing_params = {}
    
    # Load standardization parameters from specified path
    if standardization_path:
        if os.path.exists(standardization_path):
            try:
                preprocessing_params['standardization_params'] = joblib.load(standardization_path)
                print(f"✅ Loaded standardization params: {standardization_path}")
            except Exception as e:
                print(f"❌ Error loading standardization params: {e}")
                raise
        else:
            print(f"❌ Standardization file not found: {standardization_path}")
            raise FileNotFoundError(f"Standardization parameters file not found: {standardization_path}")
    else:
        # Try to find standardization parameters in models directory
        standardization_files = [
            'fbads_clean_standardization_params.joblib',
            'standardization_params.joblib',
            'fbads_standardization_params.joblib',
            'clean_standardization_params.joblib'
        ]
        
        standardization_loaded = False
        for filename in standardization_files:
            file_path = os.path.join(models_dir, filename)
            if os.path.exists(file_path):
                try:
                    preprocessing_params['standardization_params'] = joblib.load(file_path)
                    print(f"✅ Found standardization params: {filename}")
                    standardization_loaded = True
                    break
                except Exception as e:
                    print(f"⚠️ Could not load {filename}: {e}")
        
        if not standardization_loaded:
            available_files = os.listdir(models_dir)
            print(f"❌ No standardization parameters found in models directory!")
            print(f"   Available files: {available_files}")
            print(f"   💡 Use --standardization-params to specify the file location")
            raise FileNotFoundError("Standardization parameters not found. Please specify with --standardization-params")
    
    # Load one-hot encoding mapping (optional)
    ohe_files = [
        'fbads_clean_ohe_mapping.joblib',
        'ohe_mapping.joblib',
        'fbads_ohe_mapping.joblib'
    ]
    
    for filename in ohe_files:
        file_path = os.path.join(models_dir, filename)
        if os.path.exists(file_path):
            try:
                preprocessing_params['ohe_mapping'] = joblib.load(file_path)
                print(f"✅ Loaded OHE mapping: {filename}")
                break
            except Exception as e:
                print(f"⚠️ Could not load {filename}: {e}")
    
    return models, best_models_info, preprocessing_params


def apply_standardization_from_params(features: pd.DataFrame, std_params: Dict) -> pd.DataFrame:
    """Apply standardization using saved parameters"""
    
    print(f"🔧 Applying standardization to {features.shape[1]} features...")
    
    # Identify columns to exclude from standardization (one-hot encoded features)
    exclude_cols = []
    ohe_prefixes = ['campaign_objective_norm_', 'cta_type_norm_', 'impression_device_norm_', 'business_type_']
    for col in features.columns:
        if any(col.startswith(prefix) for prefix in ohe_prefixes):
            exclude_cols.append(col)
    
    # Apply standardization to numeric columns only
    numeric_cols = [col for col in features.columns if col not in exclude_cols]
    
    standardized_count = 0
    extreme_values = []
    missing_params = []
    
    for col in numeric_cols:
        if col in std_params:
            try:
                mean_val, std_val = std_params[col]
                original_val = features[col].iloc[0]
                features[col] = (features[col] - mean_val) / max(std_val, 1e-8)
                standardized_count += 1
                
                # Check for extreme standardized values
                standardized_val = features[col].iloc[0]
                if abs(standardized_val) > 5:  # More than 5 standard deviations is extreme
                    extreme_values.append((col, standardized_val, original_val))
                    
            except Exception as e:
                print(f"❌ Error standardizing {col}: {e}")
                continue
        else:
            missing_params.append(col)
    
    print(f"✅ Standardized {standardized_count}/{len(numeric_cols)} numeric features")
    print(f"   Excluded {len(exclude_cols)} one-hot encoded features")
    
    if missing_params:
        print(f"⚠️ Missing standardization params for {len(missing_params)} features: {missing_params[:5]}...")
    
    if extreme_values:
        print(f"⚠️ Extreme standardized values detected:")
        for col, std_val, orig_val in extreme_values[:3]:  # Show first 3
            print(f"   {col}: {std_val:.2f} std devs (original: {orig_val:.2f})")
    
    return features


def apply_preprocessing_with_standardization(features: Dict[str, float], preprocessing_params: Dict) -> pd.DataFrame:
    """Apply preprocessing with proper standardization"""
    
    # Convert to DataFrame for consistent handling
    feature_df = pd.DataFrame([features])
    
    # Ensure all expected features are present
    missing_features = []
    for feature in EXPECTED_FEATURES:
        if feature not in feature_df.columns:
            missing_features.append(feature)
            feature_df[feature] = 0.0
    
    if missing_features:
        print(f"⚠️ Added {len(missing_features)} missing features (set to 0.0)")
    
    # Reorder to match expected order
    feature_df = feature_df[EXPECTED_FEATURES]
    
    # Apply standardization
    if 'standardization_params' in preprocessing_params:
        std_params = preprocessing_params['standardization_params']
        feature_df = apply_standardization_from_params(feature_df, std_params)
    else:
        raise ValueError("Standardization parameters are required but not provided!")
    
    return feature_df


def make_predictions_with_proper_standardization(models: Dict, features: pd.DataFrame, best_models_info: Dict) -> Dict[str, Dict]:
    """Make predictions with proper feature scaling validation"""
    
    predictions = {}
    
    # Show feature summary for debugging
    print(f"📊 Final feature summary: {features.shape[1]} features")
    non_zero_features = (features != 0).sum(axis=1).iloc[0]
    print(f"   Non-zero features: {non_zero_features}")
    
    # Check feature statistics to verify standardization
    numeric_cols = [col for col in features.columns 
                   if not any(col.startswith(prefix) for prefix in 
                            ['campaign_objective_norm_', 'cta_type_norm_', 'impression_device_norm_', 'business_type_'])]
    
    if len(numeric_cols) > 0:
        numeric_features = features[numeric_cols]
        mean_abs_values = np.abs(numeric_features.values).mean()
        max_abs_value = np.abs(numeric_features.values).max()
        
        print(f"   Numeric feature stats: mean_abs={mean_abs_values:.3f}, max_abs={max_abs_value:.3f}")
        
        # Check if features look standardized (most values should be within [-3, 3])
        if mean_abs_values < 2 and max_abs_value < 10:
            print(f"✅ Features appear to be properly standardized")
        else:
            print(f"⚠️ WARNING: Features may not be properly standardized!")
    
    # Extract cost for sanity checks (cost is now standardized, so we use a default)
    cost_for_checks = 250  # Use default for checks since cost is standardized
    
    for target, model in models.items():
        target_clean = target.replace('_log1p', '')
        
        try:
            # Make prediction (in log scale) 
            pred_log = model.predict(features)[0]
            
            print(f"🎯 {target_clean}: Raw log prediction = {pred_log:.3f}")
            
            # Check if log prediction is reasonable
            min_log, max_log = REASONABLE_LOG_RANGES.get(target, (-1, 10))
            is_reasonable_log = (min_log - 2 <= pred_log <= max_log + 2)
            
            if not is_reasonable_log:
                print(f"⚠️ {target_clean}: Log prediction {pred_log:.3f} is outside expected range [{min_log}, {max_log}]")
            
            # Convert back to original scale with safety
            pred_original = safe_inverse_transform(pred_log, target)
            
            # Get model info
            model_info = best_models_info[target]
            model_name = model_info['model_name']
            
            # Budget-based sanity checks
            is_reasonable = True
            reason = ""
            
            if target_clean == 'clicks' and pred_original > cost_for_checks * 50:  # More than 50 clicks per dollar is suspicious
                is_reasonable = False
                reason = f"Clicks/budget ratio too high: {pred_original/cost_for_checks:.1f} clicks per dollar (expected < 50)"
            elif target_clean == 'impressions' and pred_original > cost_for_checks * 500:  # More than 500 impressions per dollar is suspicious
                is_reasonable = False  
                reason = f"Impressions/budget ratio too high: {pred_original/cost_for_checks:.1f} impressions per dollar (expected < 500)"
            elif target_clean == 'actions' and pred_original > cost_for_checks * 5:  # More than 5 actions per dollar is very optimistic
                is_reasonable = False
                reason = f"Actions/budget ratio too high: {pred_original/cost_for_checks:.1f} actions per dollar (expected < 5)"
            elif pred_original < 0:
                is_reasonable = False
                reason = "Negative prediction doesn't make business sense"
            
            # Extract performance metrics safely from the new structure
            try:
                # Try new structure first (val_metrics, test_metrics directly)
                val_r2 = model_info.get('val_metrics', {}).get('log', {}).get('r2', 0.0)
                test_r2 = model_info.get('test_metrics', {}).get('log', {}).get('r2', 0.0)
                val_mae_orig = model_info.get('val_metrics', {}).get('orig', {}).get('mae', 0.0)
                test_mae_orig = model_info.get('test_metrics', {}).get('orig', {}).get('mae', 0.0)
                
                # If not found, try old structure (results['val'])
                if val_r2 == 0.0 and 'results' in model_info:
                    val_r2 = model_info.get('results', {}).get('val', {}).get('log_r2', 0.0)
                    test_r2 = model_info.get('results', {}).get('test', {}).get('log_r2', 0.0)
                    val_mae_orig = model_info.get('results', {}).get('val', {}).get('orig_mae', 0.0)
                    test_mae_orig = model_info.get('results', {}).get('test', {}).get('orig_mae', 0.0)
                    
            except Exception as metric_error:
                print(f"⚠️ Error extracting metrics for {target_clean}: {metric_error}")
                val_r2 = test_r2 = val_mae_orig = test_mae_orig = 0.0
            
            # Calculate prediction ranges using test MAE (more conservative)
            prediction_ranges = {}
            if test_mae_orig > 0:
                prediction_ranges['confidence_68'] = calculate_prediction_range(pred_original, test_mae_orig, 0.68)
                prediction_ranges['confidence_95'] = calculate_prediction_range(pred_original, test_mae_orig, 0.95)
                prediction_ranges['confidence_99'] = calculate_prediction_range(pred_original, test_mae_orig, 0.99)
            
            predictions[target_clean] = {
                'predicted_value': float(pred_original),
                'predicted_log': float(pred_log),
                'model_used': model_name,
                'is_reasonable': is_reasonable and is_reasonable_log,
                'warning': reason if not is_reasonable else 
                          (f"Log prediction outside normal range" if not is_reasonable_log else None),
                'prediction_ranges': prediction_ranges,
                'model_performance': {
                    'val_r2': val_r2,
                    'test_r2': test_r2,
                    'val_mae_original': val_mae_orig,
                    'test_mae_original': test_mae_orig
                }
            }
            
        except Exception as e:
            print(f"❌ Error predicting {target_clean}: {str(e)}")
            predictions[target_clean] = {
                'predicted_value': None,
                'error': str(e)
            }
    
    return predictions


def format_predictions_complete(predictions: Dict[str, Dict], input_data: Dict) -> str:
    """Format predictions with complete information"""
    
    output = []
    output.append("=" * 80)
    output.append("📊 FACEBOOK ADS PERFORMANCE PREDICTION")
    output.append("=" * 80)
    
    # Input summary
    output.append("\n📋 INPUT SUMMARY:")
    output.append(f"Campaign Period: {input_data['campaign_start_date']} → {input_data['campaign_end_date']}")
    output.append(f"Ad Group Period:  {input_data['ad_group_start_date']} → {input_data['ad_group_end_date']}")
    output.append(f"Budget: ${input_data['cost']:,.2f}")
    output.append(f"Objective: {input_data['campaign_objective']}")
    output.append(f"CTA Type: {input_data['cta_type']}")
    output.append(f"Device: {input_data['impression_device']}")
    output.append(f"Business Type: {input_data['business_type']}")
    
    # Predictions
    output.append("\n🎯 PREDICTED PERFORMANCE:")
    output.append("-" * 50)
    
    unreasonable_predictions = []
    
    for target, pred_data in predictions.items():
        if 'predicted_value' in pred_data and pred_data['predicted_value'] is not None:
            value = pred_data['predicted_value']
            model_name = pred_data['model_used']
            val_r2 = pred_data['model_performance']['val_r2']
            is_reasonable = pred_data.get('is_reasonable', True)
            warning = pred_data.get('warning')
            
            # Format value based on type
            if target in ['impressions', 'clicks', 'actions', 'reach']:
                value_str = f"{value:10,.0f}"
            else:  # conversion_value
                value_str = f"${value:9,.2f}"
            
            line = f"{target.title():15} : {value_str}   (Model: {model_name}, R²: {val_r2:.3f})"
            
            if not is_reasonable:
                line += " ⚠️ UNRELIABLE"
                unreasonable_predictions.append((target, warning))
            
            output.append(line)
        else:
            output.append(f"{target.title():15} : ERROR - {pred_data.get('error', 'Unknown error')}")
    
    # Show warnings
    if unreasonable_predictions:
        output.append("\n⚠️ PREDICTION WARNINGS:")
        output.append("-" * 50)
        for target, warning in unreasonable_predictions:
            output.append(f"{target.title()}: {warning}")
        output.append("\n💡 These predictions may be unreliable because:")
        output.append("   • The input values create feature combinations not seen during training")
        output.append("   • The model is extrapolating beyond its training data range")
        output.append("   • Use predictions as rough estimates rather than precise forecasts")
    
    # Model performance summary
    output.append("\n📈 MODEL RELIABILITY:")
    output.append("-" * 50)
    successful_preds = {k: v for k, v in predictions.items() 
                       if 'predicted_value' in v and v['predicted_value'] is not None}
    
    if successful_preds:
        avg_val_r2 = np.mean([pred['model_performance']['val_r2'] for pred in successful_preds.values()])
        avg_test_r2 = np.mean([pred['model_performance']['test_r2'] for pred in successful_preds.values()])
        
        output.append(f"Average Validation R²: {avg_val_r2:.3f}")
        output.append(f"Average Test R²:       {avg_test_r2:.3f}")
        
        reliability = "High" if avg_test_r2 > 0.3 else "Medium" if avg_test_r2 > 0.15 else "Low"
        output.append(f"Overall Reliability:   {reliability}")
    
    output.append("\n" + "=" * 80)
    
    return "\n".join(output)


def create_json_output(predictions: Dict[str, Dict], input_data: Dict, best_models_info: Dict) -> Dict:
    """Create comprehensive JSON output with all prediction details"""
    
    # Calculate summary statistics
    successful_preds = {k: v for k, v in predictions.items() 
                       if 'predicted_value' in v and v['predicted_value'] is not None}
    
    avg_val_r2 = 0.0
    avg_test_r2 = 0.0
    reliability = "Unknown"
    
    if successful_preds:
        avg_val_r2 = np.mean([pred['model_performance']['val_r2'] for pred in successful_preds.values()])
        avg_test_r2 = np.mean([pred['model_performance']['test_r2'] for pred in successful_preds.values()])
        reliability = "High" if avg_test_r2 > 0.3 else "Medium" if avg_test_r2 > 0.15 else "Low"
    
    # Count warnings
    unreliable_predictions = [k for k, v in predictions.items() 
                            if not v.get('is_reasonable', True)]
    
    # Format predictions for JSON
    formatted_predictions = {}
    for target, pred_data in predictions.items():
        if 'predicted_value' in pred_data and pred_data['predicted_value'] is not None:
            value = pred_data['predicted_value']
            
            # Determine value type and format
            if target in ['impressions', 'clicks', 'actions', 'reach']:
                formatted_value = {
                    'raw_value': float(value),
                    'formatted_value': f"{value:,.0f}",
                    'unit': "count"
                }
            else:  # conversion_value
                formatted_value = {
                    'raw_value': float(value),
                    'formatted_value': f"${value:,.2f}",
                    'unit': "currency_usd"
                }
            
            formatted_predictions[target] = {
                'prediction': formatted_value,
                'log_prediction': float(pred_data['predicted_log']),
                'model_used': str(pred_data['model_used']),
                'model_type': "ensemble" if pred_data['model_used'].startswith('ensemble_') else "individual",
                'is_reliable': bool(pred_data.get('is_reasonable', True)),
                'warning': str(pred_data.get('warning')) if pred_data.get('warning') is not None else None,
                'prediction_ranges': pred_data.get('prediction_ranges', {}),
                'performance_metrics': {
                    'validation_r2': float(pred_data['model_performance']['val_r2']),
                    'test_r2': float(pred_data['model_performance']['test_r2']),
                    'validation_mae_original': float(pred_data['model_performance']['val_mae_original']),
                    'test_mae_original': float(pred_data['model_performance']['test_mae_original'])
                }
            }
        else:
            formatted_predictions[target] = {
                'prediction': None,
                'error': str(pred_data.get('error', 'Unknown error')),
                'model_used': None,
                'is_reliable': False
            }
    
    # Campaign duration calculations
    from datetime import datetime as dt
    campaign_start = dt.strptime(input_data['campaign_start_date'], '%Y-%m-%d')
    campaign_end = dt.strptime(input_data['campaign_end_date'], '%Y-%m-%d')
    campaign_duration = (campaign_end - campaign_start).days
    
    adg_start = dt.strptime(input_data['ad_group_start_date'], '%Y-%m-%d')
    adg_end = dt.strptime(input_data['ad_group_end_date'], '%Y-%m-%d')
    adg_duration = (adg_end - adg_start).days
    
    # Create comprehensive result
    result = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "prediction_engine": "Facebook Ads ML Ensemble",
            "version": "1.0",
            "standardization_applied": True,
            "total_features_used": 160,
            "ensemble_methods": ["voting", "weighted_voting", "stacking"]
        },
        "input_parameters": {
            "campaign": {
                "start_date": input_data['campaign_start_date'],
                "end_date": input_data['campaign_end_date'],
                "duration_days": campaign_duration,
                "objective": input_data['campaign_objective'],
                "budget": {
                    "amount": float(input_data['cost']),
                    "currency": "USD",
                    "daily_budget": float(input_data['cost']) / max(campaign_duration, 1)
                }
            },
            "ad_group": {
                "start_date": input_data['ad_group_start_date'],
                "end_date": input_data['ad_group_end_date'],
                "duration_days": adg_duration,
                "cta_type": input_data['cta_type'],
                "primary_device": input_data['impression_device']
            },
            "targeting": {
                "business_type": input_data['business_type']
            }
        },
        "predictions": formatted_predictions,
        "summary": {
            "total_targets": len(predictions),
            "successful_predictions": len(successful_preds),
            "failed_predictions": len(predictions) - len(successful_preds),
            "unreliable_predictions": len(unreliable_predictions),
            "overall_reliability": reliability,
            "average_validation_r2": float(avg_val_r2),
            "average_test_r2": float(avg_test_r2)
        },
        "performance_estimates": calculate_performance_metrics(predictions, float(input_data['cost']), campaign_duration),
        "warnings": []
    }
    
    # Add warnings
    if unreliable_predictions:
        for target in unreliable_predictions:
            warning_msg = predictions[target].get('warning', 'Prediction may be unreliable')
            result["warnings"].append({
                "target": str(target),
                "type": "unreliable_prediction",
                "message": str(warning_msg)
            })
    
    if avg_test_r2 < 0.2:
        result["warnings"].append({
            "type": "low_model_performance",
            "message": f"Average model R² ({avg_test_r2:.3f}) is below recommended threshold (0.2)"
        })
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Facebook Ads Performance Prediction - Complete Version")
    parser.add_argument("--models-dir", required=True, help="Directory containing trained models")
    parser.add_argument("--standardization-params", help="Path to standardization parameters file (fbads_clean_standardization_params.joblib)")
    parser.add_argument("--campaign-start-date", required=True, help="Campaign start date (YYYY-MM-DD)")
    parser.add_argument("--campaign-end-date", required=True, help="Campaign end date (YYYY-MM-DD)")
    parser.add_argument("--ad-group-start-date", required=True, help="Ad group start date (YYYY-MM-DD)")
    parser.add_argument("--ad-group-end-date", required=True, help="Ad group end date (YYYY-MM-DD)")
    parser.add_argument("--cost", type=float, required=True, help="Campaign budget/cost")
    parser.add_argument("--campaign-objective", required=True, help="Campaign objective")
    parser.add_argument("--cta-type", required=True, help="Call-to-action type")
    parser.add_argument("--impression-device", required=True, help="Primary impression device")
    parser.add_argument("--business-type", required=True, help="Business type or profile name")
    parser.add_argument("--output-format", choices=['text', 'json'], default='text', help="Output format")
    
    args = parser.parse_args()
    
    # Collect input data
    input_data = {
        'campaign_start_date': args.campaign_start_date,
        'campaign_end_date': args.campaign_end_date, 
        'ad_group_start_date': args.ad_group_start_date,
        'ad_group_end_date': args.ad_group_end_date,
        'cost': args.cost,
        'campaign_objective': args.campaign_objective,
        'cta_type': args.cta_type,
        'impression_device': args.impression_device,
        'business_type': args.business_type
    }
    
    try:
        print("🚀 Loading models and preprocessing parameters...")
        models, best_models_info, preprocessing_params = load_models_and_metadata(
            args.models_dir, 
            args.standardization_params
        )
        
        print("⚙️ Engineering features...")
        features = engineer_features(**input_data)
        print(f"   Generated {len(features)} raw features")
        
        print("🔧 Applying preprocessing...")
        processed_features = apply_preprocessing_with_standardization(features, preprocessing_params)
        
        print("🎯 Making predictions...")
        predictions = make_predictions_with_proper_standardization(models, processed_features, best_models_info)
        
        # Output results
        if args.output_format == 'json':
            # Create comprehensive JSON output
            result = create_json_output(predictions, input_data, best_models_info)
            print(json.dumps(result, indent=2))
        else:
            print(format_predictions_complete(predictions, input_data))
    
    except Exception as e:
        print(f"❌ Error during prediction: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()