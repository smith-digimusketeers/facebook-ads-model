# Input : 
```cli
!python fbads_inference.py \
  --models-dir ./models_memory_efficient \
  --standardization-params /content/fbads_clean_standardization_params.joblib \
  --campaign-start-date "2025-05-06" \
  --campaign-end-date "2025-06-15" \
  --ad-group-start-date "2025-05-06" \
  --ad-group-end-date "2025-06-15" \
  --cost 87.89 \
  --campaign-objective "OUTCOME_TRAFFIC" \
  --cta-type "MESSAGE_PAGE" \
  --impression-device "android_smartphone" \
  --business-type "real_estate" \
  --output-format json
```
# Output :
```json
{
  "metadata": {
    "timestamp": "2025-09-08T04:17:57.980078",
    "prediction_engine": "Facebook Ads ML Ensemble",
    "version": "1.0",
    "standardization_applied": true,
    "total_features_used": 160,
    "ensemble_methods": [
      "voting",
      "weighted_voting",
      "stacking"
    ]
  },
  "input_parameters": {
    "campaign": {
      "start_date": "2025-05-06",
      "end_date": "2025-06-15",
      "duration_days": 40,
      "objective": "OUTCOME_TRAFFIC",
      "budget": {
        "amount": 87.89,
        "currency": "USD",
        "daily_budget": 2.19725
      }
    },
    "ad_group": {
      "start_date": "2025-05-06",
      "end_date": "2025-06-15",
      "duration_days": 40,
      "cta_type": "MESSAGE_PAGE",
      "primary_device": "android_smartphone"
    },
    "targeting": {
      "business_type": "real_estate"
    }
  },
  "predictions": {
    "impressions": {
      "prediction": {
        "raw_value": 642.1373656955957,
        "formatted_value": "642",
        "unit": "count"
      },
      "log_prediction": 6.466358333918485,
      "model_used": "ensemble_stacking",
      "model_type": "ensemble",
      "is_reliable": true,
      "warning": null,
      "prediction_ranges": {
        "confidence_68": {
          "lower_bound": 561.9277808298388,
          "upper_bound": 722.3469505613526,
          "margin": 80.20958486575697,
          "confidence_level": 0.68
        },
        "confidence_95": {
          "lower_bound": 481.71819596408176,
          "upper_bound": 802.5565354271096,
          "margin": 160.41916973151393,
          "confidence_level": 0.95
        },
        "confidence_99": {
          "lower_bound": 401.50861109832476,
          "upper_bound": 882.7661202928666,
          "margin": 240.6287545972709,
          "confidence_level": 0.99
        }
      },
      "performance_metrics": {
        "validation_r2": 0.8738440287759996,
        "test_r2": 0.8745258409087338,
        "validation_mae_original": 90.54808708873308,
        "test_mae_original": 80.20958486575697
      }
    },
    "clicks": {
      "prediction": {
        "raw_value": 9.715245182361498,
        "formatted_value": "10",
        "unit": "count"
      },
      "log_prediction": 2.3716675108253433,
      "model_used": "ensemble_stacking",
      "model_type": "ensemble",
      "is_reliable": true,
      "warning": null,
      "prediction_ranges": {
        "confidence_68": {
          "lower_bound": 6.868408340757342,
          "upper_bound": 12.562082023965655,
          "margin": 2.846836841604156,
          "confidence_level": 0.68
        },
        "confidence_95": {
          "lower_bound": 4.021571499153186,
          "upper_bound": 15.40891886556981,
          "margin": 5.693673683208312,
          "confidence_level": 0.95
        },
        "confidence_99": {
          "lower_bound": 1.1747346575490294,
          "upper_bound": 18.255755707173968,
          "margin": 8.540510524812468,
          "confidence_level": 0.99
        }
      },
      "performance_metrics": {
        "validation_r2": 0.7016607769727796,
        "test_r2": 0.7031356050420214,
        "validation_mae_original": 2.8376348509213956,
        "test_mae_original": 2.846836841604156
      }
    },
    "actions": {
      "prediction": {
        "raw_value": 38.369678733531515,
        "formatted_value": "38",
        "unit": "count"
      },
      "log_prediction": 3.672995944737732,
      "model_used": "ensemble_stacking",
      "model_type": "ensemble",
      "is_reliable": true,
      "warning": null,
      "prediction_ranges": {
        "confidence_68": {
          "lower_bound": 0.0,
          "upper_bound": 79.63247366163702,
          "margin": 41.2627949281055,
          "confidence_level": 0.68
        },
        "confidence_95": {
          "lower_bound": 0.0,
          "upper_bound": 120.89526858974251,
          "margin": 82.525589856211,
          "confidence_level": 0.95
        },
        "confidence_99": {
          "lower_bound": 0.0,
          "upper_bound": 162.158063517848,
          "margin": 123.78838478431649,
          "confidence_level": 0.99
        }
      },
      "performance_metrics": {
        "validation_r2": 0.6709386985429622,
        "test_r2": 0.6719278043163439,
        "validation_mae_original": 39.889905671924,
        "test_mae_original": 41.2627949281055
      }
    },
    "reach": {
      "prediction": {
        "raw_value": 313.6956897309137,
        "formatted_value": "314",
        "unit": "count"
      },
      "log_prediction": 5.751606107538413,
      "model_used": "ensemble_stacking",
      "model_type": "ensemble",
      "is_reliable": true,
      "warning": null,
      "prediction_ranges": {
        "confidence_68": {
          "lower_bound": 240.17700154142335,
          "upper_bound": 387.2143779204041,
          "margin": 73.51868818949036,
          "confidence_level": 0.68
        },
        "confidence_95": {
          "lower_bound": 166.658313351933,
          "upper_bound": 460.7330661098945,
          "margin": 147.03737637898072,
          "confidence_level": 0.95
        },
        "confidence_99": {
          "lower_bound": 93.13962516244266,
          "upper_bound": 534.2517542993849,
          "margin": 220.55606456847107,
          "confidence_level": 0.99
        }
      },
      "performance_metrics": {
        "validation_r2": 0.8680285344297695,
        "test_r2": 0.8686260304470881,
        "validation_mae_original": 81.87416035233234,
        "test_mae_original": 73.51868818949036
      }
    },
    "conversion_value": {
      "prediction": {
        "raw_value": 0.05352789879236862,
        "formatted_value": "$0.05",
        "unit": "currency_usd"
      },
      "log_prediction": 0.05214443591656055,
      "model_used": "ensemble_stacking",
      "model_type": "ensemble",
      "is_reliable": true,
      "warning": null,
      "prediction_ranges": {
        "confidence_68": {
          "lower_bound": 0.0,
          "upper_bound": 1018.577266276882,
          "margin": 1018.5237383780897,
          "confidence_level": 0.68
        },
        "confidence_95": {
          "lower_bound": 0.0,
          "upper_bound": 2037.1010046549718,
          "margin": 2037.0474767561793,
          "confidence_level": 0.95
        },
        "confidence_99": {
          "lower_bound": 0.0,
          "upper_bound": 3055.6247430330613,
          "margin": 3055.571215134269,
          "confidence_level": 0.99
        }
      },
      "performance_metrics": {
        "validation_r2": 0.5435539180251178,
        "test_r2": 0.5433387452751376,
        "validation_mae_original": 1028.595313437066,
        "test_mae_original": 1018.5237383780897
      }
    }
  },
  "summary": {
    "total_targets": 5,
    "successful_predictions": 5,
    "failed_predictions": 0,
    "unreliable_predictions": 0,
    "overall_reliability": "High",
    "average_validation_r2": 0.7316051913493258,
    "average_test_r2": 0.732310805197865
  },
  "performance_estimates": {
    "cost_efficiency": {
      "cost_per_impression": 0.13687102588212274,
      "impressions_per_dollar": 7.3061482045237875,
      "impressions_per_day": 16.053434142389893,
      "cost_per_click": 9.046606477782834,
      "clicks_per_dollar": 0.11053868679441914,
      "clicks_per_day": 0.24288112955903746,
      "cost_per_action": 2.2906107869804067,
      "actions_per_dollar": 0.436564782495523,
      "actions_per_day": 0.9592419683382879
    },
    "reach_metrics": {
      "estimated_reach": 313.6956897309137,
      "reach_per_dollar": 3.5691852284777985,
      "reach_per_day": 7.842392243272843,
      "frequency": 2.0470072962953916,
      "reach_efficiency": 0.488518043785075
    },
    "engagement_metrics": {
      "click_through_rate": 0.015129543461214801,
      "ctr_percentage": 1.51295434612148,
      "ctr_benchmark": "Average",
      "conversion_rate": 3.9494297892958525,
      "conversion_rate_percentage": 394.94297892958525
    },
    "conversion_metrics": {
      "value_per_conversion": 0.0013950572576879627,
      "conversions_per_dollar": 0.436564782495523,
      "return_on_ad_spend": 0.0006090328682713462,
      "roas_percentage": 0.060903286827134626,
      "roas_benchmark": "Loss"
    },
    "quality_metrics": {
      "frequency_benchmark": "Optimal"
    }
  },
  "warnings": []
}
```

# 🎯 IMPRESSIONS
----------------------------------------
- [VAL LOG]  MAE=0.4822 RMSE=0.6734 R²=0.8738
- [VAL ORIG] MAE=90.55 RMSE=11,364.83 R²=-52.6284 WAPE=0.401 sMAPE=0.514
- [TEST LOG]  MAE=0.4819 RMSE=0.6725 R²=0.8745
- [TEST ORIG] MAE=80.21 RMSE=694.60 R²=0.7010 WAPE=0.353 sMAPE=0.514

Model: ensemble_stacking


# 🎯 CLICKS
----------------------------------------
- [VAL LOG]  MAE=0.3233 RMSE=0.5318 R²=0.7017
- [VAL ORIG] MAE=2.84 RMSE=21.67 R²=0.4134 WAPE=0.713 sMAPE=1.666
- [TEST LOG]  MAE=0.3239 RMSE=0.5323 R²=0.7031
- [TEST ORIG] MAE=2.85 RMSE=20.84 R²=0.3948 WAPE=0.710 sMAPE=1.663

Model: ensemble_stacking

# 🎯 ACTIONS
----------------------------------------
- [VAL LOG]  MAE=0.6864 RMSE=0.9810 R²=0.6709
- [VAL ORIG] MAE=39.89 RMSE=480.29 R²=0.3990 WAPE=0.734 sMAPE=1.609
- [TEST LOG]  MAE=0.6875 RMSE=0.9817 R²=0.6719
- [TEST ORIG] MAE=41.26 RMSE=548.36 R²=0.3551 WAPE=0.741 sMAPE=1.607

Model: ensemble_stacking

# 🎯 REACH
----------------------------------------
- [VAL LOG]  MAE=0.5019 RMSE=0.6923 R²=0.8680
- [VAL ORIG] MAE=81.87 RMSE=10,048.18 R²=-58.5088 WAPE=0.395 sMAPE=0.548
- [TEST LOG]  MAE=0.5015 RMSE=0.6917 R²=0.8686
- [TEST ORIG] MAE=73.52 RMSE=589.29 R²=0.7381 WAPE=0.352 sMAPE=0.547

Model: ensemble_stacking

# 🎯 CONVERSION_VALUE
----------------------------------------
- [VAL LOG]  MAE=0.2364 RMSE=0.9984 R²=0.5436
- [VAL ORIG] MAE=1,028.60 RMSE=17,576.92 R²=0.1263 WAPE=0.902 sMAPE=1.992
- [TEST LOG]  MAE=0.2368 RMSE=0.9992 R²=0.5433
- [TEST ORIG] MAE=1,018.52 RMSE=17,111.20 R²=0.1415 WAPE=0.896 sMAPE=1.992

Model: ensemble_stacking
