# Rolling 7-Day ML Models

Export: 2025-08-04 13:04
Performance: GOOD (0.93%)

## Files:
- `models/`: 12 models + scalers
- `config.json`: Configuration
- `test_results_with_scores.csv`: Test data with features + anomaly scores
- `usage.py`: Simple usage example

## 12 Rolling Features:
🕐 Hour-of-Week: success_vs_rolling_expected, volume_vs_rolling_expected
📊 Rolling 7-Day: success_vs_rolling_7day, volume_vs_rolling_7day  
⏰ Short-Term: success_vs_1h_avg, volume_vs_3h_avg, success_ma_3h
📅 Week-over-Week: success_vs_7days_ago
⚡ Volatility: success_volatility_3h, volume_volatility_3h, success_trend_3h, volume_momentum_1h

## Platforms:
- APP_IOS_NUVEI
- APP_IOS_PAYSAFE
- APP_IOS_WORLDPAY
- NUVEI_APPLEPAY
- NUVEI_CARD
- PAYSAFE_APPLEPAY
- PAYSAFE_CARD
- WEB_NUVEI
- WEB_PAYSAFE
- WEB_WORLDPAY
- WORLDPAY_APPLEPAY
- WORLDPAY_CARD
