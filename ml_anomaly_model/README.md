# ML Anomaly Detection Models

**Created:** 2025-07-02 16:58:15
**Algorithm:** Isolation Forest with contamination rates
**Features:** 14 ML features
**Performance:** NEEDS REVIEW (1.673% anomaly rate)

## Files:
- `config.json` - All configuration and contamination rates
- `models/` - Model and scaler files for each platform
- `usage.py` - Simple usage example

## Features:
- success_rate
- is_business_hours
- hour_sin
- hour_cos
- day_sin
- day_cos
- is_weekend
- success_volatility
- volume_volatility
- success_trend
- success_warning
- volume_warning
- success_relative
- volume_relative

## Contamination Rates:
- APP_IOS_WORLDPAY: 0.0040
- APP_IOS_NUVEI: 0.0050
- APP_IOS_PAYSAFE: 0.0035
- WEB_PAYSAFE: 0.0050
- WEB_WORLDPAY: 0.0035
- WEB_NUVEI: 0.0040

## Platforms: 6 total
- APP_IOS_NUVEI
- APP_IOS_PAYSAFE
- APP_IOS_WORLDPAY
- WEB_NUVEI
- WEB_PAYSAFE
- WEB_WORLDPAY

**Ready for production deployment!**
