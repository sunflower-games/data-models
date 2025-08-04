create or replace view ml_models.prompt_finance_daily_screenshot_metrics as (
select
    inserted_time,
    user_id,
    cc_coin,
    sc_coin,
    bsc_coins as bsc_balance_sc_coins,
    usc_coins as usc_unredeemable_sc_coins,
    rsc_coins as rsc_redeemable_sc_coins,
    device_category,
    device_operating_system,
    geo_region,
    geo_country,
    city,
    platform,
    bonus_abuser_ind,
    total_redeem_amount,
    chb_score
from aggregations.daily_user_properties);

create or replace view ml_models.prompt_finance_chargebacks as (
select * from aggregations.fact_chargebacks_transactions);

create or replace view ml_models.prompt_finance_token_redeem_grade as
(select * except (full_name, payment_token_id) from aggregations.fact_payment_token);

create or replace view ml_models.prompt_finance_payment_transaction as (
select * except (email, full_name) from aggregations.fact_payment_transaction);

create or replace view ml_models.prompt_finance_purchase_transaction as (
SELECT * EXCEPT (purchase_transaction, transaction_banking_id, user_event_trigger_id) FROM aggregations.fact_purchase_transaction
);

create or replace view ml_models.prompt_finance_redeem_data as (
select * except (full_name, email) from aggregations.fact_redeem);

