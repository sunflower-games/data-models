WITH min_dates AS (
   SELECT h_user_id,
          max(signup_date) signup_date,
          max(first_purchase_date) first_purchase_date,
   FROM looker.main_kpi_dataset
    where first_purchase_date is not null
   GROUP BY 1
)
SELECT d.h_user_id AS user_id,
       m.signup_date,
       m.first_purchase_date,
       d.signup_platform,
       ROUND(SUM(d.sc_total_bet_amount)) AS total_bet_amt,
       ROUND(SUM(d.sc_total_win_amount)) AS total_win_amt_sc,
       ROUND(SUM(d.cc_total_win_amount)) AS total_win_amt_cc,
       ROUND(SAFE_DIVIDE(SUM(d.sc_total_bet_amount), SUM(d.sc_total_bet_transactions)),3) AS avg_spin_amt,
       ROUND(SAFE_DIVIDE(SUM(d.sc_total_win_amount), SUM(d.sc_total_bet_amount)),3) AS rtp_ratio,
       round(SUM(d.sc_total_win_amount) - SUM(d.sc_total_bet_amount)) as ggr_amt,
       MIN(d.channel) AS channel,
       MIN(SUBSTR(d.email, STRPOS(d.email, '@') + 1)) AS signup_email_domain,
       count(d.h_user_id) as logins_first_96h,
       d.first_geo_region,
       SUM(d.sc_total_bet_transactions) AS total_bet_txn,
       max(d.purchase_amount) as max_purchase_amt_horly,
       MAX(d.redeem_amount) AS max_redeem_amt_hourly,
       SUM(d.redeem_amount) AS sum_redeem_amt,
       SUM(d.redeem_transactions) AS sum_redeem_txn,
       SUM(d.purchase_amount) AS sum_amt_total,
       SUM(d.total_transactions) AS cnt_txn_total,
       SUM(d.purchase_sc_coin) AS sum_sc_coin_total,
       SUM(d.did_sc_spin) AS sum_spin_sc,
       SUM(d.did_cc_spin) AS sum_spin_cc,
       MAX(d.eod_sc_balance) AS max_eod_sc_balance,
       SUM(d.bonus_wheel) AS sum_bonus_wheel,
       SUM(d.bonus_plinko) AS sum_bonus_plinko,
       SUM(d.offer_purchases_amount) AS sum_offer_amt,
       ROUND(SUM(d.bonus_amount)) AS sum_bonus_amt,
       SUM(d.daily_mission_reward) AS sum_daily_mission,
       SUM(d.leaderboard_reward) AS sum_leaderboard_reward
FROM looker.main_kpi_dataset d
JOIN min_dates m ON d.h_user_id = m.h_user_id
WHERE TIMESTAMP_DIFF(d.hour, m.first_purchase_date, HOUR) BETWEEN 0 AND 96
  AND m.first_purchase_date > '2023-11-01'
  AND m.first_purchase_date < '2024-08-20'
GROUP BY all
ORDER BY 1 DESC;

