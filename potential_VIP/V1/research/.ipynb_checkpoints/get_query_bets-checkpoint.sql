select
    user_id,
    max(days_of_activity) as days_of_activity,
    max(first_purchase_date) as first_purchase_date,
    max(max_bet_amount) as max_bet_amt,
    min(min_bet_amount) as min_bet_amount,
    max(max_win_amount) as max_win_amt,
    avg(rounds_bet) as avg_rounds_bet,
    avg(bet_amount) as avg_bet_amt,
    max(unique_games_day) as max_daily_games_unique,
    sum(bonus_promotion_win_amount) as sum_bonus_promotion_bet_amount
from(
SELECT
        b.day,
        kpi.first_purchase_date,
        kpi.days_of_activity,
        b.first_day_bet,
        b.user_id,
        b.bet_amount,
        b.min_bet_amount,
        b.max_bet_amount,
        b.max_win_amount,
        b.rounds_bet,
        b.unique_games_day,
        b.bonus_promotion_win_amount
FROM aggregations.fact_daily_bets  b
JOIN
    (
    select h_user_id,
            max(first_bet_date) first_bet_date,
            max(first_purchase_Date) first_purchase_Date,
            count(DISTINCT DATE(hour)) AS days_of_activity
    from looker.main_kpi_dataset kpi
    where hour >= first_purchase_date
    AND  hour < TIMESTAMP_ADD(first_purchase_date, INTERVAL 96 HOUR)
    group by 1
        ) kpi
ON (
    b.user_id = kpi.h_user_id
)
WHERE date(day) >= date(first_purchase_Date)
and date(first_purchase_date) >= DATE_SUB(date(day), INTERVAL 4 DAY)
and date(day) >= '2023-10-01'
)
group by 1