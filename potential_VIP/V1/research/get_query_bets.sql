select
    user_id,
    max(max_bet_amount) as max_bet_amt,
    min(min_bet_amount) as min_bet_amount,
    max(max_win_amount) as max_win_amt,
    avg(rounds_bet) as avg_rounds_bet,
    avg(bet_amount) as avg_bet_amt,
    max(unique_games_day) as max_daily_games_unique, 
    sum(bonus_promotion_bet_amount) as sum_bonus_promotion_bet_amount
from(
SELECT
        b.day,
        kpi.first_purchase_date,
        b.user_id,
        b.bet_amount,
        b.min_bet_amount,
        b.max_bet_amount,
        b.max_win_amount,
        b.rounds_bet,
        b.unique_games_day,
        b.bonus_promotion_bet_amount, 
        b.bonus_promotion_win_amount
FROM aggregations.fact_daily_bets b
JOIN
    (
    select h_user_id,
            max(first_bet_date) first_bet_date,
            max(first_purchase_Date) first_purchase_Date
    from looker.main_kpi_dataset kpi
    where date(signup_date) >= '2023-10-01'
        group by 1
        ) kpi
ON (
    b.user_id = kpi.h_user_id
and kpi.first_bet_date >= kpi.first_purchase_Date
AND DATE(b.day) <= DATE_ADD(DATE(kpi.first_purchase_Date), INTERVAL 3 DAY)
)
where date(b.day) >= '2023-10-01')
group by 1;