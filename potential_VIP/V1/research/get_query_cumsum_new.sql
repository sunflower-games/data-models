WITH min_dates AS (
   SELECT user_id,
          purchase_time as first_purchase_time,
          purchase_amount AS first_purchase_amt
   FROM aggregations.fact_purchase
   WHERE purchase_time IS NOT NULL
   QUALIFY ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY purchase_time) = 1
),
purchase AS (
   SELECT p.user_id AS user_id,
          p.signup_date,
          m.first_purchase_time,
          m.first_purchase_amt,
          CAST(TIMESTAMP_DIFF(p.purchase_time, m.first_purchase_time, HOUR) / 24 AS INT64) AS days_since_first_purchase,
          round(SUM(p.purchase_amount)) AS daily_total
   FROM min_dates m
   LEFT JOIN aggregations.fact_purchase p ON p.user_id = m.user_id
   WHERE TIMESTAMP_DIFF(p.purchase_time, m.first_purchase_time, HOUR) < 740
     AND m.first_purchase_time > '2023-11-01'
     AND m.first_purchase_time < '2024-08-20'
   GROUP BY 1,2,3,4,5
)
SELECT *,
       round(sum(daily_total) over(partition by user_id order by days_since_first_purchase)) cum_sum
FROM purchase
order by  user_id, days_since_first_purchase;