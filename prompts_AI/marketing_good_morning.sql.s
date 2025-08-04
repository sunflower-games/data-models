WITH base_signup_conversion AS (
    SELECT
        date(p.signup_date) as su_date,
        LOWER(COALESCE(su.platform, 'unknown')) as platform,
        su.channel,
        su.channel_type,
        case when lower(COALESCE(su.platform, '')) = 'app' then COALESCE(su.channel, 'unknown')
             else concat(COALESCE(su.channel,'unknown'), '-', COALESCE(su.channel_type,'unknown'))
        end as channel_agg,
        count(su.user_id) as total_signups,
        count(case when date(p.signup_date) = date(p.first_purchase_date) then 1 end) as day0_conversions,
        ROUND(SAFE_DIVIDE(count(case when date(p.signup_date) = date(p.first_purchase_date) then 1 end), count(su.user_id)) * 100, 2) as conversion_rate
    FROM aggregations.user_profile p
    LEFT JOIN marketing.full_attribution_last_paid_before_signup su
        ON p.user_id = su.user_id
    WHERE date(p.signup_date) >= date_sub(current_date(), interval 8 day)
      AND date(p.signup_date) < current_date()
    GROUP BY date(p.signup_date),
             LOWER(COALESCE(su.platform, 'unknown')),
             su.channel,
             su.channel_type,
             case when lower(COALESCE(su.platform, '')) = 'app' then COALESCE(su.channel, 'unknown')
                  else concat(COALESCE(su.channel,'unknown'), '-', COALESCE(su.channel_type,'unknown'))
             end
),

base_roas AS (
    SELECT
        date(up.signup_date) as su_date,
        LOWER(COALESCE(ml.platform, 'unknown')) as platform,
        ml.channel,
        ml.channel_type,
        case when lower(COALESCE(ml.platform, '')) = 'app' then COALESCE(ml.channel, 'unknown')
             else concat(COALESCE(ml.channel,'unknown'), '-', COALESCE(ml.channel_type,'unknown'))
        end as channel_agg,
        count(up.user_id) as total_users,
        count(fp.user_id) as paid_users,
        COUNT(fp.transaction_id) as d0_purchase_count,
        SUM(fp.purchase_amount) as d0_total_amount
    FROM aggregations.user_profile up
    LEFT JOIN aggregations.fact_purchase fp
        ON up.user_id = fp.user_id
        AND DATE(fp.purchase_time) = date(up.signup_date)
    LEFT JOIN (
        select user_id, channel, channel_type, platform
        from marketing.full_attribution_last_paid_before_signup
    ) ml on ml.user_id = up.user_id
    WHERE date(up.signup_date) >= date_sub(current_date(), interval 8 day)
      AND date(up.signup_date) < current_date()
    GROUP BY date(up.signup_date),
             LOWER(COALESCE(ml.platform, 'unknown')),
             ml.channel,
             ml.channel_type,
             case when lower(COALESCE(ml.platform, '')) = 'app' then COALESCE(ml.channel, 'unknown')
                  else concat(COALESCE(ml.channel,'unknown'), '-', COALESCE(ml.channel_type,'unknown'))
             end
),

daily_costs AS (
    SELECT
        day,
        LOWER(COALESCE(platform, 'unknown')) as platform,
        channel,
        channel_type,
        case when lower(COALESCE(platform, '')) = 'app' then COALESCE(channel, 'unknown')
             else concat(COALESCE(channel,'unknown'), '-', COALESCE(channel_type,'unknown'))
        end as cost_channel_agg,
        SUM(cost) as cost
    FROM marketing.marketing_channels_cost
    WHERE day >= date_sub(current_date(), interval 8 day)
      AND day < current_date()
    GROUP BY day,
             LOWER(COALESCE(platform, 'unknown')),
             channel,
             channel_type,
             case when lower(COALESCE(platform, '')) = 'app' then COALESCE(channel, 'unknown')
                  else concat(COALESCE(channel,'unknown'), '-', COALESCE(channel_type,'unknown'))
             end
),

combined_daily_summary AS (
    SELECT
        COALESCE(sc.su_date, br.su_date) as su_date,
        COALESCE(sc.platform, br.platform) as platform,
        COALESCE(sc.channel, br.channel) as channel,
        COALESCE(sc.channel_type, br.channel_type) as channel_type,
        COALESCE(sc.channel_agg, br.channel_agg) as channel_agg,
        
        -- Signup/Conversion metrics
        COALESCE(sc.total_signups, 0) as total_signups,
        COALESCE(sc.day0_conversions, 0) as day0_conversions,
        COALESCE(sc.conversion_rate, 0) as conversion_rate,
        
        -- ROAS metrics
        COALESCE(br.total_users, 0) as total_users,
        COALESCE(br.paid_users, 0) as paid_users,
        COALESCE(br.d0_purchase_count, 0) as d0_purchase_count,
        COALESCE(br.d0_total_amount, 0) as d0_total_amount,
        
        -- Cost and ROAS
        COALESCE(SUM(c.cost), 0) as total_cost,
        CASE
            WHEN SUM(c.cost) > 0 THEN SAFE_DIVIDE(br.d0_total_amount, SUM(c.cost)) * 100
            ELSE 0
        END as roas_d0
        
    FROM base_signup_conversion sc
    FULL OUTER JOIN base_roas br
        ON sc.su_date = br.su_date
        AND sc.platform = br.platform
        AND COALESCE(sc.channel, '') = COALESCE(br.channel, '')
        AND COALESCE(sc.channel_type, '') = COALESCE(br.channel_type, '')
        AND sc.channel_agg = br.channel_agg
    LEFT JOIN daily_costs c
        ON c.day = COALESCE(sc.su_date, br.su_date)
        AND c.platform = COALESCE(sc.platform, br.platform)
        AND c.cost_channel_agg = COALESCE(sc.channel_agg, br.channel_agg)
    GROUP BY COALESCE(sc.su_date, br.su_date),
             COALESCE(sc.platform, br.platform),
             COALESCE(sc.channel, br.channel),
             COALESCE(sc.channel_type, br.channel_type),
             COALESCE(sc.channel_agg, br.channel_agg),
             sc.total_signups, sc.day0_conversions, sc.conversion_rate,
             br.total_users, br.paid_users, br.d0_purchase_count, br.d0_total_amount
),

overall_daily_costs AS (
    SELECT day, SUM(cost) as daily_total_cost
    FROM daily_costs
    GROUP BY day
),

platform_daily_costs AS (
    SELECT day, platform, SUM(cost) as daily_platform_cost
    FROM daily_costs
    GROUP BY day, platform
),

-- Identify top 5 channels from yesterday by signups per platform
top_5_channels AS (
    SELECT
        platform,
        channel_agg,
        channel,
        channel_type,
        SUM(total_signups) as total_signup_volume,
        ROW_NUMBER() OVER (PARTITION BY platform ORDER BY SUM(total_signups) DESC) AS platform_rank
    FROM combined_daily_summary
    WHERE su_date = DATE_SUB(CURRENT_DATE(), INTERVAL 1 DAY)
        AND channel_agg IS NOT NULL
        AND channel_agg != 'unknown'
    GROUP BY platform, channel_agg, channel, channel_type
    QUALIFY platform_rank <= 5
),

-- Identify top 12 channels for 7-day avg by signups per platform
top_12_channels AS (
    SELECT
        platform,
        channel_agg,
        channel,
        channel_type,
        SUM(total_signups) as total_signup_volume,
        ROW_NUMBER() OVER (PARTITION BY platform ORDER BY SUM(total_signups) DESC) AS platform_rank
    FROM combined_daily_summary
    WHERE su_date BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL 8 DAY)
        AND DATE_SUB(CURRENT_DATE(), INTERVAL 2 DAY)
        AND channel_agg IS NOT NULL
        AND channel_agg != 'unknown'
    GROUP BY platform, channel_agg, channel, channel_type
    QUALIFY platform_rank <= 12
),

yesterday_metrics AS (
    SELECT
        'Yesterday' AS period,
        'Overall' AS segment,
        '' AS platform,
        '' AS channel_agg,
        COALESCE(SUM(ds.total_signups), 0) AS total_signups,
        COALESCE(SUM(ds.day0_conversions), 0) AS day0_conversions,
        COALESCE(ROUND(SAFE_DIVIDE(SUM(ds.day0_conversions), SUM(ds.total_signups)) * 100, 2), 0) AS conversion_rate,
        COALESCE(SUM(ds.total_users), 0) AS total_users,
        COALESCE(SUM(ds.paid_users), 0) AS paid_users,
        COALESCE(SUM(ds.d0_purchase_count), 0) AS d0_purchase_count,
        COALESCE(SUM(ds.d0_total_amount), 0) as d0_total_amount,
        COALESCE(odc.daily_total_cost, 0) as total_cost,
        CASE
            WHEN odc.daily_total_cost > 0 THEN SAFE_DIVIDE(SUM(ds.d0_total_amount), odc.daily_total_cost) * 100
            ELSE 0
        END as roas_d0
    FROM combined_daily_summary ds
    LEFT JOIN overall_daily_costs odc ON ds.su_date = odc.day
    WHERE ds.su_date = DATE_SUB(CURRENT_DATE(), INTERVAL 1 DAY)
    GROUP BY odc.daily_total_cost

    UNION ALL

    SELECT
        'Yesterday' AS period,
        'Platform' AS segment,
        ds.platform,
        '' AS channel_agg,
        COALESCE(SUM(ds.total_signups), 0) AS total_signups,
        COALESCE(SUM(ds.day0_conversions), 0) AS day0_conversions,
        COALESCE(ROUND(SAFE_DIVIDE(SUM(ds.day0_conversions), SUM(ds.total_signups)) * 100, 2), 0) AS conversion_rate,
        COALESCE(SUM(ds.total_users), 0) AS total_users,
        COALESCE(SUM(ds.paid_users), 0) AS paid_users,
        COALESCE(SUM(ds.d0_purchase_count), 0) AS d0_purchase_count,
        COALESCE(SUM(ds.d0_total_amount), 0) as d0_total_amount,
        COALESCE(pdc.daily_platform_cost, 0) as total_cost,
        CASE
            WHEN pdc.daily_platform_cost > 0 THEN SAFE_DIVIDE(SUM(ds.d0_total_amount), pdc.daily_platform_cost) * 100
            ELSE 0
        END as roas_d0
    FROM combined_daily_summary ds
    LEFT JOIN platform_daily_costs pdc ON ds.su_date = pdc.day AND ds.platform = pdc.platform
    WHERE ds.su_date = DATE_SUB(CURRENT_DATE(), INTERVAL 1 DAY)
    GROUP BY ds.platform, pdc.daily_platform_cost
),

avg_7day_metrics AS (
    SELECT
        '7-Day Avg' AS period,
        'Overall' AS segment,
        '' AS platform,
        '' AS channel_agg,
        COALESCE(ROUND(AVG(daily_signups), 0), 0) AS total_signups,
        COALESCE(ROUND(AVG(daily_conversions), 0), 0) AS day0_conversions,
        COALESCE(ROUND(AVG(daily_conversion_rate), 2), 0) AS conversion_rate,
        COALESCE(ROUND(AVG(daily_users), 0), 0) AS total_users,
        COALESCE(ROUND(AVG(daily_paid_users), 0), 0) AS paid_users,
        COALESCE(ROUND(AVG(daily_purchase_count), 0), 0) AS d0_purchase_count,
        COALESCE(ROUND(AVG(daily_amount), 0), 0) as d0_total_amount,
        COALESCE(ROUND(AVG(daily_cost), 0), 0) as total_cost,
        COALESCE(ROUND(AVG(daily_roas), 3), 0) as roas_d0
    FROM (
        SELECT
            ds.su_date,
            SUM(ds.total_signups) AS daily_signups,
            SUM(ds.day0_conversions) AS daily_conversions,
            SAFE_DIVIDE(SUM(ds.day0_conversions), SUM(ds.total_signups)) * 100 AS daily_conversion_rate,
            SUM(ds.total_users) AS daily_users,
            SUM(ds.paid_users) AS daily_paid_users,
            SUM(ds.d0_purchase_count) AS daily_purchase_count,
            SUM(ds.d0_total_amount) as daily_amount,
            COALESCE(odc.daily_total_cost, 0) as daily_cost,
            CASE
                WHEN odc.daily_total_cost > 0 THEN SAFE_DIVIDE(SUM(ds.d0_total_amount), odc.daily_total_cost) * 100
                ELSE 0
            END as daily_roas
        FROM combined_daily_summary ds
        LEFT JOIN overall_daily_costs odc ON ds.su_date = odc.day
        WHERE ds.su_date BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL 8 DAY)
            AND DATE_SUB(CURRENT_DATE(), INTERVAL 2 DAY)
        GROUP BY ds.su_date, odc.daily_total_cost
    )

    UNION ALL

    SELECT
        '7-Day Avg' AS period,
        'Platform' AS segment,
        platform,
        '' AS channel_agg,
        COALESCE(ROUND(AVG(daily_signups), 0), 0) AS total_signups,
        COALESCE(ROUND(AVG(daily_conversions), 0), 0) AS day0_conversions,
        COALESCE(ROUND(AVG(daily_conversion_rate), 2), 0) AS conversion_rate,
        COALESCE(ROUND(AVG(daily_users), 0), 0) AS total_users,
        COALESCE(ROUND(AVG(daily_paid_users), 0), 0) AS paid_users,
        COALESCE(ROUND(AVG(daily_purchase_count), 0), 0) AS d0_purchase_count,
        COALESCE(ROUND(AVG(daily_amount), 0), 0) as d0_total_amount,
        COALESCE(ROUND(AVG(daily_cost), 0), 0) as total_cost,
        COALESCE(ROUND(AVG(daily_roas), 3), 0) as roas_d0
    FROM (
        SELECT
            ds.su_date,
            ds.platform,
            SUM(ds.total_signups) AS daily_signups,
            SUM(ds.day0_conversions) AS daily_conversions,
            SAFE_DIVIDE(SUM(ds.day0_conversions), SUM(ds.total_signups)) * 100 AS daily_conversion_rate,
            SUM(ds.total_users) AS daily_users,
            SUM(ds.paid_users) AS daily_paid_users,
            SUM(ds.d0_purchase_count) AS daily_purchase_count,
            SUM(ds.d0_total_amount) as daily_amount,
            COALESCE(pdc.daily_platform_cost, 0) as daily_cost,
            CASE
                WHEN pdc.daily_platform_cost > 0 THEN SAFE_DIVIDE(SUM(ds.d0_total_amount), pdc.daily_platform_cost) * 100
                ELSE 0
            END as daily_roas
        FROM combined_daily_summary ds
        LEFT JOIN platform_daily_costs pdc ON ds.su_date = pdc.day AND ds.platform = pdc.platform
        WHERE ds.su_date BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL 8 DAY)
            AND DATE_SUB(CURRENT_DATE(), INTERVAL 2 DAY)
        GROUP BY ds.su_date, ds.platform, pdc.daily_platform_cost
    )
    GROUP BY platform
),

top_channels_yesterday AS (
    SELECT
        'Yesterday' AS period,
        'Top 5 Channels' AS segment,
        platform,
        channel_agg,
        COALESCE(total_signups, 0) as total_signups,
        COALESCE(day0_conversions, 0) as day0_conversions,
        COALESCE(ROUND(conversion_rate, 2), 0) AS conversion_rate,
        COALESCE(total_users, 0) as total_users,
        COALESCE(paid_users, 0) as paid_users,
        COALESCE(d0_purchase_count, 0) AS d0_purchase_count,
        COALESCE(d0_total_amount, 0) as d0_total_amount,
        COALESCE(total_cost, 0) as total_cost,
        COALESCE(roas_d0, 0) as roas_d0,
        ROW_NUMBER() OVER (PARTITION BY platform ORDER BY total_signups DESC) AS rank
    FROM combined_daily_summary
    WHERE su_date = DATE_SUB(CURRENT_DATE(), INTERVAL 1 DAY)
        AND channel_agg IS NOT NULL
        AND channel_agg != 'unknown'
    QUALIFY rank <= 5
),

top_channels_7day_avg AS (
    SELECT
        '7-Day Avg' AS period,
        'Top 12 Channels' AS segment,
        t12.platform,
        t12.channel_agg,
        COALESCE(ROUND(AVG(daily_signups), 0), 0) AS total_signups,
        COALESCE(ROUND(AVG(daily_conversions), 0), 0) AS day0_conversions,
        COALESCE(ROUND(AVG(daily_conversion_rate), 2), 0) AS conversion_rate,
        COALESCE(ROUND(AVG(daily_users), 0), 0) AS total_users,
        COALESCE(ROUND(AVG(daily_paid_users), 0), 0) AS paid_users,
        COALESCE(ROUND(AVG(daily_purchase_count), 0), 0) AS d0_purchase_count,
        COALESCE(ROUND(AVG(daily_amount), 0), 0) as d0_total_amount,
        COALESCE(ROUND(AVG(daily_cost), 0), 0) as total_cost,
        COALESCE(ROUND(AVG(daily_roas), 3), 0) as roas_d0
    FROM top_12_channels t12
    JOIN (
        SELECT
            ds.su_date,
            ds.platform,
            ds.channel,
            ds.channel_type,
            ds.channel_agg,
            SUM(ds.total_signups) AS daily_signups,
            SUM(ds.day0_conversions) AS daily_conversions,
            SAFE_DIVIDE(SUM(ds.day0_conversions), SUM(ds.total_signups)) * 100 AS daily_conversion_rate,
            SUM(ds.total_users) AS daily_users,
            SUM(ds.paid_users) AS daily_paid_users,
            SUM(ds.d0_purchase_count) AS daily_purchase_count,
            SUM(ds.d0_total_amount) as daily_amount,
            SUM(ds.total_cost) as daily_cost,
            AVG(ds.roas_d0) as daily_roas
        FROM combined_daily_summary ds
        WHERE ds.su_date BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL 8 DAY)
            AND DATE_SUB(CURRENT_DATE(), INTERVAL 2 DAY)
        GROUP BY ds.su_date, ds.platform, ds.channel, ds.channel_type, ds.channel_agg
    ) daily_channel_data
    ON t12.platform = daily_channel_data.platform
    AND t12.channel = daily_channel_data.channel
    AND COALESCE(t12.channel_type, '') = COALESCE(daily_channel_data.channel_type, '')
    GROUP BY t12.platform, t12.channel_agg
)

SELECT
    period,
    segment,
    platform,
    channel_agg,
    COALESCE(total_signups, 0) as total_signups,
    COALESCE(day0_conversions, 0) as day0_conversions,
    COALESCE(conversion_rate, 0) as conversion_rate,
    COALESCE(total_users, 0) as total_users,
    COALESCE(paid_users, 0) as paid_users,
    COALESCE(d0_purchase_count, 0) as d0_purchase_count,
    COALESCE(d0_total_amount, 0) as d0_total_amount,
    COALESCE(total_cost, 0) as total_cost,
    COALESCE(roas_d0, 0) as roas_d0
FROM yesterday_metrics

UNION ALL

SELECT
    period,
    segment,
    platform,
    channel_agg,
    COALESCE(total_signups, 0) as total_signups,
    COALESCE(day0_conversions, 0) as day0_conversions,
    COALESCE(conversion_rate, 0) as conversion_rate,
    COALESCE(total_users, 0) as total_users,
    COALESCE(paid_users, 0) as paid_users,
    COALESCE(d0_purchase_count, 0) as d0_purchase_count,
    COALESCE(d0_total_amount, 0) as d0_total_amount,
    COALESCE(total_cost, 0) as total_cost,
    COALESCE(roas_d0, 0) as roas_d0
FROM avg_7day_metrics

UNION ALL

SELECT
    period,
    segment,
    platform,
    channel_agg,
    COALESCE(total_signups, 0) as total_signups,
    COALESCE(day0_conversions, 0) as day0_conversions,
    COALESCE(conversion_rate, 0) as conversion_rate,
    COALESCE(total_users, 0) as total_users,
    COALESCE(paid_users, 0) as paid_users,
    COALESCE(d0_purchase_count, 0) as d0_purchase_count,
    COALESCE(d0_total_amount, 0) as d0_total_amount,
    COALESCE(total_cost, 0) as total_cost,
    COALESCE(roas_d0, 0) as roas_d0
FROM top_channels_yesterday

UNION ALL

SELECT
    period,
    segment,
    platform,
    channel_agg,
    COALESCE(total_signups, 0) as total_signups,
    COALESCE(day0_conversions, 0) as day0_conversions,
    COALESCE(conversion_rate, 0) as conversion_rate,
    COALESCE(total_users, 0) as total_users,
    COALESCE(paid_users, 0) as paid_users,
    COALESCE(d0_purchase_count, 0) as d0_purchase_count,
    COALESCE(d0_total_amount, 0) as d0_total_amount,
    COALESCE(total_cost, 0) as total_cost,
    COALESCE(roas_d0, 0) as roas_d0
FROM top_channels_7day_avg

ORDER BY
    CASE WHEN period = 'Yesterday' THEN 1
         WHEN period = '7-Day Avg' THEN 2
         ELSE 3 END,
    CASE WHEN segment = 'Overall' THEN 1
         WHEN segment = 'Platform' THEN 2
         ELSE 3 END,
    platform,
    total_signups DESC;