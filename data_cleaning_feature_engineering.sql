-- Remove the view if it already exists
DROP VIEW IF EXISTS ig_post_metrics;

-- Create a view aggregating Instagram post data with engineered features
CREATE VIEW ig_post_metrics AS

-- Step 1: Extract core post-level fields from JSON
WITH posts AS (
    SELECT 
        post_id,
        json_extract(fields_json, '$.media_type') AS media_type, -- Type of post (e.g., image, video, reel)
        json_extract(fields_json, '$.like_count') AS like_count, -- Number of likes
        json_extract(fields_json, '$.comments_count') AS comments_count, -- Number of comments
        REPLACE(SUBSTR(json_extract(fields_json, '$.timestamp'), 1, 19), 'T', ' ') AS timestamp, -- Post timestamp (formatted)
        LENGTH(json_extract(fields_json, '$.caption')) AS caption_length, -- Total characters in caption
        LENGTH(json_extract(fields_json, '$.caption')) - LENGTH(
            REPLACE(json_extract(fields_json, '$.caption'), '#', '')) AS num_hashtags -- Count of hashtags
    FROM kgc_json
),

-- Step 2: Join post fields with engagement and reach metrics
posts_with_insights AS (
    SELECT
        p.*,
        i.reach, i.saved, i.shares, i.total_interactions,
        i.follows, i.profile_visits, i.profile_activity,
        i.views, i.ig_reels_avg_watch_time,
        i.ig_reels_video_view_total_time
    FROM 
        posts p
    LEFT JOIN 
        insights i ON i.id = p.post_id
),

-- Step 3: Add temporal features (posting patterns)
post_time_features AS (
    SELECT 
        post_id, 
        JULIANDAY(timestamp) - LAG(JULIANDAY(timestamp)) 
            OVER (ORDER BY timestamp) AS days_since_last_post, -- Gap since previous post
        JULIANDAY(timestamp) - JULIANDAY(FIRST_VALUE(timestamp)
            OVER(ORDER BY timestamp ASC)) AS days_since_first_post, -- Gap since first post in dataset        
        CASE 
            WHEN STRFTIME('%w', timestamp) = '0' THEN 'Sunday'
            WHEN STRFTIME('%w', timestamp) = '1' THEN 'Monday'
            WHEN STRFTIME('%w', timestamp) = '2' THEN 'Tuesday'
            WHEN STRFTIME('%w', timestamp) = '3' THEN 'Wednesday'
            WHEN STRFTIME('%w', timestamp) = '4' THEN 'Thursday'
            WHEN STRFTIME('%w', timestamp) = '5' THEN 'Friday'
            ELSE 'Saturday' 
        END AS day_of_week, -- Day of week posted
        CASE 
            WHEN CAST(STRFTIME('%H', timestamp) AS INT) BETWEEN 6 AND 11 THEN 'Morning'
            WHEN CAST(STRFTIME('%H', timestamp) AS INT) BETWEEN 12 AND 16 THEN 'Afternoon'
            WHEN CAST(STRFTIME('%H', timestamp) AS INT) BETWEEN 17 AND 20 THEN 'Evening'
            ELSE 'Night'
        END AS time_of_day -- Time of day posted
    FROM
        posts_with_insights
),

-- Step 4: Compute engagement-related metrics
post_engagement_metrics AS (
    SELECT 
        post_id, 
        like_count + comments_count AS engagement, -- Total engagements
        ROUND(
            (like_count + comments_count) * 100 / NULLIF(reach, 0),
            2) AS engagement_per_reach_pct, -- Engagement as % of reach
        ROUND(
            AVG(like_count + comments_count) 
            OVER (ORDER BY timestamp ROWS BETWEEN 2 PRECEDING AND CURRENT ROW))
            AS rolling_avg_engagement_3posts, -- Rolling average engagement
        ROUND(
            AVG(reach)
            OVER (ORDER BY timestamp ROWS BETWEEN 2 PRECEDING AND CURRENT ROW))
            AS rolling_avg_reach_3posts, -- Rolling average reach
        RANK() 
            OVER (PARTITION BY media_type ORDER BY (like_count + comments_count) DESC) 
            AS engagement_rank_by_media_type -- Group by media type, rank by post engagement
    FROM 
        posts_with_insights
)

-- Step 5: Combine all features into final view
SELECT 
    pwi.*,
    ptf.days_since_last_post,
    ptf.days_since_first_post,
    ptf.day_of_week,
    ptf.time_of_day,
    pem.engagement,
    pem.rolling_avg_engagement_3posts,
    pem.rolling_avg_reach_3posts,
    pem.engagement_per_reach_pct,
    pem.engagement_rank_by_media_type
FROM 
    posts_with_insights pwi
LEFT JOIN 
    post_time_features ptf ON pwi.post_id = ptf.post_id
LEFT JOIN 
    post_engagement_metrics pem ON pwi.post_id = pem.post_id
ORDER BY timestamp DESC