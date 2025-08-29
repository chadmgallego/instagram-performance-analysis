# Instagram Metrics Analysis
This project analyzes Instagram post performance for Tokyo Student Mobilization to explore how factors such as hashtags, media type, and posting behavior influence engagement and reach. The goal is to identify content patterns that perform well and highlight opportunities for growth through optimal posting strategies. 

## Key Questions
- How do engagement and reach change over time since the first post?  
- Do posts with hashtags perform better than posts without hashtags?  
- What role does media type (image, video, carousel) play in engagement and reach?  
- Do temporal patterns (day of week, time of day) for post publishing impact performance?  

## Methods Used
- Python (pandas, matplotlib, seaborn, OOP)
- Instagram API for data collection
- SQLite for data querying and feature engineering
- Descriptive statistics (mean, median, interquartile range) 
- Exploratory Data Analysis (EDA)
- Feature correlation analysis 

## Key Insights
- VIDEO posts drive higher engagement compared to IMAGE or CAROUSEL_ALBUM posts.
- Posting on weekends tends to increase engagement.
- Hashtags positively impact engagement and reach.
- Reach is gradually increasing over time, but engagement does not exhibit a corresponding upward trend.
- Posting time has minimal impact on post performance.


## Data Source Disclosure
The dataset is derived from Instagram Graph API. Raw files are not shared publicly to respect platform policies and data privacy. All visualizations and findings are based on internally collected post-level data.  

## Files
- `instagram_analysis.ipynb`: Full Jupyter notebook with code, visualizations, and insights
- `ig_post_metrics.csv`: Processed dataset of Instagram post performance metrics
