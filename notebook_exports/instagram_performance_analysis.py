#!/usr/bin/env python
# coding: utf-8

# # Instagram Performance Analysis: Engagement & Reach
# > *This project examines Instagram post performance for Tokyo Student Mobilization, with a focus on reach and engagement. By analyzing post-level metrics alongside engineered features, such as posting time, media type, caption length, and hashtag usage, we aim to uncover patterns and relationships that drive audience interaction. Using trend analysis, distribution plots, and correlation studies, this analysis investigates how different posting strategies impact both content exposure and follower engagement.*
# >
# > **Key Questions**
# > - How do reach and engagement evolve over time?
# > - How do posting features, like media type and caption length, relate to engagement and reach?
# > - Which features or patterns are most strongly associated with high engagement or high reach posts?
# >- Do temporal patterns, such as the time of day or day of the week a post is published, influence overall reach or engagement trends?

# In[1]:


#Install the 'ipython-sql' and 'prettytable' libraries using pip
get_ipython().system('pip install ipython-sql prettytable')

# Import necessary Python modules for API calls, JSON handling, SQLite, datetime, and data analysis
import requests, json, sqlite3, sys
from datetime import datetime
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import RobustScaler
import prettytable 
get_ipython().run_line_magic('matplotlib', 'inline')
prettytable.DEFAULT = 'DEFAULT'

# Load SQL magic extension to run SQL queries directly in notebook cells
get_ipython().run_line_magic('load_ext', 'sql')


# ## Instagram Data Collection & Preparation
# > In this section, we set up a object-oriented pipeline to **collect raw post data and insights from the Instagram Graph API**, store them in a local SQLite database, and then **transform the JSON responses into a structured dataset** for analysis.

# ### Data Collection Class: `InstaCollector`
# > 
# >This class manages all **data ingestion** tasks, including:
# >- Initializing API connection parameters and SQLite database
# >- Creating local SQLite tables to store raw JSON post and account data
# >- Fetching post fields and handling pagination to store data locally
# >- Fetching detailed post insights (metrics) and updating the local database
# >- Closing the database connection upon completion

# In[8]:


# Define InstaCollector class to:
# - Initialize connection details for Instagram Graph API
# - Connect and create SQLite tables
# - Fetch post fields and store raw JSON data
# - Fetch post insights (metrics) and update SQLite database
# - Close database connection

class InstaCollector:
    def __init__(self, ACCESS_TOKEN, IG_BUSINESS_ID, base_url, endpoints, queries):
        self.access_token = ACCESS_TOKEN
        self.ig_business_id = IG_BUSINESS_ID
        self.base_url = base_url
        self.endpoints = endpoints
        self.queries = queries

    def db_initializer(self, db):
        # Connect locally to SQLite database 
        self.db = db
        self.conn = sqlite3.connect(self.db)
        self.cursor = self.conn.cursor()

    def db_tables(self):
        # Create tables for storing post JSON and account info
        self.cursor.executescript('''
        	DROP TABLE IF EXISTS kgc_json;
    		DROP TABLE IF EXISTS kgc_account;

    		CREATE TABLE IF NOT EXISTS kgc_json (
    		post_id INTEGER NOT NULL PRIMARY KEY UNIQUE,
    		fields_json TEXT, -- post fields 
    		insights_json TEXT ); -- feed/reels insights 

    		CREATE TABLE IF NOT EXISTS kgc_account (
    		id INTEGER NOT NULL PRIMARY KEY UNIQUE, -- autoincremented index 
    		followers_count INTEGER,
    		media_count INTEGER,
    		follows_count INTEGER,
    		date_collected DATETIME UNIQUE ) -- logic key
    		''')
        self.conn.commit()

    def get_fields(self):
        # Fetch post fields using Instagram API, handle pagination,
        # and store raw JSON in SQLite kgc_json table
        self.post_ids = list()
        while True: # execute the while loop for the sake of pagination (there are many pages of data returned by the API call)
            try: # the page url will change with each pagination
                fields_response = requests.get(next_page, params = self.queries['post_fields']) # this will only work after the first page
            except:
                fields_url = base_url + self.endpoints['instagram_business_id'] + '/' + self.endpoints['media'] # this will only work on the first page
                fields_response = requests.get(fields_url, params = self.queries['post_fields'])
            try:
                fields_data = json.loads(fields_response.text)['data'] # extracting raw data from the json formatted string 
            except: 
                print('access token has expired!')
                sys.exit()
            self.post_ids.extend(post['id'] for post in fields_data) # append post ids for this page 

            for post in fields_data:
                self.cursor.execute('''INSERT OR IGNORE INTO kgc_json (post_id, fields_json) VALUES (?, ?)''', (int(post['id']), json.dumps(post))) #loads raw json data into SQL table 
                self.conn.commit()
	        	# the reason why we have to call the json.dumps(post) method is because post is a dictionary and we want to dump it into a json formatted string 
	        	# for storing in SQL, which we will later convert back into a python dict by using json.loads(post) when we parse the data

            if 'paging' in fields_response.json() and 'next' in fields_response.json()['paging']: #this is where we check for the existence of a 'next' page
                next_page = fields_response.json()['paging']['next'] # the url for next page is stored at the end of the response 
            else: break # once all pages have been combed we break the while loop and return all of the data

        # Fetch and store account-level fields
        account_url = self.base_url + '/' + self.endpoints['instagram_business_id']
        account_response = requests.get(account_url, params = self.queries['account_fields'])
        account_data = json.loads(account_response.text)

        self.cursor.execute('''INSERT OR IGNORE INTO kgc_account (followers_count, media_count, follows_count, date_collected) VALUES (?, ?, ?, ?)''', 
			(int(account_data['followers_count']), int(account_data['media_count']), int(account_data['follows_count']), str(datetime.now())))
        self.conn.commit()

    def get_insights(self):
        # Retrieve all posts from SQLite and fetch insights (metrics) from Instagram API
        self.cursor.execute('''SELECT fields_json FROM kgc_json''')
        posts = self.cursor.fetchall() #returns a list

        # Iterate through all posts and their respective JSON strings 
        for post in posts:
            post = json.loads(post[0]) # each post is a single valued tuple containing json formatted string
            insights_url = self.base_url + post['id'] + '/' + self.endpoints['insights'] # the base url will be the same for both feed and reels 
            if post['media_type'] == 'VIDEO':
                insights_queries = self.queries['reels_insights'] # establish API queries specific to reels 
            else: 
                insights_queries = self.queries['feed_insights'] # establish API queries specific to feed (non-reels)

            insights_response = requests.get(insights_url, params = insights_queries) 
            insights_data = insights_response.text # no need to use json.loads() method because we will parse through the json string later 
            if 'error' in insights_data: continue # many of the posts were made before the account was converted to creator type so they will not return any insights

            self.cursor.execute('''UPDATE kgc_json SET insights_json = ? WHERE post_id = ?''', (insights_data, post['id']))
            self.conn.commit()

    def db_closer(self):
        # Close the database connection 
        self.conn.close()


# ### Data Loading Class: `InstaLoader`
# >
# >This class manages **data transformation and loading** tasks, including:
# >- Retrieving stored JSON data from SQLite
# >- Converting JSON strings into Python dictionaries
# >- Flattening insights JSON into a structured pandas DataFrame
# >- Loading the cleaned dataset back into SQLite as a table

# In[9]:


# Define InstaLoader class to:
# - Load raw JSON fields and insights from SQLite
# - Convert JSON strings to Python dictionaries
# - Flatten insights JSON into structured DataFrame and save back to SQLite

class InstaLoader:
    def __init__(self, db):
        self.db = db
        self.conn = sqlite3.connect(self.db)
        self.cursor = self.conn.cursor()

    def fields_loader(self):
        # Load posts' fields_json where insights exist, convert JSON strings to dicts
        self.cursor.execute('''
            SELECT fields_json FROM kgc_json WHERE insights_json NOT NULL''')
		# cursor.fetchall() returns a list of single-valued tuples
  			# so we take the 0th value of each tuple in the list and convert it to a python dictionary 
        self.fields_dicts = [json.loads(post[0]) for post in self.cursor.fetchall()] # a list of dictionaries 

    def insights_loader(self):
        # Define metrics to extract depending on media type
        insights_metrics = {
            'feed' : ['reach','saved','shares','total_interactions','follows','profile_visits','profile_activity','views'],
            'reels' : ['reach','saved','shares','total_interactions','views','ig_reels_video_view_total_time','ig_reels_avg_watch_time']
        }

        # Load insights_json where available and convert to dicts
        self.cursor.execute('''SELECT insights_json FROM kgc_json WHERE insights_json NOT NULL''')
		# cursor.fetchall() returns a list of single-valued tuples
  			# so we take the 0th value of each tuple in the list and convert it to a python dictionary 
        insights_raw = [json.loads(post[0]) for post in self.cursor.fetchall()] 
        self.insights_dicts = list()

        # Map metrics to each post based on media type
        for p in range(len(insights_raw)):
            post_dict = {'id' : self.fields_dicts[p]['id']}
            if self.fields_dicts[p]['media_type'] == 'VIDEO': metrics = insights_metrics['reels']
            else: metrics = insights_metrics['feed']
            for m in range(len(metrics)):
                post_dict[metrics[m]] = insights_raw[p]['data'][m]['values'][0]['value']
            self.insights_dicts.append(post_dict)

        # Convert to pandas DataFrame and save as SQL table 'insights'
        insights_df = pd.DataFrame(self.insights_dicts)
        insights_df.to_sql(name = 'insights', con = conn, if_exists = 'replace', index = False)
        print("Successfully loaded insights table to SQLite!")

    def db_closer(self):
        # Close the database connection 
        self.conn.close()


# ### API Credentials & Queries
# >
# >API credentials are securely stored in a local graph_api.txt file (not uploaded to GitHub).
# The following queries are defined for use with the Graph API:
# >- **Post fields**: post metadata (caption, timestamp, likes, comments, etc.)
# >- **Feed insights**: reach, saves, shares, total interactions, follows, profile activity, views
# >- **Reels insights**: reach, saves, shares, total interactions, views, average watch time, total watch time
# >- **Account fields**: follower count, media count, follows count

# In[10]:


# Load API credentials from local JSON file
with open('graph_api.txt', 'r') as file:
	info = json.loads(file.read())
	ACCESS_TOKEN = info['ACCESS_TOKEN']
	IG_BUSINESS_ID = info['IG_BUSINESS_ID']
	APP_ID = info['APP_ID']
	APP_SECRET = info['APP_SECRET']

# Define API queries for posts, insights, and account data
insta_queries = {
  'post_fields' : {
 	'access_token': ACCESS_TOKEN, # the access token must be used as a field for all API calls
    'fields' : 'id,media_type,media_url,username,comments_count,like_count,timestamp,caption,comments',
  },
  'feed_insights' : {
    'access_token': ACCESS_TOKEN,
    'metric' : 'reach,saved,shares,total_interactions,follows,profile_visits,profile_activity,views'
  },
  'reels_insights' : {
    'access_token': ACCESS_TOKEN,
    'metric' : 'reach,saved,shares,total_interactions,views,ig_reels_video_view_total_time,ig_reels_avg_watch_time'
  },
  'account_fields' : {
    'access_token' : ACCESS_TOKEN,
    'fields' : 'followers_count,media_count,follows_count'
  }
}

# Base URL and endpoints for API calls
base_url = "https://graph.facebook.com/v22.0/" # used for all API calls
endpoints = {
    'instagram_business_id': IG_BUSINESS_ID,
    'media': 'media',
    'insights': 'insights'
}


# ### Initialize Collector & Gather Data
# >Set up the database, fetch post data and insights, and store them in SQLite.

# In[ ]:


# Initialize the Instagram data collector and database
insta_collect = InstaCollector(ACCESS_TOKEN, IG_BUSINESS_ID, base_url, endpoints, insta_queries)
insta_collect.db_initializer(db = 'ig_data2.db')

# Uncomment if you want to reset tables (run only once or when resetting)
# insta_collect.db_tables()

# Collect posts data
insta_collect.get_fields()

# Collect post insights (metrics)
insta_collect.get_insights()

# Close DB connection
insta_collect.db_closer()


# ### Transform & Save Insights
# >Parse and flatten raw JSON into a clean DataFrame, then save to SQLite.

# In[ ]:


# Initialize data loader to transform raw JSON into structured data
insta_load = InstaLoader('ig_data2.db')

# Load posts fields from SQLite JSON to Python dicts
insta_load.fields_loader()

# Load insights JSON, flatten, and save to SQLite table 'insights'
insta_load.insights_loader()

# Close the DB connection
insta_load.db_closer()


# ## Extracting & Engineering Instagram Post Metrics
# > - Connect to SQLite database
# > - Connect to the local database `ig_data2.db` containing Instagram post JSON and insights data.
# > - Initialize SQL Magic (`%sql`) to run queries directly from the notebook.

# In[11]:


# Connect to the local SQLite database containing Instagram insights and post JSON data
db = 'ig_data2.db'
conn = sqlite3.connect(db)
cursor = conn.cursor()

# Initialize SQL Magic with database connection
get_ipython().run_line_magic('sql', 'sqlite:///ig_data2.db')


# ### Create SQL View
# > **Step 1: Extract core post-level fields from JSON**
# > - `media_type`: Type of post (image, video, reel)
# > - `like_count`, `comments_count`: Engagement metrics
# > - `timestamp`: Post timestamp (formatted)
# > - `caption_length`: Number of characters in caption
# > - `num_hashtags`: Number of hashtags in caption
# >
# > **Step 2: Join post fields with engagement and reach metrics**
# > - Merge `posts` with `insights` table to include `reach`, `saved`, `shares`, `profile_activity`, etc.
# >
# > **Step 3: Add temporal features**
# > - `days_since_last_post`: Gap since previous post
# > - `days_since_first_post`: Gap since first post
# > - `day_of_week` and `time_of_day`: Categorical time features for posting patterns
# > 
# > **Step 4: Compute engagement-related metrics**
# > - `engagement`: Total likes + comments
# > - `engagement_per_reach_pct`: Engagement relative to reach
# > - `rolling_avg_engagement_3posts`: Rolling average engagement over last 3 posts
# > - `engagement_rank_by_media_type`: Rank posts by engagement within each media type
# > 
# > **Step 5: Combine all features**
# > - Join post-level fields, temporal features, and engagement metrics into a single SQL view
# > - Order by most recent posts (`timestamp DESC`)

# In[ ]:


get_ipython().run_cell_magic('sql', '', "\n-- Remove the view if it already exists\nDROP VIEW IF EXISTS ig_post_metrics;\n\n-- Create a view aggregating Instagram post data with engineered features\nCREATE VIEW ig_post_metrics AS\n\n-- Step 1: Extract core post-level fields from JSON\nWITH posts AS (\n    SELECT \n        post_id,\n        json_extract(fields_json, '$.media_type') AS media_type, -- Type of post (e.g., image, video, reel)\n        json_extract(fields_json, '$.like_count') AS like_count, -- Number of likes\n        json_extract(fields_json, '$.comments_count') AS comments_count, -- Number of comments\n        REPLACE(SUBSTR(json_extract(fields_json, '$.timestamp'), 1, 19), 'T', ' ') AS timestamp, -- Post timestamp (formatted)\n        LENGTH(json_extract(fields_json, '$.caption')) AS caption_length, -- Total characters in caption\n        LENGTH(json_extract(fields_json, '$.caption')) - LENGTH(\n            REPLACE(json_extract(fields_json, '$.caption'), '#', '')) AS num_hashtags -- Count of hashtags\n    FROM kgc_json\n),\n\n-- Step 2: Join post fields with engagement and reach metrics\nposts_with_insights AS (\n    SELECT\n        p.*,\n        i.reach, i.saved, i.shares, i.total_interactions,\n        i.follows, i.profile_visits, i.profile_activity,\n        i.views, i.ig_reels_avg_watch_time,\n        i.ig_reels_video_view_total_time\n    FROM \n        posts p\n    LEFT JOIN \n        insights i ON i.id = p.post_id\n),\n\n-- Step 3: Add temporal features (posting patterns)\npost_time_features AS (\n    SELECT \n        post_id, \n        JULIANDAY(timestamp) - LAG(JULIANDAY(timestamp)) \n            OVER (ORDER BY timestamp) AS days_since_last_post, -- Gap since previous post\n        JULIANDAY(timestamp) - JULIANDAY(FIRST_VALUE(timestamp)\n            OVER(ORDER BY timestamp ASC)) AS days_since_first_post, -- Gap since first post in dataset        \n        CASE \n            WHEN STRFTIME('%w', timestamp) = '0' THEN 'Sunday'\n            WHEN STRFTIME('%w', timestamp) = '1' THEN 'Monday'\n            WHEN STRFTIME('%w', timestamp) = '2' THEN 'Tuesday'\n            WHEN STRFTIME('%w', timestamp) = '3' THEN 'Wednesday'\n            WHEN STRFTIME('%w', timestamp) = '4' THEN 'Thursday'\n            WHEN STRFTIME('%w', timestamp) = '5' THEN 'Friday'\n            ELSE 'Saturday' \n        END AS day_of_week, -- Day of week posted\n        CASE \n            WHEN CAST(STRFTIME('%H', timestamp) AS INT) BETWEEN 6 AND 11 THEN 'Morning'\n            WHEN CAST(STRFTIME('%H', timestamp) AS INT) BETWEEN 12 AND 16 THEN 'Afternoon'\n            WHEN CAST(STRFTIME('%H', timestamp) AS INT) BETWEEN 17 AND 20 THEN 'Evening'\n            ELSE 'Night'\n        END AS time_of_day -- Time of day posted\n    FROM\n        posts_with_insights\n),\n\n-- Step 4: Compute engagement-related metrics\npost_engagement_metrics AS (\n    SELECT \n        post_id, \n        like_count + comments_count AS engagement, -- Total engagements\n        ROUND(\n            (like_count + comments_count) * 100 / NULLIF(reach, 0),\n            2) AS engagement_per_reach_pct, -- Engagement as % of reach\n        ROUND(\n            AVG(like_count + comments_count) \n            OVER (ORDER BY timestamp ROWS BETWEEN 2 PRECEDING AND CURRENT ROW))\n            AS rolling_avg_engagement_3posts, -- Rolling average engagement\n        ROUND(\n            AVG(reach)\n            OVER (ORDER BY timestamp ROWS BETWEEN 2 PRECEDING AND CURRENT ROW))\n            AS rolling_avg_reach_3posts, -- Rolling average reach\n        RANK() \n            OVER (PARTITION BY media_type ORDER BY (like_count + comments_count) DESC) \n            AS engagement_rank_by_media_type -- Group by media type, rank by post engagement\n    FROM \n        posts_with_insights\n)\n\n-- Step 5: Combine all features into final view\nSELECT \n    pwi.*,\n    ptf.days_since_last_post,\n    ptf.days_since_first_post,\n    ptf.day_of_week,\n    ptf.time_of_day,\n    pem.engagement,\n    pem.rolling_avg_engagement_3posts,\n    pem.rolling_avg_reach_3posts,\n    pem.engagement_per_reach_pct,\n    pem.engagement_rank_by_media_type\nFROM \n    posts_with_insights pwi\nLEFT JOIN \n    post_time_features ptf ON pwi.post_id = ptf.post_id\nLEFT JOIN \n    post_engagement_metrics pem ON pwi.post_id = pem.post_id\nORDER BY timestamp DESC\n")


# ### Load SQL View Into DataFrame
# >
# >- Use `%sql` to query `ig_post_metrics` and convert results to a DataFrame (`ig_metrics_df`) for further analysis.
# >- Once data is in pandas, we close the database connection.

# In[ ]:


# Query the engineered SQL view into a pandas DataFrame for analysis
ig_post_metrics = get_ipython().run_line_magic('sql', 'SELECT * FROM ig_post_metrics')
ig_metrics_df = ig_post_metrics.DataFrame()

# Close the SQLite connection (not needed once data is in DataFrame)
conn.close()


# ## Data Overview & Cleaning
# >
# >- Display the DataFrame structure, column data types, and non-null counts.
# >- Generate summary statistics for both numeric and categorical columns.
# >- Identify the number of missing values in each column.
# >- Replace `NaN` values with 0 in selected columns (`caption_length`, `num_hashtags`, `days_since_last_post`) before analysis.

# In[14]:


# Display summary of DataFrame structure, data types, and non-null counts
ig_metrics_df.info()


# In[11]:


# Descriptive statistics for all columns (numeric & categorical)
ig_metrics_df.describe(include = 'all')


# In[15]:


# Check the number of missing values in each column of the DataFrame
ig_metrics_df.isnull().sum()


# In[16]:


# Replace NaN values with 0 in selected columns to handle missing data before analysis
cols = ['caption_length', 'num_hashtags', 'days_since_last_post']
ig_metrics_df[cols] = ig_metrics_df[cols].fillna(0)


# ## Exploring Patterns in Instagram Reach & Engagement
# > In this section, we perform **exploratory data analysis (EDA)** to uncover patterns, relationships, and potential drivers of post performance. The analysis evaluates a range of features including reach, engagement metrics, posting time, caption length, and media type to **better understand how different factors influence content outcomes**.

# ### Correlation Analysis of Post Metrics
# > To identify relationships between numeric features, we computed a **correlation matrix** (excluding `post_id`) and visualized it with a heatmap. This helps highlight **which variables are strongly associated with each other** and may influence post performance.

# In[ ]:


# Compute correlation matrix for all numeric features, excluding post_id
ig_corr = (
    ig_metrics_df
        .select_dtypes(include = ['int64', 'float64'])
        .drop(columns = 'post_id')
        .corr()
)

# Initialize figure
fig = plt.figure(figsize = (15, 8))

# Plot heatmap of correlations
sns.heatmap(
    data = ig_corr, cmap = 'vlag', annot = True, linecolor = 'black', 
    linewidths = 0.5, fmt = '.2f', cbar_kws = {'label': 'Correlation Coefficient'}
)

# Set figure title 
fig.suptitle('Correlation Heatmap of Instagram Post Metrics', fontweight = 'bold', fontsize = 18)

fig.tight_layout()
plt.savefig('corr_heatmap.png')


# >Since there are many features and corresponding correlation coefficients, we won’t discuss each individually. For this analysis, we focus on the features most relevant to our objectives and hypotheses.
# >
# >**Engagement:**
# >- The **engineered features** (e.g., `caption_length`, `days_since_first_post`) **show no strong positive or negative correlations** with `engagement` in the heatmap. This is somewhat unexpected, particularly for `days_since_first_post`, which might reasonably be assumed to correlate positively with `engagement` as an account’s follower base grows over time.
# >- Excluding features that are direct linear combinations of `engagement`, the strongest correlation appears with `ig_reels_video_view_total_time`, which shows a coefficient of roughly 0.90.
# >
# >**Reach:**
# >- Like `engagement`, `reach` shows **generally weak correlations with most engineered features**. However, unlike `engagement`, `reach` exhibits a **moderately strong positive correlation with** `days_since_first_post`, which aligns with expectations: as an account gains followers over time, the potential audience for each post naturally increases.
# >- It’s worth noting that `engagement` and `reach` have a fairly strong positive correlation, as expected: **greater post exposure naturally leads to higher engagement**, and posts with higher engagement are more likely to be boosted by the Instagram algorithm.

# ![corr_heatmap.png](attachment:16b69532-c8f0-446d-a549-a7ba83d7f951.png)

# ### Engagement & Reach Across Quantitative Features
# >As noted earlier, the correlation heatmap shows mostly weak relationships between **quantitative engineered features**, such as `days_since_last_post`, `caption_length`, and `num_hashtags`, and both `engagement` and `reach`. To further examine these relationships, we use scatter plots paired with **Pearson correlation coefficients** and their respective **p-values**, providing a clearer view of how these features relate to post performance. **For the following analyses, the predetermined level of significance ($\alpha$) is 0.05.**

# In[ ]:


# Set Seaborn style for figures 
sns.set_style("whitegrid")

# Define a color palette to be used in the following figures
palette = sns.color_palette("muted")


# In[ ]:


def get_reach_engagement_quant(dataframe, feature, xlabel):
    # Create side-by-side subplots: one for engagement, one for reach
    fig, ax = plt.subplots(1, 2, figsize = (15, 8))

    # --- Engagement vs Feature ---
    # Compute Pearson correlation & p-value
    engagement_pearson_coeff, engagement_p_value = stats.pearsonr(x = dataframe[feature], y = dataframe['engagement'])

    # Scatter plot of feature vs. engagement
    ax[0].scatter(dataframe[feature], dataframe['engagement'], color = palette[4])

    # Annotate correlation values	
    # transform = ax[i].transAxes sets the coordinates relative to the axes
    # verticalalignment = 'top' makes the text start from the top, so it doesn’t overlap the axes.
    ax[0].text(
        0.45, 0.95, f'Engagment & {xlabel}:\npearson_coeff = {engagement_pearson_coeff}\np_value = {engagement_p_value}', 
        transform = ax[0].transAxes, verticalalignment = 'top', fontweight = 'bold'
    )

    ax[0].set_xlabel(xlabel, fontsize = 12)
    ax[0].set_ylabel('Engagement (Likes + Comments)', fontsize = 12)
    ax[0].set_title(f'Engagement vs. {xlabel}', fontsize = 18, fontweight = 'bold')

    # --- Reach vs Feature ---
    # Drop rows with null reach values before computing correlation
    ig_reach = dataframe[dataframe['reach'].notnull()]
    reach_pearson_coeff, reach_p_value = stats.pearsonr(x = ig_reach[feature], y = ig_reach['reach'])

    # Scatter plot of feature vs. reach
    ax[1].scatter(dataframe[feature], dataframe['reach'], color = palette[3])

    # Annotate correlation values
    # transform = ax[i].transAxes sets the coordinates relative to the axes
    # verticalalignment = 'top' makes the text start from the top, so it doesn’t overlap the axes.
    ax[1].text(
        0.45, 0.95, f'Reach & {xlabel}:\npearson_coeff = {reach_pearson_coeff}\np_value = {reach_p_value}',
        transform = ax[1].transAxes, verticalalignment = 'top', fontweight = 'bold'
    )

    ax[1].set_xlabel(xlabel, fontsize = 12)
    ax[1].set_ylabel('Reach (Unique Accounts)', fontsize = 12)
    ax[1].set_title(f'Reach vs. {xlabel}', fontsize = 18, fontweight = 'bold')

    fig.tight_layout()
    plt.savefig(f'{feature}_plots.png')
    plt.close()


# In[ ]:


# Generate scatter plots for days_since_last_post vs. engagement & reach
get_reach_engagement_quant(ig_metrics_df, 'days_since_last_post', 'Days Since Last Post')


# >**Engagment & Reach vs. Days Since Last Post**
# >- `days_since_last_post` measures the time gap (in days) between a post and the one immediately before it.
# >- Scatter plots and Pearson correlation coefficients indicate **a weak positive correlation** between `days_since_last_post` and both `engagement` and `reach`. 
# >- For engagement, $p < \alpha = 0.05$ indicates that we **reject the null hypothesis** of no correlation and conclude that `days_since_last_post` & `engagement` **are, in fact, correlated**.
# >- For reach, $p > \alpha = 0.05$ indicates that we **fail to reject the null hypothesis** of no correlation, and no definitive conclusions can be made about the correlation between `days_since_last_post` & `reach`. 

# ![eng_reach_days_since_last.png](attachment:fb865776-503e-4783-a0e3-44067ccb16ba.png)

# In[ ]:


# Generate scatter plots for caption_length vs. engagement & reach
get_reach_engagement_quant(ig_metrics_df, 'caption_length', 'Caption Length')


# >**Engagment & Reach vs. Caption Length**
# >- `caption_length` represents the number of characters in a post’s caption.
# >- Scatter plots and Pearson correlation results suggest **little to no statistical correlation** between `caption_length` & `engagement` and a **weak negative correlation** between `caption_length` & `reach`.
# >- However, for both tests, $p > \alpha = 0.05$, so **we fail to reject the null hypothesis** and therefore cannot conclude anything about the correlation between `caption_length` and both `engagement` & `reach`. 

# ![eng_reach_caption_length.png](attachment:b0daf244-2d05-4205-b53b-493e4fe83838.png)

# **Engagement & Reach vs. Number of Hashtags**
# > Descriptive statistics in the previous section indicate that the **interquartile range (IQR) for** `num_hashtags` **is 0**. This suggests a need for **further analysis on the usage of hashtags** and their impact (or lack thereof) on post `engagement` and `reach`.

# In[31]:


# Quick check of the distribution of hashtag usage (how many posts use 0, 1, 2, etc.)
(
    ig_metrics_df['num_hashtags']
        .astype('int64')
        .value_counts()
        # Rename the Series index (its only axis) so it becomes a
        # meaningful column label after reset_index()
        # DataFrames have two axes (index, columns)
        .rename_axis('Number of Hashtags')
        .reset_index(name = 'Post Counts')
)


# In[33]:


# Show performance metrics for posts that include at least one hashtag
hash_columns = [
    'post_id', 'media_type', 'num_hashtags', 'engagement', 'reach', 
    'rolling_avg_engagement_3posts', 'rolling_avg_reach_3posts'
]

(
    ig_metrics_df
        .loc[~ig_metrics_df['num_hashtags'].eq(0), hash_columns]
        .reset_index(drop = True)
)


# >- The table indicates that **posts with hashtags** generally achieve **higher** `engagement` compared to the rolling 3-post average (`rolling_avg_engagement_3posts`).
# >- The single hashtagged post with available `reach` data performed **significantly better** than the corresponding rolling 3-post average (`rolling_avg_reach_3posts`).
# >- However, since only one post in the dataset contains hashtags, the **sample size is too limited for strong conclusions**; these results should be **viewed as suggestive rather than definitive**.

# ### Engagement & Reach Trends Over Time
# > 
# > This section examines Instagram post performance over time, focusing on `engagement`, `reach`, and `engagement` efficiency (`engagement_per_reach_pct`). We analyze trends, distributions, and correlations to uncover relationships between key metrics and temporal patterns. A key question we aim to address is how `engagement` and `reach` move together—for instance, whether `engagement_per_reach_pct` remains stable as `reach` fluctuates over time.

# In[ ]:


# Initialize figure with two subplots
fig, ax = plt.subplots(2, 1, figsize = (15, 8))

# Pearson correlation and p-value between engagement and time
engagement_pearson_coeff, engagement_p_value = stats.pearsonr(x = ig_metrics_df['days_since_first_post'], y = ig_metrics_df['engagement'])

# Plot raw engagement over time 
ax[0].plot(ig_metrics_df['days_since_first_post'], ig_metrics_df['engagement'], color = palette[1])

# Rolling average engagement (smoother trendline)
ax[0].plot(ig_metrics_df['days_since_first_post'], ig_metrics_df['rolling_avg_engagement_3posts'], color = palette[0])

# Add legend
ax[0].legend(labels = ['engagement', 'rolling_avg_engagement_3posts'], fontsize = 8)

ax[0].fill_between(ig_metrics_df['days_since_first_post'], 0, ig_metrics_df['engagement'], alpha = .3, color = palette[1])
ax[0].fill_between(ig_metrics_df['days_since_first_post'], 0, ig_metrics_df['rolling_avg_engagement_3posts'], alpha = .3, color = palette[0])

ax[0].set_xlabel('Days Since First Post', fontsize = 12)
ax[0].set_ylabel('Engagement', fontsize = 12)
ax[0].set_title('Instagram Engagement Over Time', fontweight = 'bold', fontsize = 18)

# Pearson correlation annotation
ax[0].text(50, 80, f'Engagement & Days Since First Post:\npearson_coeff = {engagement_pearson_coeff}\np_value = {engagement_p_value}', fontweight = 'bold')

# Remove missing values before correlation
ig_reach = ig_metrics_df[ig_metrics_df['reach'].notnull()][['days_since_first_post', 'reach']]

# Pearson correlation and p-value between reach and time
reach_pearson_coeff, reach_p_value = stats.pearsonr(x = ig_reach['days_since_first_post'], y = ig_reach['reach'])

# Plot reach over time
ax[1].plot(ig_metrics_df['days_since_first_post'], ig_metrics_df['reach'], color = palette[3])

# Rolling average reach (smoother trendline)
ax[1].plot(ig_metrics_df['days_since_first_post'], ig_metrics_df['rolling_avg_reach_3posts'], color = palette[0])

# Add legend
ax[1].legend(labels = ['reach', 'rolling_avg_reach_3posts'], fontsize = 8)

ax[1].fill_between(ig_metrics_df['days_since_first_post'], 0, ig_metrics_df['reach'], alpha = .3, color = palette[3])
ax[1].fill_between(ig_metrics_df['days_since_first_post'], 0, ig_metrics_df['rolling_avg_reach_3posts'], alpha = .3, color = palette[0])

ax[1].set_title('Instagram Reach Over Time', fontweight = 'bold', fontsize = 18)
ax[1].set_xlabel('Days Since First Post', fontsize = 12)
ax[1].set_ylabel('Reach (Unique Accounts)', fontsize = 12)

# Pearson correlation annotations
ax[1].text(780, 3300, f'Reach & Days Since First Post:\npearson_coeff = {reach_pearson_coeff}\np_value = {reach_p_value}', fontweight = 'bold')

# Vertical line showing when reach data begins
reach_days_min = ig_metrics_df[ig_metrics_df['reach'].notnull()].agg({'days_since_first_post': 'min'}).values[0]
ax[0].axvline(x = reach_days_min, color = palette[3], linestyle = '--')
ax[0].annotate(
    'beginning of reach data', xy = (reach_days_min, 85), xytext = (reach_days_min + 50, 85), 
    arrowprops = dict(facecolor = 'black', shrink = 0.05), fontweight = 'bold'
)

fig.tight_layout()
plt.savefig('engagement_reach_time.png')


# >**Engagement & Reach Over Time:**
# >
# >- We visualize how raw `engagement`, rolling average engagement (`rolling_avg_engagement_3posts`), and `reach` change as the account ages. Pearson correlation coefficients quantify the relationship between `days_since_first_post` and each metric.
# >
# >- It appears that `engagement` **does not show any clear trend over time**, and the `rolling_avg_engagement_3posts` **fails to smooth the series** in a way that reveals actionable insights about changes in `engagement`. On the other hand, the Pearson coefficient indicates a **weak positive correlation** between `engagement` and `days_since_first_post`. However, the **p-value is greater than the level of significance**, which suggests that **we cannot reject the null hypothesis of no correlation**.
# >
# >- In the **Instagram Engagement Over Time** plot, we annotate with a red line marking where `reach` data collection begins. This reflects the account’s transition to a "Creator" account, which provided access to advanced metrics. As a result, only the most recent 35 posts include recorded `reach` data.
# >
# >- Although the `reach` trendline is **not strictly linear**, it shows a **gradual upward trajectory** over time. The `rolling_avg_reach_3posts` provides a smoother and more interpretable view of this trend. Furthermore, the Pearson coefficient and respective p-value indicate a **moderately strong correlation** between `reach` and `days_since_first_post` that is **statistically reliable**. In other words, **we can reject the null hypothesis and conclude that the two features are correlated**.
# >
# >- `Engagement` **does not appear to consistently increase alongside** `reach` **over time**. While we might expect higher `reach` to drive higher `engagement`, the data does not clearly support this. In the next section, we take a closer look at the relationship between `reach` and `engagement_per_reach_pct` to gain a deeper understanding of how these metrics interact.

# ![engagement_reach_time.png](attachment:b72f98a2-04bb-4e89-993c-06d6468242ca.png)

# In[ ]:


# Initialize figure with two subplots
fig, ax = plt.subplots(1, 2, figsize = (15, 8))

# Plot boxplot for engagement per reach (%)
sns.boxplot(
	data = ig_metrics_df, y = 'engagement_per_reach_pct', ax = ax[0], 
	color = palette[4], flierprops = {'mfc' : 'black', 'marker': 'D'}
)

ax[0].set_title('Engagement per Reach (%) Distribution', fontsize = 18, fontweight = 'bold')
ax[0].set_ylabel('Engagement / Reach (%)', fontsize = 12)
ax[0].set_xlabel('')  # No x-axis label needed

# Plot boxplot for reach
sns.boxplot(
	data = ig_metrics_df, y = 'reach', ax = ax[1], 
	color = palette[3], flierprops = {'mfc' : 'black', 'marker': 'D'}
)

ax[1].set_title('Reach Distribution', fontsize = 18, fontweight = 'bold')
ax[1].set_ylabel('Number of Accounts Reached', fontsize = 12)
ax[1].set_xlabel('')  # No x-axis label needed

plt.tight_layout()
plt.savefig('reach_engagement_rate_box.png')


# > **Distribution of Engagement Rate & Reach:**
# >
# >- Next, we will plot `engagement_per_reach_pct` and `reach` over time (`days_since_first_post`) to better understand how `reach` impacts `engagement` efficiency.
# >
# >- Since these metrics are on very different scales, we first **need to standardize them**. Two common approaches are **StandardScaler** and **RobustScaler**, both available in `sklearn`. The choice depends on the shape of each feature’s distribution.
# >
# >- In our case, `engagement_per_reach_pct` is **fairly right-skewed**, while `reach` is **relatively symmetric** but includes a **strong outlier**. Because extreme values can distort scaling when using the mean, we opt for **RobustScaler**, which leverages the median and interquartile range (IQR) for a more robust transformation.

# ![reach_engagement_rate_box.png](attachment:15d30fbe-f5be-457e-bba2-2945a822b717.png)

# In[ ]:


# Initialize figure 
fig = plt.figure(figsize = (15, 8))

# Scale engagement_per_reach_pct and reach using RobustScaler to make them comparable
scaler = RobustScaler()
ig_metrics_df[['scaled_engagement_rate', 'scaled_reach']] = (
    scaler.fit_transform(ig_metrics_df[['engagement_per_reach_pct', 'reach']])
)

# Plot scaled reach over time
plt.plot(ig_metrics_df['days_since_first_post'], ig_metrics_df['scaled_reach'], color = palette[3])

# Plot scaled engagement rate over time
plt.plot(ig_metrics_df['days_since_first_post'], ig_metrics_df['scaled_engagement_rate'], color = palette[4])

# Add legend
plt.legend(labels = ['scaled_reach', 'scaled_engagement_rate'], fontsize = 8)

plt.fill_between(ig_metrics_df['days_since_first_post'], 0, ig_metrics_df['scaled_reach'], alpha = .3, color = palette[3])
plt.fill_between(ig_metrics_df['days_since_first_post'], 0, ig_metrics_df['scaled_engagement_rate'], alpha = .3, color = palette[4])

plt.xlabel('Days Since First Post', fontsize = 12)
plt.ylabel('Scaled Metric Value', fontsize = 12)
plt.title('Comparison of Reach and Engagement per Reach Over Time', fontweight = 'bold', fontsize = 18)

# Select only rows where 'reach' is not null and keep relevant columns
ig_reach_engagement = ig_metrics_df[ig_metrics_df['reach'].notnull()][['engagement_per_reach_pct', 'reach']]

# Pearson correlation coefficient and p-value between reach and engagement per reach
reach_engagement_pearson_coeff, reach_engagement_p_value = (
    stats.pearsonr(x = ig_reach_engagement['reach'], y = ig_reach_engagement['engagement_per_reach_pct'])
)

# Pearson correlation annotation
plt.text(780, 4, f'Reach & Engagement per Reach (%):\npearson_coeff = {reach_engagement_pearson_coeff}\np_value = {reach_engagement_p_value}', fontweight = 'bold')

plt.savefig('reach_eng_rate_trend.png')

# Drop the temporary scaled columns from the dataframe
ig_metrics_df.drop(columns = ['scaled_engagement_rate', 'scaled_reach'])


# >**Comparing Reach & Engagement Efficiency Over Time:**
# >
# >The plot shows that as `reach` grows, `engagement_per_reach_pct` declines. The Pearson correlation coefficient confirms this trend, and the p-value (< 0.05) indicates the **relationship is statistically significant**, leading us to **reject the null hypothesis**. In other words, **broader exposure to posts does not necessarily translate into proportionally higher** `engagement`. The pattern—higher `engagement_per_reach_pct` when `reach` is low, followed by a decline as `reach` expands—could imply that **posts are either losing their overall engaging quality over time**, or that **new audiences being reached are less responsive than the original core audience**, which remains consistently engaged.

# ![reach_eng_rate_trend.png](attachment:af93a3c2-8c6e-4843-917d-16f49b56d303.png)

# ### Engagement & Reach Across Qualitative Features
# >To better understand performance patterns, we examine how `reach` and `engagement` vary with different **qualitative (categorical) features** such as:  
# >- `media_type`
# >- `time_of_day`
# >- `day_of_week`
# >
# >By plotting these relationships, we can identify whether certain posting strategies are associated with higher exposure and interaction.  For example, **certain media types may consistently gain broader visibility or generate more likes/comments** than others. These insights may help us evaluate not just overall performance, but also the **drivers** behind changes in `reach` and `engagement`.
# >
# >As mentioned earlier, because the account was not set as a "Creator" for a period of time and therefore lacked access to advanced metrics like `reach`, the following figures include two sets of bar charts: one showing the number of posts for each feature using the full `engagement` dataset (all posts), and another showing the number of posts using the `reach` subset (the most recent 35 posts).
# >
# >Because the `reach` **data is limited** to just 35 posts, the following boxplots should be **viewed as indicative rather than definitive** of long-term performance trends.
# >
# >All figures in this section are generated using the `get_reach_engagement` function.

# In[ ]:


def get_reach_engagement_qual(dataframe, feature, xlabel, ax10_title, ax11_title):
    # Count how many posts fall into each category of the chosen feature
    # NOTE: engagement is recorded for all ~100 posts, but reach is only available for 35 posts.
    # To make fair comparisons, we generate separate count distributions for engagement and reach.
    eng_feature_counts = (
        dataframe
            .groupby(feature, as_index = False)
            .agg(**{'number_of_posts': ('post_id', 'count')})
    )

    reach_feature_counts = (
        dataframe
            .loc[dataframe['reach'].notnull()]
            .groupby(feature, as_index = False)
            .agg(**{'number_of_posts': ('post_id', 'count')})
    )

    # Alphabetically order feature labels (helps align plots consistently)
    eng_ordered_features = sorted(eng_feature_counts[feature].unique())
    reach_ordered_features = sorted(reach_feature_counts[feature].unique())

    # Initialize figure 
    fig, ax = plt.subplots(2, 2, figsize = (15, 8))

    # Set up Seaborn color palettes to be used in the following plots
    palette0 = sns.color_palette("deep")
    palette1 = sns.color_palette("viridis")

    # Bar chart of post counts by feature (Engagement dataset, ~100 posts)
    ax[0, 0].bar(eng_feature_counts[feature], eng_feature_counts['number_of_posts'], color = palette0)
    ax[0, 0].set_xlabel(xlabel, fontsize = 12)  # e.g., "Media Type" or "Day of Week"
    ax[0, 0].set_ylabel('Number of Posts (All Engagement Data)', fontsize = 12)
    ax[0, 0].set_title('Post Counts by ' + xlabel + ' (Engagement Dataset)', fontsize = 18, fontweight = 'bold')

    # Boxplot of engagement by feature 
    sns.boxplot(
        data = dataframe, x = feature, y = 'engagement', 
        palette = palette0, ax = ax[1, 0], order = eng_ordered_features
    )

    ax[1, 0].set_xlabel(xlabel, fontsize = 12)
    ax[1, 0].set_ylabel('Engagement (Likes + Comments)', fontsize = 12)
    ax[1, 0].set_title(ax10_title, fontsize = 18, fontweight = 'bold')

    # Bar chart of post counts by feature (Reach dataset, 35 posts)
    ax[0, 1].bar(reach_feature_counts[feature], reach_feature_counts['number_of_posts'], color = palette1)
    ax[0, 1].set_xlabel(xlabel, fontsize = 12)  # e.g., "Media Type" or "Day of Week"
    ax[0, 1].set_ylabel('Number of Posts (Reach Subset)', fontsize = 12)
    ax[0, 1].set_title('Post Counts by ' + xlabel + ' (Reach Dataset)', fontsize = 18, fontweight = 'bold')

    # Boxplot of reach by feature 
    sns.boxplot(
        data = dataframe, x = feature, y = 'reach', 
        palette = palette1, ax = ax[1, 1], order = reach_ordered_features
    )

    ax[1, 1].set_xlabel(xlabel, fontsize = 12)
    ax[1, 1].set_ylabel('Reach (Unique Accounts)', fontsize = 12)
    ax[1, 1].set_title(ax11_title, fontsize = 18, fontweight = 'bold')

    # Adjust layout to prevent overlap
    fig.tight_layout()

    plt.savefig(f'{feature}_plots.png')
    plt.close()


# In[ ]:


# Generate bar charts & boxplots for engagement & reach by media_type
get_reach_engagement_qual(ig_metrics_df, 'media_type', 'Media Type', 'Engagement Distribution By Media Type', 'Reach Distribution By Media Type')


# >**Engagement & Reach by Media Type**
# >
# >- For all posts, `CAROUSEL_ALBUM` and `IMAGE` are the most common media types.
# >- Within the smaller `reach` subset, `IMAGE` and `VIDEO` are most represented.
# >- The `engagement` boxplot shows that `VIDEO` posts generally achieve **higher engagement than other types**.
# >- The `reach` boxplot shows that `VIDEO` and `IMAGE` **achieve comparable performance**, while `CAROUSEL_ALBUM` lags behind. This is somewhat unexpected, as **Instagram’s algorithm typically prioritizes** `VIDEO` **content** for discovery, which would suggest higher reach compared to `IMAGE`.

# ![media_type_plots.png](attachment:8ee76308-9833-496e-8c28-17bb6570355a.png)

# In[ ]:


# Generate bar charts & boxplots for engagement & reach by time_of_day
get_reach_engagement_qual(ig_metrics_df, 'time_of_day', 'Time of Day', 'Engagement Distribution By Time of Day', 'Reach Distribution By Time of Day')


# > **Engagement & Reach by Time of Day**
# >- The `time_of_day` feature represents when a post was published. To simplify analysis, we grouped posting times into four bins: `Morning`, `Afternoon`, `Evening`, and `Night`, as 24 distinct hours would be too granular for 100 posts.
# >- Notably, **no posts were published during the** `Evening` **bin**.
# >- Both the `engagement` and `reach` bar charts show that most posts occurred in the `Morning`, followed by `Night`, with `Afternoon` being the least represented.
# >- The `engagement` boxplot indicates that `Afternoon` posts have the highest median `engagement`, followed by `Morning` and `Night`, but the differences are minor, suggesting that `time_of_day` **has little impact on** `engagement`.
# >- Similarly, the `reach` boxplot display fairly symmetric distributions with comparable medians across bins, implying that `reach` **is also not strongly influenced by posting time**.

# ![time_of_day_plots.png](attachment:d8bc15ef-5827-4b3b-a4f5-e0e0a126ae55.png)

# In[ ]:


# Generate bar charts & boxplots for engagement & reach by day_of_week
get_reach_engagement_qual(ig_metrics_df, 'day_of_week', 'Day of Week', 'Engagement Distribution By Day of Week', 'Reach Distribution By Day of Week')


# > **Engagement & Reach by Day of Week**
# >- The `day_of_week` feature indicates the day a post was published, ranging from `Monday` to `Sunday`, similar to `time_of_day`.
# >- Bar charts for both `engagement` and `reach` show that most posts were published on `Wednesday` and `Friday`.
# >- The `engagement` boxplot suggests that `engagement` tends to be highest on `Monday` and `Sunday`. This **aligns with our expectations** given that while the company posts from Tokyo, a large portion of followers are in the U.S., so posting on `Monday` or `Sunday` in Tokyo corresponds to the U.S. weekend, when **users are likely more active on social media**, as expected.
# >- The `reach` boxplot shows no single day clearly outperforming others. However, it’s worth noting that although `Thursday` and `Friday` account for a large share of posts, their median `reach` values are actually lower than the lower quartiles of all other days.

# ![day_of_week_plots.png](attachment:7bc3bca7-5e1c-47c0-bbf9-143807e827d2.png)

# ## Top Performing Posts
# > We filter the dataset by `engagement_rank_by_media_type` to select the **top 3 posts** within each `media_type` (`IMAGE`, `CAROUSEL_ALBUM`, `VIDEO`). These posts represent the highest engagement levels relative to their category.  
# >
# > Highlighting top performers is useful for deciding which posts to **Boost** on Instagram (i.e., promote via paid advertising) to maximize reach and audience engagement. 

# In[34]:


# Select the top 3 posts for each media_type (`engagement_rank_by_media_type` < 4)
# Sort them first by rank, then by media_type for readability
(
    ig_metrics_df
        .loc[ig_metrics_df['engagement_rank_by_media_type'].lt(4)]
        .sort_values(by = ['engagement_rank_by_media_type', 'media_type'])
)


# ## Export Final Metrics Table 
# > 1. **Export DataFrame to CSV**: Save the cleaned and processed DataFrame to a CSV file without including the index.
# > 2. **Preview the Final Table**: Display the first ten rows of the DataFrame

# In[4]:


# Save DataFrame to CSV file
ig_metrics_df.to_csv('ig_post_metrics.csv', index = False)

# View the final table
ig_metrics_df.head(10)


# ## Key Insights & Recommendations
# >
# > **1. Engagement & Reach Over Time**
# >
# >While `reach` shows a gradual upward trend, `engagement` has not followed the same pattern. This may indicate that **newer audiences are less engaged than the account’s original followers**, or that posts are generally becoming less engaging over time.
# >
# > **2. Quantitative Features & Performance**
# >
# >- There **does not seem to be any significant relationship** (whether positive or negative) between `caption_length` & `days_since_last_post` and `engagement`.
# >- There is a **moderate negative correlation** between `caption_length` and `reach`; however, the p-value suggests this relationship may not be statistically significant, and **we cannot reject the null hypothesis**. Alternatively, it may reflect that audiences are **less inclined to engage with longer captions**—particularly on reels—leading to shorter viewing times (`ig_reels_video_view_total_time`) and reduced algorithmic promotion. We recommend **exploring the use of shorter, more concise captions**, particularly for `VIDEO` type posts.
# >- Although the sample size is very small, the use of **hashtags does appear to be linked to both higher** `engagement` & `reach`. We recommend **increasing both the number and diversity of hashtags** used, and then comparing the performance of these posts against those without hashtags.
# >
# > **3. Qualitative Features & Performance**
# >
# >- The results **do not show any significantly link** between `media_type` and `reach`. While `CAROUSEL_ALBUM` posts appear to underperform relative to `IMAGE` and `VIDEO`, this difference **may simply reflect the smaller sample size** of `CAROUSEL_ALBUM` posts.
# >- On average, `VIDEO` posts generate **higher levels of** `engagement` compared to other media types. We recommend **prioritizing** `VIDEO` **content** to maximize audience engagement.
# >- The findings suggest that `time_of_day` **has little impact on both** `engagement` & `reach`.
# >- **Consistent with our expectations**, posts published on `Sunday` and `Monday`—which align with `Saturday` and `Sunday`, respectively, in the U.S., where a large portion of our audience is based—**tend to generate higher** `engagement`. We recommend **prioritizing these days** for future posting.
# >
# > **4. Boosting Top Performing Posts**
# >
# >We recommend leveraging Instagram’s paid promotion feature to **Boost the top 3 posts** (ranked by `engagement`) in each `media_type` category to **increase audience reach and interaction**.

# ## Limitations & Future Directions
# >
# >**1. Limited Dataset**
# >
# >This analysis is based on roughly 100 posts, with only about 35 containing advanced metrics such as reach, views, and ig_reels_video_view_total_time. Given the small sample size, the ability to draw robust conclusions about feature relationships and post performance is limited. The findings should be interpreted as indicative patterns rather than definitive results. As additional post data is collected over time, content performance trends should become easier to analyze and interpret.
# >
# >**2. Lack of Advanced Metrics**
# >
# >Post performance is influenced by numerous factors, many of which fall outside the scope of this analysis. Elements such as visual quality, caption sentiment, and audience demographics can directly affect both engagement and reach. Future analyses could incorporate more advanced techniques—for example, applying NLP to evaluate caption sentiment or using regression models to predict performance.
# >
# >**3. Algorithmic Complexity**
# >
# >Social media algorithms are highly complex in determining post exposure, and audience engagement is equally difficult to quantify, as it depends on numerous unpredictable and unmeasurable factors. The advanced modeling techniques discussed earlier could help account for unmeasured complexities and enable a more robust analysis of overall post performance.

# ## Conlusion
# >This analysis offers insights into Instagram post performance, revealing that reels and weekend publishing—aligned with peak U.S. follower activity—tend to generate higher engagement. While reach has gradually increased over time, engagement has not followed the same pattern, suggesting newer audiences may be less engaged or content resonates primarily with long-term followers. Quantitative features like caption length and posting frequency showed limited impact, though hashtags appear to positively influence both engagement and reach. Leveraging Instagram’s Boost feature on top-performing posts could further expand audience exposure and interaction.
# >
# >This study is constrained by a small dataset, missing advanced metrics, and the inherent complexity of social media algorithms. Future analyses should incorporate larger datasets and advanced modeling techniques, including NLP for captions and machine learning for performance prediction, to better account for unmeasured factors and provide more robust insights.
