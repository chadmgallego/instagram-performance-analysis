#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import necessary Python modules for API calls, JSON handling, & SQLite
import requests, json, sqlite3, sys
from datetime import datetime
import pandas as pd
import numpy as np


# ## Data Collection Class: `InstaCollector`
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


# ## Data Loading Class: `InstaLoader`
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

